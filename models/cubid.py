"""
CubiD: Cubic Discrete Diffusion for High-Dimensional Representation Tokens
"""

from functools import partial
import math

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from timm.models.vision_transformer import Block


def mask_by_order(mask_len, order, bsz, seq_len):
    """Create a mask based on the specified order and mask length."""
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                          src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking


class CubiD(nn.Module):
    """CubiD model for high-dimensional discrete diffusion."""

    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 quant_levels=32,
                 vae_embed_dim=16,
                 mask_ratio_min=0.0,
                 mask_std=0.1,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 grad_checkpointing=False,
                 quant_min=-5.0,
                 quant_max=5.0,
                 std_range=3.0
                 ):
        super().__init__()
        # Quantization
        self.quant_levels = quant_levels
        self.vae_embed_dim = vae_embed_dim
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.std_range = std_range

        # Model architecture
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # Class embedding for CFG
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # Learnable mask token
        self.mask_value = nn.Parameter(torch.zeros(1))

        # Mask ratio sampler (truncated Gaussian)
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / mask_std, 0, loc=1.0, scale=mask_std)

        # Encoder
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer, proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer, proj_drop=proj_dropout,
                  attn_drop=attn_dropout)
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.output_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_embed_dim * 2, decoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_embed_dim * 2, quant_levels * vae_embed_dim)
        )

        self.initialize_weights()
        self._init_gaussian_quantization()

    def _init_gaussian_quantization(self):
        """Initialize Gaussian-based quantization bins."""
        device = next(self.parameters()).device
        dtype = torch.float32

        probs = torch.linspace(0, 1, self.quant_levels + 1, device=device, dtype=dtype)
        boundaries = torch.tensor(stats.norm.ppf(probs.cpu()), device=device, dtype=dtype)
        boundaries = torch.clamp(boundaries, -self.std_range, self.std_range)

        reconstruction_values = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i+1]
            mean = self._truncated_normal_mean(a, b)
            reconstruction_values.append(mean)

        self.register_buffer('reconstruction_values',
                           torch.tensor(reconstruction_values, dtype=dtype))
        self.register_buffer('boundaries', boundaries)

    def _truncated_normal_mean(self, a, b):
        """Compute mean of truncated normal distribution in [a, b]."""
        sqrt_2 = math.sqrt(2)
        sqrt_2pi = math.sqrt(2 * math.pi)

        phi_a = torch.exp(-0.5 * a**2) / sqrt_2pi
        phi_b = torch.exp(-0.5 * b**2) / sqrt_2pi

        Phi_a = 0.5 * (1 + torch.erf(a / sqrt_2))
        Phi_b = 0.5 * (1 + torch.erf(b / sqrt_2))

        denominator = Phi_b - Phi_a
        denominator = torch.where(denominator == 0,
                                torch.tensor(1e-10, device=a.device, dtype=a.dtype),
                                denominator)

        return (phi_a - phi_b) / denominator

    def initialize_weights(self):
        """Initialize model weights."""
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_value, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.output_pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize linear and layernorm weights."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def quantize(self, x):
        """Quantize continuous values to discrete indices."""
        x_normalized = (x - self.quant_min) / (self.quant_max - self.quant_min) * \
                      (2 * self.std_range) - self.std_range
        x_clamped = x_normalized.clamp(-self.std_range, self.std_range)

        x_expanded = x_clamped.unsqueeze(-1)
        dists = (x_expanded - self.reconstruction_values).abs()
        indices = dists.argmin(dim=-1)

        normalized_values = self.reconstruction_values
        values = (normalized_values + self.std_range) / (2 * self.std_range) * \
                (self.quant_max - self.quant_min) + self.quant_min
        dequant = values[indices]

        return indices, dequant

    def patchify(self, x):
        """Convert image features to patch sequence."""
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        """Convert patch sequence back to image features."""
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_mask_ratio(self):
        """Sample mask ratio from truncated Gaussian."""
        return self.mask_ratio_generator.rvs(1)[0]

    def random_channel_masking(self, continuous_latents):
        """Apply random per-element masking across spatial and channel dimensions."""
        mask_ratio = self.sample_mask_ratio()
        mask = torch.rand_like(continuous_latents) < mask_ratio
        masked_latents = continuous_latents.clone()
        masked_latents[mask] = self.mask_value
        return masked_latents, mask

    def forward_encoder(self, x, mask, class_embedding):
        """Encode masked continuous tokens."""
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([
            torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device),
            x
        ], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + \
                            (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_decoder(self, x):
        """Decode encoder output to predictions."""
        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed_learned

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.output_pos_embed
        return x

    # Keep old names for checkpoint compatibility
    def forward_mae_encoder(self, x, mask, class_embedding):
        return self.forward_encoder(x, mask, class_embedding)

    def forward_mae_decoder(self, x):
        return self.forward_decoder(x)

    def forward(self, imgs, labels):
        """Training forward pass with masked prediction loss."""
        x = self.patchify(imgs)
        continuous_latents = x.clone()

        gt_tokens, quant_values = self.quantize(continuous_latents)
        masked_latents, mask = self.random_channel_masking(quant_values)

        class_embedding = self.class_emb(labels)

        x = self.forward_encoder(masked_latents, mask, class_embedding)
        z = self.forward_decoder(x)

        logits = self.prediction_head(z)
        logits = logits.reshape(z.shape[0], z.shape[1], self.vae_embed_dim, self.quant_levels)

        loss = F.cross_entropy(
            logits[mask].reshape(-1, self.quant_levels),
            gt_tokens[mask].reshape(-1).long()
        )

        return loss

    def sample_orders_3d(self, bsz):
        """Generate random ordering for 3D (spatial+channel) sampling."""
        total_tokens = self.seq_len * self.vae_embed_dim
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(total_tokens)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.tensor(np.array(orders), device='cuda').long()
        return orders

    def mask_by_order_3d(self, mask_len, order, bsz):
        """Create 3D mask based on order."""
        total_tokens = self.seq_len * self.vae_embed_dim
        masking = torch.zeros(bsz, total_tokens, device='cuda')
        masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()],
                            src=torch.ones(bsz, total_tokens, device='cuda')).bool()
        masking = masking.reshape(bsz, self.seq_len, self.vae_embed_dim)
        return masking

    def sample_tokens(self, bsz, num_iter=256, cfg=1.0, cfg_schedule="linear",
                     labels=None, cfg_start_ratio=0., cfg_end_ratio=1.0, temperature=1.0, progress=False):
        """Generate tokens via iterative unmasking."""
        device = next(self.parameters()).device

        continuous_tokens = torch.full((bsz, self.seq_len, self.vae_embed_dim),
                                     self.mask_value.item(), device=device)

        mask = torch.ones(bsz, self.seq_len, self.vae_embed_dim, device=device).bool()
        total_positions = self.seq_len * self.vae_embed_dim
        orders = self.sample_orders_3d(bsz)

        steps = list(range(num_iter))
        if progress:
            steps = tqdm(steps)

        for step in steps:
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            if cfg != 1.0:
                continuous_tokens_cfg = torch.cat([continuous_tokens, continuous_tokens], dim=0)
                class_embedding = torch.cat([
                    class_embedding,
                    self.fake_latent.repeat(bsz, 1)
                ], dim=0)
                mask_cfg = torch.cat([mask, mask], dim=0)
            else:
                continuous_tokens_cfg = continuous_tokens
                mask_cfg = mask

            x = self.forward_encoder(continuous_tokens_cfg, mask_cfg, class_embedding)
            z = self.forward_decoder(x)

            logits = self.prediction_head(z)
            logits = logits.reshape(z.shape[0], z.shape[1], self.vae_embed_dim, self.quant_levels)

            # Cosine schedule
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(total_positions * mask_ratio)], device=device)
            current_masked = mask.sum(dim=(-1, -2))
            mask_len = torch.maximum(
                torch.tensor([1], device=device),
                torch.minimum(current_masked - 1, mask_len)
            )

            mask_next = self.mask_by_order_3d(mask_len[0], orders, bsz)

            if step >= num_iter - 1:
                mask_to_pred = mask
            else:
                mask_to_pred = torch.logical_xor(mask, mask_next)
            mask = mask_next

            if cfg != 1.0:
                mask_to_pred_cfg = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            else:
                mask_to_pred_cfg = mask_to_pred

            # CFG
            if cfg != 1.0:
                if cfg_schedule == "linear":
                    cfg_iter = 1 + (cfg - 1) * (total_positions - mask_len[0]) / total_positions
                elif cfg_schedule == "interval":
                    if cfg_start_ratio <= mask_ratio <= cfg_end_ratio:
                        cfg_iter = cfg
                    else:
                        cfg_iter = 1.0
                elif cfg_schedule == "linear_interval":
                    if cfg_start_ratio <= mask_ratio <= cfg_end_ratio:
                        interval_progress = (cfg_end_ratio - mask_ratio) / (cfg_end_ratio - cfg_start_ratio)
                        cfg_iter = 1 + (cfg - 1) * interval_progress
                    else:
                        cfg_iter = 1.0
                else:
                    cfg_iter = cfg
                logits_cond, logits_uncond = logits.chunk(2, dim=0)
                logits = logits_uncond + cfg_iter * (logits_cond - logits_uncond)

            if mask_to_pred.sum() > 0:
                pred_logits = logits[mask_to_pred_cfg[:bsz]]

                if temperature == 0:
                    pred_indices = pred_logits.argmax(dim=-1)
                else:
                    probs = F.softmax(pred_logits / temperature, dim=-1)
                    pred_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)

                pred_continuous = self.reconstruction_values[pred_indices]
                pred_continuous = (pred_continuous + self.std_range) / (2 * self.std_range) * \
                               (self.quant_max - self.quant_min) + self.quant_min

                continuous_tokens[mask_to_pred] = pred_continuous

        final_tokens = self.unpatchify(continuous_tokens)
        return final_tokens


def cubid_base(**kwargs):
    return CubiD(
        encoder_embed_dim=1536, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1536, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def cubid_large(**kwargs):
    return CubiD(
        encoder_embed_dim=1920, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1920, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)


def cubid_huge(**kwargs):
    return CubiD(
        encoder_embed_dim=3072, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=3072, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
