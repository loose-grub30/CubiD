"""
RAE (Representation Autoencoder) for visual tokenization.
Uses DINOv2 encoder with a ViT decoder for image reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2WithRegistersModel, AutoImageProcessor
from typing import Optional
import numpy as np
from pathlib import Path
import math
from PIL import Image
import argparse

# ============================================
# Position embeddings (same as before)
# ============================================
def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb

# ============================================
# ViT MAE Layer Components
# ============================================
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x_norm = self.norm2(x)
        x_mlp = self.fc2(self.drop(self.act(self.fc1(x_norm))))
        x = x + self.drop(x_mlp)
        return x

# ============================================
# Simple Decoder
# ============================================
class SimpleDecoder(nn.Module):
    def __init__(
        self,
        in_dim=768,
        decoder_dim=1152,
        decoder_depth=28,
        decoder_heads=16,
        mlp_dim=4096,
        patch_size=16,
        image_size=256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.decoder_embed = nn.Linear(in_dim, decoder_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim), requires_grad=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        
        mlp_ratio = mlp_dim / decoder_dim
        self.blocks = nn.ModuleList([
            Block(decoder_dim, decoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        
        self.norm = nn.LayerNorm(decoder_dim, eps=1e-12)
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * 3, bias=True)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], 
            int(self.num_patches**0.5), 
            add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.cls_token, std=0.02)
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = int(self.num_patches**0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward(self, x):
        x = self.decoder_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.decoder_pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        x = self.pred(x)
        x = x[:, 1:, :]  # Remove CLS token
        
        return x

# ============================================
# DINOv2 Encoder with proper normalization
# ============================================
class DINOv2Encoder(nn.Module):
    def __init__(self, model_name='facebook/dinov2-with-registers-base', normalize=True):
        super().__init__()
        self.encoder = Dinov2WithRegistersModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        
        # CRITICAL: Properly disable layernorm affine transformation
        if normalize:
            # This matches official Dinov2withNorm implementation
            self.encoder.layernorm.elementwise_affine = False
            # Set weight and bias to None
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
    
    def forward(self, x):
        # Forward with proper output
        outputs = self.encoder(x, return_dict=True)
        # Remove CLS and register tokens (1 CLS + 4 register tokens)
        z = outputs.last_hidden_state[:, 5:, :]
        return z

# ============================================
# Fixed RAE Model
# ============================================
class RAE(nn.Module):
    def __init__(
        self,
        encoder_name='facebook/dinov2-with-registers-base',
        encoder_size=224,
        decoder_dim=1152,
        decoder_depth=28,
        decoder_heads=16,
        decoder_patch_size=16,
        noise_tau=0.0,  # CRITICAL: Default to 0 for inference
    ):
        super().__init__()
        
        # Initialize encoder
        self.encoder = DINOv2Encoder(encoder_name, normalize=True)
        
        # Get image processor for normalization
        self.image_processor = AutoImageProcessor.from_pretrained(encoder_name)
        
        # Store normalization parameters as buffers
        encoder_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        encoder_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        self.register_buffer('encoder_mean', encoder_mean)
        self.register_buffer('encoder_std', encoder_std)
        
        # Store encoder parameters
        self.encoder_size = encoder_size
        self.encoder_patch_size = self.encoder.patch_size
        self.encoder_hidden_size = self.encoder.hidden_size
        self.num_patches = (encoder_size // self.encoder_patch_size) ** 2
        
        # Noise for training (default 0 for inference)
        self.noise_tau = noise_tau
        
        # Decoder
        decoder_image_size = decoder_patch_size * int(math.sqrt(self.num_patches))
        self.decoder = SimpleDecoder(
            in_dim=self.encoder_hidden_size,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            mlp_dim=4096,
            patch_size=decoder_patch_size,
            image_size=decoder_image_size,
        )
        
        # Latent normalization
        self.register_buffer('latent_mean', torch.zeros(1, 1, 1, 1))
        self.register_buffer('latent_var', torch.ones(1, 1, 1, 1))
        self.normalize_latents = False
        self.eps = 1e-5
        
        print(f"Fixed RAE initialized:")
        print(f"  Encoder: {encoder_name}")
        print(f"  Encoder size: {encoder_size}")
        print(f"  Decoder output: {decoder_image_size}x{decoder_image_size}")
        print(f"  Noise tau: {noise_tau} (0 for inference)")
    
    def noising(self, x: torch.Tensor) -> torch.Tensor:
        """Add noise during training"""
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise
    
    def encode(self, x):
        """Encode with proper resize and normalization"""
        b, c, h, w = x.shape
        
        # Resize if needed (256 -> 224)
        if h != self.encoder_size or w != self.encoder_size:
            x = F.interpolate(x, size=(self.encoder_size, self.encoder_size), 
                            mode='bicubic', align_corners=False)
        
        # Normalize with ImageNet stats
        x = (x - self.encoder_mean) / self.encoder_std
        
        # Encode
        with torch.cuda.amp.autocast(enabled=False):
            z = self.encoder(x)
        
        # Add noise only during training
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        
        # Reshape to 2D
        h = w = int(math.sqrt(z.shape[1]))
        z = z.transpose(1, 2).reshape(b, -1, h, w)
        # Apply latent normalization if available
        if self.normalize_latents:
            z = (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)
        return z
    
    def decode(self, z):
        """Decode with proper denormalization and clamping"""
        # Denormalize latents if needed
        if self.normalize_latents:
            z = z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean
        # Reshape from 2D to sequence
        b, c, h, w = z.shape
        z = z.reshape(b, c, -1).transpose(1, 2)
        
        # Decode
        x = self.decoder(z)
        x = self.decoder.unpatchify(x)
        
        # Denormalize from ImageNet stats
        x = x * self.encoder_std + self.encoder_mean
        
        # CRITICAL: Clamp to valid range [0, 1]
        x = x.clamp(0, 1)
        
        return x
    
    def forward(self, x):
        """Full forward pass"""
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec
    
    def load_decoder(self, path):
        """Load decoder weights"""
        print(f"Loading decoder from {path}")
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        # Map keys
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'trainable_cls_token' in k:
                new_k = k.replace('trainable_cls_token', 'cls_token')
                new_state_dict[new_k] = v
            elif 'decoder_layers' in k:
                new_k = k.replace('decoder_layers', 'blocks')
                new_k = new_k.replace('.attention.attention.', '.attn.')
                new_k = new_k.replace('.attention.output.dense', '.attn.proj')
                new_k = new_k.replace('.intermediate.dense', '.fc1')
                new_k = new_k.replace('.output.dense', '.fc2')
                new_k = new_k.replace('.layernorm_before', '.norm1')
                new_k = new_k.replace('.layernorm_after', '.norm2')
                
                if '.query' in k or '.key' in k or '.value' in k:
                    continue
                    
                new_state_dict[new_k] = v
            elif 'decoder_norm' in k:
                new_state_dict[k.replace('decoder_norm', 'norm')] = v
            elif 'decoder_pred' in k:
                new_state_dict[k.replace('decoder_pred', 'pred')] = v
            else:
                new_state_dict[k] = v
        
        # Construct QKV weights
        for i in range(len(self.decoder.blocks)):
            q_key = f'decoder_layers.{i}.attention.attention.query.weight'
            k_key = f'decoder_layers.{i}.attention.attention.key.weight'
            v_key = f'decoder_layers.{i}.attention.attention.value.weight'
            
            if all(key in state_dict for key in [q_key, k_key, v_key]):
                q = state_dict[q_key]
                k = state_dict[k_key]
                v = state_dict[v_key]
                qkv = torch.cat([q, k, v], dim=0)
                new_state_dict[f'blocks.{i}.attn.qkv.weight'] = qkv
            
            q_bias = f'decoder_layers.{i}.attention.attention.query.bias'
            k_bias = f'decoder_layers.{i}.attention.attention.key.bias'
            v_bias = f'decoder_layers.{i}.attention.attention.value.bias'
            
            if all(key in state_dict for key in [q_bias, k_bias, v_bias]):
                q = state_dict[q_bias]
                k = state_dict[k_bias]
                v = state_dict[v_bias]
                qkv = torch.cat([q, k, v], dim=0)
                new_state_dict[f'blocks.{i}.attn.qkv.bias'] = qkv
        
        missing, unexpected = self.decoder.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded decoder - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return missing, unexpected
    
    def load_stats(self, path):
        """Load normalization statistics"""
        print(f"Loading stats from {path}")
        stats = torch.load(path, map_location='cpu', weights_only=False)
        
        if 'mean' in stats and stats['mean'] is not None:
            mean = stats['mean']
            if mean.dim() == 3:
                mean = mean.unsqueeze(0)
            elif mean.dim() == 2:
                mean = mean.unsqueeze(0).unsqueeze(-1)
            self.latent_mean.data = mean
            print(f"Loaded mean: shape {mean.shape}")
        
        if 'var' in stats and stats['var'] is not None:
            var = stats['var']
            if var.dim() == 3:
                var = var.unsqueeze(0)
            elif var.dim() == 2:
                var = var.unsqueeze(0).unsqueeze(-1)
            self.latent_var.data = var
            self.normalize_latents = True
            print(f"Loaded var: shape {var.shape}")

# ============================================
# Utility functions
# ============================================
def create_rae(
    encoder_name='facebook/dinov2-with-registers-base',
    encoder_size=224,
    decoder_path=None,
    stats_path=None,
    device='cuda',
):
    """Create and initialize fixed RAE model"""
    
    model = RAE(
        encoder_name=encoder_name,
        encoder_size=encoder_size,
        decoder_dim=1152,
        decoder_depth=28,
        decoder_heads=16,
        decoder_patch_size=16,
        noise_tau=0.0,  # CRITICAL: No noise for inference
    )
    
    if decoder_path and Path(decoder_path).exists():
        model.load_decoder(decoder_path)
    
    if stats_path and Path(stats_path).exists():
        model.load_stats(stats_path)
    
    # CRITICAL: Set to eval mode
    model = model.to(device).eval()
    return model

def center_crop_arr(pil_image, image_size):
    """Center crop matching official implementation"""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def load_image(path, size=256):
    """Load and preprocess image"""
    img = Image.open(path).convert('RGB')
    img = center_crop_arr(img, size)
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to test image')
    parser.add_argument('--decoder_path', type=str, default='/mnt/bn/dq-storage-ckpt/wangyuqing/huggingface/hub/RAE-collections/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
                       help='Path to decoder weights')
    parser.add_argument('--stats_path', type=str, default='/mnt/bn/dq-storage-ckpt/wangyuqing/huggingface/hub/RAE-collections/stats/dinov2/wReg_base/imagenet1k/stat.pt',
                       help='Path to normalization stats')
    parser.add_argument('--output_dir', type=str, default='./test_fixed')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Create model
    print("Creating FIXED RAE model...")
    model = create_rae(
        encoder_size=224,
        decoder_path=args.decoder_path,
        stats_path=args.stats_path,
        device=args.device
    )
    
    # Test
    img = load_image(args.image_path).to(args.device)
    
    with torch.no_grad():
        recon = model(img)
        print(f"Input range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"Output range: [{recon.min():.3f}, {recon.max():.3f}]")
        print(f"MSE: {F.mse_loss(recon, img).item():.6f}")
        
        # Save
        recon_np = recon[0].cpu().numpy()
        recon_np = (recon_np * 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(recon_np).save(f"{args.output_dir}/reconstruction_fixed.png")
        
        print(f"Saved to {args.output_dir}")