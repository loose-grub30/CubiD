# CubiD: Cubic Discrete Diffusion for High-Dimensional Representation Tokens <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2603.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2603.XXXXX)&nbsp;
<img width="1378" height="469" alt="image" src="https://github.com/user-attachments/assets/9a34395c-58be-424f-a6d7-16965467136c" />

> *Can we generate high-dimensional semantic representations discretely, just like language models generate text?*

Generating high-dimensional semantic representations has long been a pursuit for visual generation, yet discrete methods, the paradigm shared with language models, remain stuck with low-dimensional tokens. **CubiD** breaks this barrier with fine-grained cubic masking across the h×w×d tensor, directly modeling dependencies across both spatial and dimensional axes in 768+-dim representation space, while the discretized tokens preserve their original understanding capabilities.

This is a PyTorch/GPU implementation of the paper [Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](https://arxiv.org/abs/2506.XXXXX):

```
@article{wang2025cubic,
  title={Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens},
  author={Wang, Yuqing and Ma, Chuofan and Lin, Zhijie and Teng, Yao and Yu, Lijun and Wang, Shuai and Han, Jiaming and Feng, Jiashi and Jiang, Yi and Liu, Xihui},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2025}
}
```

## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/YuqingWang1029/CubiD.git
cd CubiD
```

Please refer to [TokenBridge](https://github.com/YuqingWang1029/TokenBridge) and [RAE](https://github.com/nyu-visionx/RAE) for environment setup.

### Pre-trained Models

Download pre-trained CubiD models and RAE weights from [Hugging Face](https://huggingface.co/Epiphqny/CubiD).


## Generation

### Evaluation (ImageNet 256x256)

For example, evaluate CubiD-Large (without CFG):

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cubid.py \
--img_size 256 --encoder_size 224 \
--encoder_name facebook/dinov2-with-registers-base \
--decoder_path ${RAE_DECODER_PATH} \
--stats_path ${RAE_STATS_PATH} \
--vae_embed_dim 768 --vae_stride 14 \
--model cubid_large \
--quant_bits 3 --quant_min -9.0 --quant_max 9.0 \
--eval_bsz 32 --num_images 50000 \
--num_iter 1536 --cfg 1.0 --cfg_schedule constant --temperature 1.0 \
--output_dir ${OUTPUT_DIR} \
--resume cubid_ckpts/cubid_large \
--data_path ${IMAGENET_PATH} --evaluate
```

- The `--resume` argument points to a folder (e.g., `cubid_ckpts/cubid_large`), which automatically loads the checkpoint inside.
- Generation steps can be set from 256 to 1536. More steps generally lead to better results.

### (Optional) Caching RAE Latents

The RAE latents can be pre-computed and saved to `CACHED_PATH` to accelerate training:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --encoder_size 224 \
--encoder_name facebook/dinov2-with-registers-base \
--decoder_path ${RAE_DECODER_PATH} \
--stats_path ${RAE_STATS_PATH} \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

### Training

Script for the default setting (CubiD-Large, 800 epochs, 64 GPUs):

```bash
torchrun --nproc_per_node=8 --nnodes=8 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_cubid.py \
--img_size 256 --encoder_size 224 \
--encoder_name facebook/dinov2-with-registers-base \
--decoder_path ${RAE_DECODER_PATH} \
--stats_path ${RAE_STATS_PATH} \
--vae_embed_dim 768 --vae_stride 14 --patch_size 1 \
--model cubid_large \
--quant_bits 3 --quant_min -9.0 --quant_max 9.0 \
--mask_ratio_min 0.5 --mask_std 0.1 \
--epochs 800 --warmup_epochs 100 --batch_size 32 --blr 5e-5 --lr_schedule cosine \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```

- (Optional) To train with cached RAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments.
- (Optional) To save GPU memory during training, add `--grad_checkpointing` to the arguments.


## Acknowledgements

Part of the code is based on [MAR](https://github.com/LTH14/mar) and [TokenBridge](https://github.com/YuqingWang1029/TokenBridge). We use [RAE](https://github.com/nyu-visionx/RAE) for representation encoding and decoding. Thanks for their awesome work!
