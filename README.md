# CuBID: Cubic Discrete Diffusion for High-Dimensional Representation Tokens <br><sub>Official PyTorch Implementation</sub>

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2506.XXXXX)&nbsp;

<p align="center">
  <img width="1350" alt="image" src="assets/demo.png" />
</p>

This is a PyTorch/GPU implementation of the paper [Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](https://arxiv.org/abs/2506.XXXXX):

```
@article{wang2025cubic,
  title={Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens},
  author={Wang, Yuqing and Ma, Chuofan and Lin, Zhijie and Teng, Yao and Yu, Lijun and Wang, Shuai and Han, Jiaming and Feng, Jiashi and Jiang, Yi and Liu, Xihui},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2025}
}
```

## Highlights

* **First discrete generation on high-dimensional tokens**: CuBID directly generates 768-dimensional representation tokens, preserving semantic richness for both understanding and generation
* **Cubic masking strategy**: Fine-grained masking across the entire h×w×d tensor, capturing complex dependencies both within and across spatial positions
* **Efficient parallel generation**: Transforms intractable O(h×w×d) sequential generation into O(T) parallel iterations where T ≪ h×w×d
* **Strong scaling behavior**: Consistent improvement from 900M to 3.7B parameters


## Preparation

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/YuqingWang1029/CuBID.git
cd CuBID
```

A suitable [conda](https://conda.io/) environment named `cubid` can be created and activated with:

```
conda env create -f environment.yaml
conda activate cubid
```

### Pre-trained Models

Download pre-trained CuBID models:

| Model | Params | FID | Inception Score | Download |
|-------|--------|-----|-----------------|----------|
| CuBID-Base | 946M | 2.37 | 213.4 | [Coming Soon]() |
| CuBID-Large | 1.4B | 2.04 | 217.0 | [Coming Soon]() |
| CuBID-Huge | 3.7B | 1.88 | 247.0 | [Coming Soon]() |

Download RAE (Representation AutoEncoder) models from [huggingface](https://huggingface.co/RAE-collections), including decoders and normalization stats.


## Generation

### Evaluation (ImageNet 256x256)

Evaluate CuBID-Base with classifier-free guidance:
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cubid.py \
--model cubid_base \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --cfg 1.0 --quant_bits 6 --cfg_schedule linear --temperature 1.0 \
--output_dir output/test_cubid_base \
--resume pretrained_models/cubid_base \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate CuBID-Large with classifier-free guidance:
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cubid.py \
--model cubid_large \
--eval_bsz 256 --num_images 50000 \
--num_iter 256 --cfg 1.0 --quant_bits 6 --cfg_schedule linear --temperature 1.0 \
--output_dir output/test_cubid_large \
--resume pretrained_models/cubid_large \
--data_path ${IMAGENET_PATH} --evaluate
```

Evaluate CuBID-Huge with classifier-free guidance:
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cubid.py \
--model cubid_huge \
--eval_bsz 128 --num_images 50000 \
--num_iter 256 --cfg 1.0 --quant_bits 6 --cfg_schedule linear --temperature 1.0 \
--output_dir output/test_cubid_huge \
--resume pretrained_models/cubid_huge \
--data_path ${IMAGENET_PATH} --evaluate
```

- Generation speed can be increased by reducing the number of iterations (e.g., `--num_iter 64`).

### (Optional) Caching RAE Latents

The RAE latents can be pre-computed and saved to `CACHED_PATH` to accelerate training:

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_cache.py \
--img_size 256 --encoder_size 224 \
--batch_size 128 \
--data_path ${IMAGENET_PATH} --cached_path ${CACHED_PATH}
```

### Training

Script for the default setting (CuBID-Base):

```bash
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_cubid.py \
--img_size 256 --encoder_size 224 --vae_embed_dim 768 --vae_stride 16 --patch_size 1 \
--model cubid_base --quant_bits 6 --quant_min -20.0 --quant_max 20.0 \
--epochs 800 --warmup_epochs 100 --batch_size 64 --blr 5.0e-5 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH}
```

- (Optional) To train with cached RAE latents, add `--use_cached --cached_path ${CACHED_PATH}` to the arguments.
- (Optional) To save GPU memory during training, add `--grad_checkpointing` to the arguments.


## Acknowledgements

Part of the code is based on [MAR](https://github.com/LTH14/mar) and [TokenBridge](https://github.com/YuqingWang1029/TokenBridge). We use [RAE](https://github.com/xxxx/RAE) for representation encoding and decoding.
