# KVEraser

[![arXiv](https://img.shields.io/badge/📄arXiv-2606.17034-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e)](https://arxiv.org/abs/2606.17034)

![fig](theme_figure.png)

## Table of Contents

- [Setup](#setup)
- [Training](#training)
- [KVEraser Evaluation](#kveraser-evaluation)
- [Exact Recompute Baseline Evaluation](#exact-recompute-baseline-evaluation)

## Setup

```bash
conda create -n kveraser python=3.11 -y
conda activate kveraser
pip install -r requirements.txt
```

Authenticate for W&B logging:

```bash
wandb login
```

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model Qwen/Qwen3-8B \
  --output_dir checkpoints/kveraser_niah_joint \
  --wandb_project kveraser-niah
```

## KVEraser Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_kveraser.py \
  --model Qwen/Qwen3-8B \
  --checkpoint checkpoints/kveraser_niah_joint \
  --placement single_gpu
```

## Exact Recompute Baseline Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_recompute.py \
  --model Qwen/Qwen3-8B \
  --placement single_gpu
```

## Citation

```bibtex
@article{li2026kveraser,
  title={KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing},
  author={Li, Mufei and Liu, Shikun and Fu, Dongqi and Wang, Haoyu and Xia, Yinglong and Li, Hong and Yan, Hong and Li, Pan},
  journal={arXiv preprint arXiv:2606.17034},
  year={2026}
}
```
