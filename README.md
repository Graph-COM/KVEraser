# KVEraser

[![arXiv](https://img.shields.io/badge/📄arXiv-2606.17034-ff6b6b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e)](https://arxiv.org/abs/2606.17034)

![fig](theme_figure.png)

Qwen3-8B LoRA KVEraser for localized context erasing, with pre-training, task-specific fine-tuning, and evaluation.

## Table of Contents

- [Setup](#setup)
- [Pretrained Checkpoint](#pretrained-checkpoint)
- [Pre-training](#pre-training)
- [Task-specific Fine-tuning](#task-specific-fine-tuning)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Setup

```bash
conda create -n kveraser python=3.10 -y
conda activate kveraser
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Requires CUDA with BF16 support. Tested with Python 3.10, PyTorch 2.6.0 / CUDA 12.4, and RTX 6000 Ada 48GB GPUs: four for training, one for evaluation.

W&B logging is optional and disabled by default. Run `wandb login`, then add `--wandb_mode online --wandb_project kv-eraser` to either training command to enable it.

## Pretrained Checkpoint

The released [LoRA adapter](https://huggingface.co/Graph-COM/KVEraser) includes pre-training and joint task-specific fine-tuning. Evaluate it directly:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint Graph-COM/KVEraser \
  --measure_performance --output_dir outputs/pretrained
```

The adapter, Qwen3-8B backbone, and [evaluation data](https://huggingface.co/datasets/Graph-COM/KVEraser) download automatically and are cached. Evaluation shares one backbone between eraser and generator, with the LoRA adapter resident in memory.

## Pre-training

Pre-train on 72,000 Wikipedia span-erasure samples, with 8,000 validation samples:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain.py \
  --placement balanced \
  --output_dir checkpoints/qwen3_8b/pretrain
```

Training data downloads automatically. Defaults are seed 42, one epoch, AdamW, LR `1e-5`, weight decay `0.01`, gradient clipping `1.0`, and gradient accumulation 8. LoRA uses rank 16, alpha 32, and dropout 0.

## Task-specific Fine-tuning

Jointly fine-tune on NIAH, HotpotQA, Natural Questions, TriviaQA, and ToolBench (11,491 training / 1,642 validation samples):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py \
  --checkpoint checkpoints/qwen3_8b/pretrain \
  --placement balanced \
  --output_dir checkpoints/qwen3_8b/finetune
```

This loads the pretrained adapter with a fresh optimizer and uses gradient accumulation 1; other defaults match pre-training. Each training directory contains the adapter, `run_config.json`, and `metrics.json`.

## Evaluation

Evaluate a local adapter on all 1,522 test samples: six NIAH lengths (1K–32K), 2Wiki, MuSiQue, IIRC, and BFCL.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint checkpoints/qwen3_8b/finetune \
  --measure_performance --output_dir outputs/finetune
```

To evaluate the pre-training-only checkpoint, use `--checkpoint checkpoints/qwen3_8b/pretrain` and a separate output directory. Select a task with, for example, `--datasets eval/niah/ctx32000`.

Full-recompute baseline:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --method full-recompute \
  --measure_performance --output_dir outputs/full-recompute
```

Other baselines: `--method discard`, `instruction`, `partial-erase`, or `partial-query`.

Each evaluation writes per-sample JSONL and `summary.json` with EM, QA F1, and configuration. `--measure_performance` adds latency and CUDA memory; online timing excludes loading and initial prefill, and includes erasing/recomputation, query replay, and generation. Three warmup samples per dataset are excluded from the results.

Use an empty output directory for each run. All three scripts accept `--data_root` for a local data root, `--cache_dir`, and `--local_files_only`. Data paths follow `qwen3_8b/v1/<pretrain|finetune|eval>/<task>/<variant>/<split>-*.parquet`. Use `--data_revision` and, when loading an adapter, `--checkpoint_revision` to select HF revisions; resolved commits are recorded in the output configuration.

## Citation

```bibtex
@article{li2026kveraser,
  title={KVEraser: Learning to Steer KV Cache for Efficient Localized Context Erasing},
  author={Li, Mufei and Liu, Shikun and Fu, Dongqi and Wang, Haoyu and Xia, Yinglong and Li, Hong and Yan, Hong and Li, Pan},
  journal={arXiv preprint arXiv:2606.17034},
  year={2026}
}
```
