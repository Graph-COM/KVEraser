# KVEraser NIAH Release

This directory is a standalone Apache-2.0 implementation of the NIAH experiment from KVEraser. It trains one full KVEraser backbone jointly across six context sizes and evaluates KVEraser and exact recomputation with separate scripts.

The release intentionally excludes pretraining, QA datasets, LoRA, and non-recompute baselines.

## Setup

Use a CUDA environment where PyTorch can see the intended GPU(s). The scripts raise immediately if CUDA is unavailable.

```bash
conda create -n kveraser-niah python=3.11 -y
conda activate kveraser-niah

# Install the PyTorch build matching your CUDA driver first if needed.
pip install -r requirements.txt
```

Authenticate for gated model or dataset access and W&B logging:

```bash
hf auth login
wandb login
```

Set `HF_HUB_OFFLINE=1` or `TRANSFORMERS_OFFLINE=1` only after the model and dataset are already cached locally.

## Dataset

The scripts first look for local Parquet files under the release directory:

```text
data/niah/ctx1000_t32/{train,validation,test}-00000-of-00001.parquet
```

If a complete local split set is not present for a context, the scripts automatically download that context from the public Hugging Face dataset repository `Graph-COM/KVEraser`.

Expected Hugging Face configs are:

```text
niah_ctx1000_t32
niah_ctx2000_t32
niah_ctx4000_t32
niah_ctx8000_t32
niah_ctx16000_t32
niah_ctx32000_t32
```

Each config contains `train`, `validation`, and `test` splits with 200, 100, and 100 rows respectively.

## Training

Paper protocol defaults are Qwen3-8B, six context sizes from 1K to 32K, batch size 1, AdamW, learning rate `1e-5`, weight decay `0.01`, gradient clipping at `1`, and one epoch.

Two-GPU training command:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py --placement balanced
```

Useful explicit form:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model Qwen/Qwen3-8B \
  --placement balanced \
  --output_dir checkpoints/kveraser_niah_joint \
  --wandb_project kveraser-niah
```

The script concatenates the six 200-row train splits into one seed-shuffled 1,200-example sweep. It logs per-step loss, learning rate, gradient norm, per-context validation EM, and macro validation EM. The eraser backbone is saved with safetensors plus `run_config.json` metadata.

Use `--wandb_mode offline` or `--wandb_mode disabled` when needed.

## KVEraser Evaluation

Run latency evaluation with one visible 80GB A100 and explicit single-GPU placement:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_kveraser.py \
  --model Qwen/Qwen3-8B \
  --checkpoint checkpoints/kveraser_niah_joint \
  --placement single_gpu
```

The script loads the frozen generator and trained eraser, evaluates all six 100-example test splits, and prints per-context sample count, EM, total latency, average latency, average generated tokens, editing latency, delayed-query latency, decoding latency, and macro EM.

## Exact Recompute Evaluation

Exact recomputation has no eraser checkpoint dependency:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_recompute.py \
  --model Qwen/Qwen3-8B \
  --placement single_gpu
```

It removes the designated span from the cached prompt, reuses the valid prefix cache, recomputes the complete edited suffix, then uses the same delayed-query processing and greedy decoding behavior as KVEraser.

## Timing Boundary

Both evaluation scripts prefill the original context before timing starts. Timed latency begins after the original-context KV cache exists and includes:

- KVEraser cache editing or exact suffix recomputation
- delayed-query processing
- complete greedy decoding

CUDA is synchronized around timed regions. One untimed warmup sample is run per context. The scripts print GPU name/count, Torch/CUDA versions, dtype, attention backend, and resolved generator device map.

## Local Dataset Publication Tool

`tools/publish_niah.py` is intentionally ignored by Git. It validates and writes the Hugging Face dataset layout locally and can optionally upload it when credentials are available:

```bash
python tools/publish_niah.py --output_dir /tmp/KVEraser_hf_dataset
python tools/publish_niah.py --output_dir /tmp/KVEraser_hf_dataset --push --repo_id Graph-COM/KVEraser
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
