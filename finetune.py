"""Jointly fine-tune a pretrained Qwen3-8B LoRA KVEraser on NIAH, QA, and ToolBench."""
from dataclasses import asdict
from pathlib import Path
import random

from kveraser.adapters import BASE_MODEL, BASE_REVISION, AdapterCheckpoint
from kveraser.data import DatasetSource
from kveraser.model import KVEraser
from kveraser.runtime import load_tokenizer, load_training_models, require_cuda, set_seed, use_local_files_only
from kveraser.training import TrainingConfig, build_training_parser, run_training


def build_arg_parser():
    parser = build_training_parser(__doc__, grad_accum_steps=1, output_dir="checkpoints/finetune")
    parser.add_argument("--checkpoint", required=True, help="HF model ID or local pretrained adapter directory.")
    parser.add_argument("--checkpoint_revision", default="main")
    return parser


def epoch_sample_order(sample_count, *, seed, epoch):
    """Reproduce successive shuffles of the joint dataset from the same seeded RNG."""
    rng = random.Random(seed)
    for _ in range(epoch):
        indices = list(range(sample_count))
        rng.shuffle(indices)
    return indices


def main(args):
    for name in ("epochs", "max_samples", "max_validation_samples"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be positive.")
    training = TrainingConfig(args.lr, args.weight_decay, args.max_grad_norm, args.grad_accum_steps)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    require_cuda()
    set_seed(args.seed)
    offline = args.local_files_only or use_local_files_only()
    adapter = AdapterCheckpoint(args.checkpoint, revision=args.checkpoint_revision,
                                cache_dir=args.cache_dir, local_files_only=offline)
    source = DatasetSource(data_root=args.data_root, repo_id=args.data_repo, revision=args.data_revision,
                           cache_dir=args.cache_dir, local_files_only=offline)
    samples = [row for dataset in source.finetune_order for row in source.load(dataset, "train")]
    validation = [row for dataset in source.finetune_order for row in source.load(dataset, "validation")]
    validation = validation[:args.max_validation_samples]
    tokenizer = load_tokenizer(BASE_MODEL, revision=BASE_REVISION,
                               cache_dir=args.cache_dir, local_files_only=offline)
    generator, eraser = load_training_models(checkpoint=adapter, placement=args.placement,
                                              cache_dir=args.cache_dir, local_files_only=offline)
    model = KVEraser(tokenizer, generator, eraser)
    config = dict(vars(args), **asdict(training), base_model=BASE_MODEL, base_revision=BASE_REVISION,
                  tokenizer_revision=BASE_REVISION, dtype="bf16", attn_backend="sdpa",
                  local_files_only=offline, data_commit=source.revision, checkpoint_commit=adapter.revision,
                  datasets=list(source.finetune_order), train_samples=min(len(samples), args.max_samples or len(samples)),
                  validation_samples=len(validation))
    return run_training(model, samples, validation, training, config, epoch_sample_order)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
