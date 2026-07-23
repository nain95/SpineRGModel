"""
Configuration for SpineRGModel: a plain argparse CLI builds a dataclass Config,
which model.py / trainer.py consume via dotted attribute access
(e.g. config.model.llm.model_name, config.training.scheduler.name).
"""
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    train_csv_path: str = ""
    val_csv_path: str = ""
    test_csv_path: str = ""
    target_size: List[int] = field(default_factory=lambda: [256, 256])
    image_indices: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    batch_size: int = 32
    num_workers: int = 8
    shuffle_train: bool = True


@dataclass
class ExperimentConfig:
    experiment_name: str = "spine_rg"
    experiment_description: str = ""
    output_dir: str = "outputs/spine_rg"
    log_dir: str = "outputs/spine_rg/logs"
    checkpoint_dir: str = "outputs/spine_rg/checkpoints"


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.995])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    name: str = "cosine_annealing"
    T_max: int = 30
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-7
    warmup_ratio: float = 0.1  # used by cosine_with_warmup
    warmup_steps: int = 0  # alternative to warmup_ratio


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    learning_rate: float = 7e-5
    weight_decay: float = 0.1
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.5
    early_stopping_patience: int = 5
    min_epochs: int = 3  # minimum epochs before early stopping can trigger
    mixed_precision: str = "bf16"  # bf16, fp16, or no
    contrastive_weight: float = 0.1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class VisualEncoderConfig:
    swin_model_name: str = "microsoft/swinv2-base-patch4-window8-256"
    max_views: int = 6
    target_size: List[int] = field(default_factory=lambda: [256, 256])
    num_tokens_per_view: int = 6  # latent tokens per view compressed via Perceiver Resampler


@dataclass
class LLMConfig:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    torch_dtype: str = "bfloat16"
    use_cache: bool = False


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ContrastiveConfig:
    temperature: float = 0.05


@dataclass
class ModelConfig:
    visual_encoder: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    contrastive: ContrastiveConfig = field(default_factory=ContrastiveConfig)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    min_length: int = 5
    num_beams: int = 5
    do_sample: bool = False
    top_p: float = 1.0
    temperature: float = 1.0  # not used when do_sample=False
    repetition_penalty: float = 1.3
    length_penalty: float = 0.8
    no_repeat_ngram_size: int = 3
    prompt_text: str = "Lumbar spine report:"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42

    def setup_experiment_dirs(self) -> None:
        for dir_path in [self.experiment.output_dir, self.experiment.log_dir, self.experiment.checkpoint_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Flatten to a plain dict, for WandB config logging."""
        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            return obj
        return dataclass_to_dict(self)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpineRGModel training/evaluation")

    data = parser.add_argument_group("data")
    data.add_argument("--train_csv", type=str, default="data/processed/train_dataset_unique.csv",
                       help="Path to train CSV (relative to cwd unless absolute)")
    data.add_argument("--val_csv", type=str, default="data/processed/val_dataset_unique.csv",
                       help="Path to val CSV (relative to cwd unless absolute)")
    data.add_argument("--test_csv", type=str, default="data/processed/test_dataset_unique.csv",
                       help="Path to test CSV (relative to cwd unless absolute)")
    data.add_argument("--views", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6],
                       help="X-ray view indices to use, e.g. --views 1 4 for AP+lateral")
    data.add_argument("--target_size", type=int, nargs=2, default=[256, 256], metavar=("H", "W"))
    data.add_argument("--batch_size", type=int, default=32)
    data.add_argument("--num_workers", type=int, default=8)
    data.add_argument("--no_shuffle_train", action="store_true", help="Disable train-set shuffling")

    model = parser.add_argument_group("model")
    model.add_argument("--swin_model", type=str, default="microsoft/swinv2-base-patch4-window8-256")
    model.add_argument("--max_views", type=int, default=6)
    model.add_argument("--num_tokens_per_view", type=int, default=6)
    model.add_argument("--llm_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    model.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16", "float64"])
    model.add_argument("--use_cache", action="store_true",
                        help="Enable KV cache in the base LLM (usually off during training)")
    model.add_argument("--lora_r", type=int, default=8)
    model.add_argument("--lora_alpha", type=int, default=16)
    model.add_argument("--lora_dropout", type=float, default=0.05)
    model.add_argument("--lora_target_modules", type=str, nargs="+", default=["q_proj", "v_proj"])
    model.add_argument("--contrastive_temperature", type=float, default=0.05)

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--lr", type=float, default=7e-5)
    train.add_argument("--weight_decay", type=float, default=0.1)
    train.add_argument("--grad_accum", type=int, default=2)
    train.add_argument("--max_grad_norm", type=float, default=1.5)
    train.add_argument("--early_stopping_patience", type=int, default=5)
    train.add_argument("--min_epochs", type=int, default=3)
    train.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "no"])
    train.add_argument("--contrastive_weight", type=float, default=0.1)
    train.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.995], metavar=("BETA1", "BETA2"))
    train.add_argument("--eps", type=float, default=1e-8)
    train.add_argument("--scheduler", type=str, default="cosine_annealing",
                        choices=["cosine_annealing_warm_restarts", "cosine_annealing",
                                 "cosine_with_warmup", "reduce_lr_on_plateau"])
    train.add_argument("--t_max", type=int, default=30, help="cosine_annealing[_warm_restarts]")
    train.add_argument("--t_0", type=int, default=10, help="cosine_annealing_warm_restarts")
    train.add_argument("--t_mult", type=int, default=2, help="cosine_annealing_warm_restarts")
    train.add_argument("--eta_min", type=float, default=1e-7)
    train.add_argument("--warmup_ratio", type=float, default=0.1, help="cosine_with_warmup")
    train.add_argument("--warmup_steps", type=int, default=0, help="cosine_with_warmup, overrides warmup_ratio")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--max_new_tokens", type=int, default=128)
    gen.add_argument("--min_length", type=int, default=5)
    gen.add_argument("--num_beams", type=int, default=5)
    gen.add_argument("--do_sample", action="store_true")
    gen.add_argument("--top_p", type=float, default=1.0)
    gen.add_argument("--gen_temperature", type=float, default=1.0, help="only used when --do_sample is set")
    gen.add_argument("--repetition_penalty", type=float, default=1.3)
    gen.add_argument("--length_penalty", type=float, default=0.8)
    gen.add_argument("--no_repeat_ngram_size", type=int, default=3)
    gen.add_argument("--prompt_text", type=str, default="Lumbar spine report:")

    exp = parser.add_argument_group("experiment")
    exp.add_argument("--exp_name", type=str, default="spine_rg")
    exp.add_argument("--exp_description", type=str, default="")
    exp.add_argument("--output_dir", type=str, default=None,
                      help="Root output dir (default: outputs/<exp_name>)")
    exp.add_argument("--seed", type=int, default=42)

    return parser


def build_config(args: argparse.Namespace) -> Config:
    output_dir = args.output_dir or f"outputs/{args.exp_name}"

    return Config(
        data=DataConfig(
            train_csv_path=args.train_csv,
            val_csv_path=args.val_csv,
            test_csv_path=args.test_csv,
            target_size=list(args.target_size),
            image_indices=list(args.views),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_train=not args.no_shuffle_train,
        ),
        experiment=ExperimentConfig(
            experiment_name=args.exp_name,
            experiment_description=args.exp_description,
            output_dir=output_dir,
            log_dir=f"{output_dir}/logs",
            checkpoint_dir=f"{output_dir}/checkpoints",
        ),
        training=TrainingConfig(
            num_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.grad_accum,
            max_grad_norm=args.max_grad_norm,
            early_stopping_patience=args.early_stopping_patience,
            min_epochs=args.min_epochs,
            mixed_precision=args.mixed_precision,
            contrastive_weight=args.contrastive_weight,
            optimizer=OptimizerConfig(betas=list(args.betas), eps=args.eps),
            scheduler=SchedulerConfig(
                name=args.scheduler,
                T_max=args.t_max,
                T_0=args.t_0,
                T_mult=args.t_mult,
                eta_min=args.eta_min,
                warmup_ratio=args.warmup_ratio,
                warmup_steps=args.warmup_steps,
            ),
        ),
        model=ModelConfig(
            visual_encoder=VisualEncoderConfig(
                swin_model_name=args.swin_model,
                max_views=args.max_views,
                target_size=list(args.target_size),
                num_tokens_per_view=args.num_tokens_per_view,
            ),
            llm=LLMConfig(model_name=args.llm_model, torch_dtype=args.torch_dtype, use_cache=args.use_cache),
            lora=LoRAConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=list(args.lora_target_modules),
            ),
            contrastive=ContrastiveConfig(temperature=args.contrastive_temperature),
        ),
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            min_length=args.min_length,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.gen_temperature,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            prompt_text=args.prompt_text,
        ),
        seed=args.seed,
    )
