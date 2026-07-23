import os
import random
import warnings

import numpy as np
import torch

from config import build_config, get_parser
from dataloader import create_dataloader
from model import SpineRGModel
from trainer import SpineXrayRGTrainer

warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*GLIBC.*")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", message=".*could work incorrectly in ReplayMode.*")
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_data_loaders(config):
    train_dataloader = create_dataloader(
        csv_path=config.data.train_csv_path,
        batch_size=config.data.batch_size,
        shuffle=config.data.shuffle_train,
        num_workers=config.data.num_workers,
        target_size=tuple(config.data.target_size),
        image_indices=config.data.image_indices,
        mode="train",
        seed=config.seed,
    )

    val_dataloader = create_dataloader(
        csv_path=config.data.val_csv_path,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        target_size=tuple(config.data.target_size),
        image_indices=config.data.image_indices,
        mode="val",
        seed=config.seed,
    )

    test_dataloader = None
    if config.data.test_csv_path:
        test_dataloader = create_dataloader(
            csv_path=config.data.test_csv_path,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            target_size=tuple(config.data.target_size),
            image_indices=config.data.image_indices,
            mode="test",
            seed=config.seed,
        )

    return train_dataloader, val_dataloader, test_dataloader


def main():
    parser = get_parser()
    parser.add_argument("--ckpt_path", type=str, default=None, help="Resume training from this checkpoint")
    args = parser.parse_args()
    config = build_config(args)

    print(f"Experiment: {config.experiment.experiment_name}")
    print(f"Image indices: {config.data.image_indices}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")

    if args.ckpt_path:
        print(f"Resuming from checkpoint: {args.ckpt_path}")

    config.setup_experiment_dirs()
    set_seed(config.seed)

    print("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(config)
    print(f"Data loaders created - Train: {len(train_dataloader)}, Val: {len(val_dataloader)}")

    print("Creating model...")
    model = SpineRGModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("Creating trainer...")
    trainer = SpineXrayRGTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=config,
    )

    if args.ckpt_path:
        print(f"Loading checkpoint from: {args.ckpt_path}")
        trainer.load_checkpoint(args.ckpt_path)
        print(f"Resuming from epoch {trainer.current_epoch}, step {trainer.global_step}")
        print(f"Best validation loss so far: {trainer.best_val_loss:.4f}")

    print("Starting training...")
    trainer.train()
    print("Training completed!")

    if test_dataloader is not None and trainer.accelerator.is_main_process:
        print("Starting test evaluation...")
        if trainer.best_model_path.exists():
            print(f"Loading best model from {trainer.best_model_path}")
            trainer.load_checkpoint(str(trainer.best_model_path))
            print("Best model loaded for testing")
        else:
            print("Warning: Best model checkpoint not found, using last model")
        try:
            test_results = trainer.test()
            print("Test evaluation completed!")

            if test_results:
                # Medical-domain-optimized: content overlap weighted over exact wording
                test_combined_score = (
                    0.3 * test_results.get("bleu_4", 0.0) + 0.7 * test_results.get("rougeL", 0.0)
                )
                print(f"Test combined score: {test_combined_score:.4f}")
        except Exception as e:
            print(f"Test evaluation failed: {e}")


if __name__ == "__main__":
    main()
