#!/usr/bin/env python3
"""
Evaluate a trained SpineRGModel checkpoint on a test set.

Usage:
    python src/test_only.py --checkpoint /path/to/best_model.pt --test_csv path/to/test.csv --views 1
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import build_config, get_parser
from dataloader import create_dataloader
from metrics import calculate_metrics
from model import SpineRGModel

warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
warnings.filterwarnings("ignore", message=".*GLIBC.*")
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_checkpoint(model, checkpoint_path: str, device: str = "cuda"):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # strict=False tolerates architecture drift between an old checkpoint and current model code
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint (ignored): {unexpected_keys}")

    print("Checkpoint loaded successfully!")
    print(f"  - Trained epochs: {checkpoint.get('epoch', 'unknown')}")
    print(f"  - Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")

    return model


def evaluate_test_set(model, test_dataloader, device: str = "cuda", save_dir: str = "./test_results"):
    """Run evaluation on test set, saving predictions and metrics to save_dir."""
    model.eval()
    model.to(device)

    all_ground_truth = []
    all_generated = []
    all_file_paths = []

    print(f"\nEvaluating {len(test_dataloader.dataset)} test samples...")

    progress_bar = tqdm(
        test_dataloader, desc="Testing", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            if batch_data is None:
                continue

            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, texts = batch_data
            else:
                continue

            images = images.to(device)
            batch_size = len(texts)

            generated_reports = model.generate_report(images)

            for i in range(batch_size):
                if sample_idx + i < len(test_dataloader.dataset.df):
                    all_file_paths.append(test_dataloader.dataset.df.iloc[sample_idx + i]["path"])
                else:
                    all_file_paths.append("N/A")

            all_ground_truth.extend(texts)
            all_generated.extend(generated_reports)
            sample_idx += batch_size

            progress_bar.set_postfix({"Samples": f"{len(all_generated)}/{len(test_dataloader.dataset)}"})

            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nGeneration completed. Total samples: {len(all_generated)}")

    test_metrics = calculate_metrics(all_ground_truth, all_generated)

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    # Medical-domain-optimized: content overlap weighted over exact wording
    combined_score = 0.3 * test_metrics.get("bleu_4", 0.0) + 0.7 * test_metrics.get("rougeL", 0.0)
    print(f"COMBINED_SCORE (0.3×BLEU-4 + 0.7×ROUGE-L): {combined_score:.4f}")
    print("=" * 60)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(
        {"File_Path": all_file_paths, "Ground_Truth": all_ground_truth, "Generated_Report": all_generated}
    )
    csv_path = save_dir / "test_predictions.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")

    metrics_df = pd.DataFrame([{**test_metrics, "combined_score": combined_score}])
    metrics_path = save_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")

    return test_metrics


def main():
    parser = get_parser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--save_dir", type=str, default="./test_results", help="Directory to save test results")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.set_defaults(batch_size=8)  # smaller default: generation is more memory-hungry than training
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    config = build_config(args)

    print(f"Test CSV: {config.data.test_csv_path}")
    print(f"Image indices: {config.data.image_indices}")
    print(f"Batch size: {config.data.batch_size}")

    print("\nCreating model...")
    model = SpineRGModel(config)
    model = load_checkpoint(model, args.checkpoint, args.device)

    print("\nCreating test dataloader...")
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
    print(f"Test samples: {len(test_dataloader.dataset)}")

    print("\nStarting evaluation...")
    evaluate_test_set(model=model, test_dataloader=test_dataloader, device=args.device, save_dir=args.save_dir)

    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()
