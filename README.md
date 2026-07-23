# SpineRGModel: Radiology Report Generation for Spine X-Rays

A Swin/SwinV2 visual encoder feeds a Flamingo-style Perceiver Resampler, producing visual tokens
for a LoRA-tuned Llama-3.1-8B-Instruct. Trained with a combined generative (report-generation) and
contrastive (image-text InfoNCE) loss.

## Data Format

Prepare CSVs (train/val/test) containing:
- `path`: folder containing that sample's DICOM files, named by view index (e.g. `00001.dcm`, `00002.dcm`, ...)
- `CR_reading`: ground-truth radiology report text

View indices correspond to the following lumbar spine X-ray series:

| index | view              |
|-------|-------------------|
| 1     | AP                |
| 2     | Oblique (right)   |
| 3     | Oblique (left)    |
| 4     | Lateral           |
| 5     | Flexion           |
| 6     | Extension         |

`--views` selects which of these to load, e.g. `--views 1` (AP only) or `--views 1 4` (AP + lateral).

`data/FORTE_Spine.json` is a spine-specific keyword dictionary for use with
[FORTE](https://github.com/charlierabea/FORTE) clinical entity evaluation; see `data/README.md`
for details.

## Usage

```bash
# Configure Accelerate once per machine (GPU count, mixed precision, etc.)
accelerate config

# Single or multi-GPU training (view selection via --views, e.g. --views 1 4 for AP + lateral)
accelerate launch src/train.py --train_csv path/to/train.csv --val_csv path/to/val.csv --views 1

# Evaluate a trained checkpoint
python src/test_only.py --checkpoint /path/to/best_model.pt --test_csv path/to/test.csv --views 1
```

Run `python src/train.py --help` for the full list of options (data/model/training/generation),
all defaulted to the values used in the paper.

## Repository Structure

```
.
└── src/
    ├── config.py       # argparse CLI + config dataclasses
    ├── dataloader.py   # DICOM loading, preprocessing, augmentation
    ├── model.py         # SpineRGModel (visual encoder, resampler, LLM, contrastive head)
    ├── trainer.py       # training/validation loop (Accelerate)
    ├── metrics.py        # BLEU-1..4, ROUGE-L, CIDEr
    ├── train.py         # training entry point
    └── test_only.py      # checkpoint evaluation entry point
```

## Requirements

* Python 3.11
* PyTorch 2.7, Transformers 4.52, PEFT 0.15, Accelerate 1.6
* albumentations, opencv-python, pydicom, pandas, numpy, pycocoevalcap

See `requirements.txt` for pinned versions. Generating Llama-3.1-8B-Instruct requires a Hugging
Face account with access to the gated `meta-llama` repos (`huggingface-cli login`).

## Citation

```
TBD
```
