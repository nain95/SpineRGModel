# Data Format

Place your train/val/test CSVs here (or pass `--train_csv`/`--val_csv`/`--test_csv` to point
elsewhere). Each CSV needs these columns:

| column       | description                                                                 |
|--------------|-------------------------------------------------------------------------------|
| `path`       | path to a folder containing that sample's DICOM files                        |
| `CR_reading` | ground-truth radiology report text                                           |

Inside each sample folder, X-ray views are DICOM files named by view index, e.g. `00001.dcm`,
`00002.dcm`, .... The `--views` flag selects which indices are loaded and fed to the model.

| index | view              |
|-------|-------------------|
| 1     | AP                |
| 2     | Oblique (right)   |
| 3     | Oblique (left)    |
| 4     | Lateral           |
| 5     | Flexion           |
| 6     | Extension         |

Default paths (relative to wherever you run the command from):
- `data/processed/train_dataset_unique.csv`
- `data/processed/val_dataset_unique.csv`
- `data/processed/test_dataset_unique.csv`

## FORTE_Spine.json

Spine-specific keyword dictionary (degree/landmark/feature/impression categories) for use with
[FORTE](https://github.com/charlierabea/FORTE) clinical entity evaluation.
