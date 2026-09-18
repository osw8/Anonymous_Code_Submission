# EpiTAR-Bench

Code accompanying the anonymous ICLR2027 submission:

**EpiTAR-Bench: Benchmarking Transferability, Adaptation, and Reliability of Epilepsy Models**

EpiTAR-Bench is a reproducibility package for benchmarking epilepsy EEG models under transfer, adaptation, and reliability settings. The release is anonymized for peer review. Raw EEG recordings, private checkpoints, author identities, local machine paths, and institution-specific paths are not redistributed.

## Scope

EpiTAR-Bench evaluates epilepsy models from three perspectives:

- **Transferability:** zero-shot generalization across datasets, temporal splits, and scalpEEGtoiEEG settings.
- **Adaptation:** target-domain adaptation under limited labeled supervision.
- **Reliability:** calibration, probability quality, robustness, and prediction consistency.

The implemented benchmark covers:

- cross-dataset detection and prediction;
- cross-time detection and prediction;
- cross-modal scalpEEGtoiEEG transfer for detection and prediction;
- iEEG localization and in-domain iEEG evaluation;
- reliability analyses including ECE, Brier score, NLL, AURC, and AUROC-ECE association.

## Repository Layout

```text
.
├── LICENSE
├── README.md
└── MyCode/
    ├── eeg_benchmark/
    │   ├── tasks/
    │   │   ├── cross_dataset.py
    │   │   ├── cross_time.py
    │   │   └── cross_modal.py
    │   └── analysis/
    │       ├── reliability.py
    │       ├── spectral_evidence.py
    │       └── brain_region_features.py
    ├── models/
    │   ├── RIVER/
    │   └── baselines/
    ├── preprocessing/
    │   ├── configs/
    │   ├── pipeline.py
    │   └── build_chbmit_cross_time_cache.py
    ├── requirement.txt
    ├── setup_environment.sh
    └── train.sh
```

`MyCode/train.sh` is the main experiment launcher. It dispatches cross-dataset, cross-time, cross-modal, localization, in-domain, and reliability jobs through the corresponding Python modules.

## Installation

The project can be installed with the bundled setup script:

```bash
cd MyCode
bash setup_environment.sh
source .venv/bin/activate
```

Alternatively, use conda:

```bash
conda create -n epitar python=3.11 -y
conda activate epitar
cd MyCode
pip install -r requirement.txt
```

GPU execution requires a CUDA-compatible PyTorch installation matching the local system. If the default PyPI wheel is not appropriate for the target server, install the matching PyTorch build first and then install `requirement.txt`.

## Data

Raw EEG data are not redistributed. Users should obtain datasets from the original providers and follow the corresponding data-use agreements.

Public dataset sources include:

- CHB-MIT: https://physionet.org/content/chbmit/
- TUSZ: https://isip.piconepress.com/projects/tuh_eeg/
- Siena Scalp EEG Database: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- Epilepsy_iEEG: https://openneuro.org/datasets/ds003029/versions/1.0.7
- Thalamocortical_iEEG: https://openneuro.org/datasets/ds007445

The default local layout expected by the launcher is:

```text
MyCode/data/
├── raw/
│   ├── chbmit/
│   ├── tusz/
│   ├── siena/
│   ├── epilepsy_ieeg/
│   └── thalamocortical_ieeg/
├── PreprocessResults_eeg_cross_dataset/
├── preprocessed/
│   └── cross_time/
├── PreprocessResults_eeg_ieeg_cross_modal/
└── results/
```

Dataset roots can be changed with `DATA_ROOT`, `PREPROCESS_ROOT`, and `RESULT_ROOT`.

## Preprocessing

Cross-dataset scalp EEG preprocessing:

```bash
cd MyCode
python -m preprocessing.pipeline \
  --config preprocessing/configs/eeg_cross_dataset.yaml \
  --datasets chbmit tusz siena \
  --tasks detection prediction \
  --window-seconds 12 \
  --output-root data/PreprocessResults_eeg_cross_dataset
```

Cross-modal scalpEEGtoiEEG preprocessing:

```bash
cd MyCode
python -m preprocessing.pipeline \
  --config preprocessing/configs/eeg_ieeg_cross_modal.yaml \
  --datasets tusz chbmit siena epilepsy_ieeg thalamocortical_ieeg \
  --tasks detection prediction \
  --window-seconds 12 \
  --output-root data/PreprocessResults_eeg_ieeg_cross_modal
```

Cross-time cache construction:

```bash
cd MyCode
python preprocessing/build_chbmit_cross_time_cache.py --help
```

Run the same preprocessing commands with `--window-seconds 60` to build the 60s context-window cache.

## Main Experiment Launcher

All training and evaluation jobs are launched through `MyCode/train.sh`. The launcher exposes deterministic controls, GPU selection, seeds, batch size, workers, budgets, and model lists as environment variables.

Default reproducibility settings include:

- `GPU=0` for manual GPU selection;
- `RAND_SEED=1`;
- `DETERMINISTIC=1`;
- `BATCH_SIZE=128`;
- `NUM_WORKERS=8`;
- disabled data augmentation and dropout inside the launcher;
- deterministic CUDA-related environment variables where supported.

### Cross-dataset zero-shot transfer

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=dataset_transfer \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='tusz:chbmit chbmit:tusz siena:chbmit chbmit:siena' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0' \
SEEDS_TEXT='1 2 3' \
bash train.sh
```

### Target-domain adaptation

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=dataset_transfer \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='tusz:chbmit chbmit:tusz siena:chbmit chbmit:siena' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0 25 50 75 100' \
SEEDS_TEXT='1 2 3' \
bash train.sh
```

### Cross-time transfer

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=cross_time \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='chbmit_historical:chbmit_future' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0 25 50 75 100' \
SEEDS_TEXT='1 2 3' \
bash train.sh
```

### Cross-modal scalpEEGtoiEEG transfer

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=eeg_ieeg_transfer \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='tusz:epilepsy_ieeg tusz:thalamocortical_ieeg siena:epilepsy_ieeg siena:thalamocortical_ieeg chbmit:epilepsy_ieeg chbmit:thalamocortical_ieeg' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0 25 50 75 100' \
SEEDS_TEXT='1 2 3' \
bash train.sh
```

### iEEG localization

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
BATCH_SIZE=128 \
NUM_WORKERS=8 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=eeg_ieeg_localization \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain ScatterFormer TSMNet BF-EML BENDR CST' \
TASKS_TEXT='localization' \
DIRECTIONS_TEXT='tusz:epilepsy_ieeg' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0' \
SEEDS_TEXT='1 2 3' \
bash train.sh
```

### Reliability analysis

```bash
cd MyCode
GPU=0 \
RAND_SEED=1 \
DETERMINISTIC=1 \
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=epishift_reliability \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='all' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0 25 50 75 100' \
SEEDS_TEXT='1 2 3' \
EPISHIFT_CROSS_DATASET_RESULT_ROOT=$PWD/data/Results \
EPISHIFT_CROSS_MODAL_RESULT_ROOT=$PWD/data/results/cross_modal \
bash train.sh
```

## Outputs

Experiment outputs are written under the selected `RESULT_ROOT`. Depending on the task, outputs include:

- model checkpoints;
- prediction files;
- probability and calibration summaries;
- AUROC, AUPRC, balanced-accuracy, ECE, Brier, NLL, and AURC tables;
- localization summaries;
- sampling and run-audit artifacts;
- logs emitted by the launcher and task modules.

Training artifacts are intentionally retained to support reproducibility audits.

## Models

The repository includes RIVER and the benchmark baselines used by the paper. Baseline implementations are organized under `MyCode/models/baselines/`, while RIVER is organized under `MyCode/models/RIVER/`.

Implemented model names accepted by the launcher include:

```text
RIVER
BENDR
BF-EML
BIOT
CBraMod
CMMN
CST
EEGNet
EEGPT
EvoBrain
LaBraM
RandomForest
ScatterFormer
STEEGFormer
SVM
TSMNet
```

Some baselines may require pretrained weights or external assets governed by their original licenses. Such assets are not redistributed in this anonymous package.

## Reproducibility Notes

- All benchmark settings use subject-disjoint splits.
- Main tables are computed over independent seeds configured through `SEEDS_TEXT`.
- The default launcher exposes `GPU`, `RAND_SEED`, `BATCH_SIZE`, `NUM_WORKERS`, `WINDOWS_TEXT`, `BUDGETS_TEXT`, and model lists for reproducible reruns.
- Target-domain supervision is controlled by `BUDGETS_TEXT`, with `0` denoting zero-shot transfer and `25 50 75 100` denoting target-domain adaptation budgets.
- Reliability analysis uses the frozen prediction artifacts generated by the benchmark runs.

## Citation

```bibtex
@inproceedings{anonymous2027epitarbench,
  title = {EpiTAR-Bench: Benchmarking Transferability, Adaptation, and Reliability of Epilepsy Models},
  author = {Anonymous},
  booktitle = {International Conference on Learning Representations},
  year = {2027}
}
```

## License

This code release is distributed under the MIT License. Dataset use is governed by each dataset provider. Third-party model components and pretrained weights are governed by their original licenses.
