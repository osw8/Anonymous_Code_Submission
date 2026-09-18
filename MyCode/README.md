# EEG Transfer Benchmark

This repository provides a reproducible EEG benchmark for cross-dataset, cross-time, cross-modal, and iEEG localization experiments. It includes the RIVER model, baseline integrations, shared preprocessing and evaluation contracts, reliability analysis, and experiment artifact preservation. RIVER ablation experiments are excluded from this release.

## Scope

Supported experiment groups are:

- Cross-dataset detection and prediction
- Cross-time detection and prediction
- Scalp EEG to iEEG transfer
- iEEG localization and in-domain evaluation
- Reliability and probability-quality analysis

The model registry contains RIVER, BENDR, BF-EML, BIOT, CBraMod, CMMN, CST, EEGNet, EEGPT, EvoBrain, LaBraM, RandomForest, ScatterFormer, STEEGFormer, SVM, and TSMNet.

## Layout

```text
MyCode/
  eeg_benchmark/       Shared experiment engine, tasks, and analysis
  models/RIVER/         RIVER implementation and adapter
  models/baselines/     Baseline implementations and adapters
  preprocessing/        Dataset preprocessing entry points
  requirement.txt       Pinned Python dependencies
  setup_environment.sh  Public virtual-environment setup script
  train.sh              Unified experiment launcher
```

## Environment

Create an isolated environment from the single dependency manifest:

```bash
bash setup_environment.sh
source .venv/bin/activate
```

The installer uses the active Python package index and does not require machine-specific configuration.

## Running experiments

The public launcher accepts project, data, task, model, budget, seed, and GPU settings through environment variables. Data should be placed below `data/` or supplied through `DATA_ROOT`.

```bash
PROJECT_ROOT=$PWD \
DATA_ROOT=$PWD/data \
MISSION=dataset_transfer \
MODELS_TEXT='RIVER BIOT CBraMod EEGPT LaBraM STEEGFormer EEGNet EvoBrain SVM ScatterFormer CMMN TSMNet BF-EML RandomForest BENDR CST' \
TASKS_TEXT='detection prediction' \
DIRECTIONS_TEXT='tusz:chbmit chbmit:tusz' \
WINDOWS_TEXT='12 60' \
BUDGETS_TEXT='0 25 50 75 100' \
bash train.sh
```

Use `MISSION=cross_time` for the dedicated cross-time task. The equivalent Python entry point is:

```bash
python -m eeg_benchmark.tasks.cross_time --help
```

Before training, the launcher prints a Markdown model-information table. Training logs and complete run artifacts are written below the configured data and result roots. GPU selection is explicit through `GPU`; no automatic GPU selection is performed.

## Reproducibility

The launcher exposes deterministic controls, seeds, batch size, worker count, model hyperparameters, data roots, result roots, and budget settings. It records the active configuration with each run and retains checkpoints, predictions, metrics, histories, logs, and sampling audits.

## License

The repository-level code is released under the MIT License in `LICENSE`.
Third-party license files remain in the corresponding model directories. Dataset files and pretrained checkpoints are not included in this source release.
