from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from models.baselines.SVM.benchmark_adapter import (
    build_localization_evidence,
    build_statistic_occlusion_localization_evidence,
    filter_budget_manifest,
    load_features,
    load_manifest,
)
from eeg_benchmark.tasks.cross_dataset import (
    add_mission_arguments,
    balance_prediction_training_frame,
    build_spec,
    patient_budget_rehearsal_indices,
    prepare_mission,
    resolve_in_domain_source_checkpoint,
    undersample_detection_training_frame,
    validate_mode_args,
    validate_zero_shot_reference,
    mission_sampling_summary,
)
from eeg_benchmark.tasks.cross_modal import (
    finalize_precomputed_interpretability,
    run_epilepsy_localization_task,
    safe_refresh_budget_interpretability,
)
from eeg_benchmark.engine import (
    build_channel_union,
    configure_reproducibility,
    evaluate_predictions,
    save_json,
    select_f1_threshold,
)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('benchmark.task1.randomforest')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler = logging.FileHandler(output_dir / 'train.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def print_model_table(args: argparse.Namespace, spec, channel_count: int) -> None:
    print('| Field | Value |')
    print('|:--|:--|')
    rows = {
        'Model': 'RandomForest',
        'Mission': spec.mission_type,
        'Mode': spec.mode,
        'Budget percent': spec.budget_percent,
        'Budget unit': spec.to_dict()['budget_unit'],
        'Window seconds': spec.window_seconds,
        'Task': spec.task,
        'Source': spec.source_dataset,
        'Target': spec.target_dataset,
        'Pretrained': False,
        'Channel policy': 'Shared semantic feature contract with channel mask',
        'Union channels': channel_count,
        'N estimators': args.n_estimators,
        'Max depth': args.max_depth,
        'Max features': args.max_features,
        'Min samples leaf': args.min_samples_leaf,
        'Seed': spec.seed,
        'GPU': 'CPU',
        'Statistics CPU workers': spec.stats_num_workers,
        'Bootstrap resamples': spec.bootstrap_resamples,
        'Training sampling': mission_sampling_summary(spec, single_fit=True),
        'Output': spec.output_dir,
    }
    for key, value in rows.items():
        print(f'| {key} | {value} |')


def _score(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(features)[:, 1].astype(np.float64)
    return expit(model.decision_function(features))


def _feature_importance(model, transformed_features: np.ndarray) -> np.ndarray:
    del transformed_features
    if hasattr(model, 'feature_importances_'):
        return np.asarray(model.feature_importances_, dtype=np.float32)
    raise ValueError('RandomForest does not expose feature importances')


def run(args: argparse.Namespace) -> None:
    validate_mode_args(args)
    if args.use_pretrained:
        raise ValueError('RandomForest does not use pretrained weights')
    spec = build_spec(args, 'RandomForest')
    reproducibility = configure_reproducibility(spec.seed, spec.deterministic)
    logger = setup_logger(spec.output_dir)
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(
        source_root / 'manifest.csv',
        target_root / 'manifest.csv',
        spec.source_dataset,
        spec.target_dataset,
        cross_modal_policy='inductive_permutation_invariant_native_channel_statistics',
    )
    contract.save(spec.output_dir / 'channel_union.json')
    print_model_table(args, spec, len(contract.channel_keys))
    model_arguments = {
        'n_estimators': int(args.n_estimators),
        'max_depth': args.max_depth,
        'max_features': args.max_features,
        'min_samples_leaf': int(args.min_samples_leaf),
        'random_seed': int(spec.seed),
    }
    save_json(spec.output_dir / 'args.json', {
        **spec.to_dict(),
        **model_arguments,
        'reproducibility': reproducibility,
        'budget_finetune_strategy': 'budget_refit_with_source_rehearsal' if spec.budget_percent > 0.0 else 'none',
        'source_rehearsal_fraction': spec.source_rehearsal_fraction,
    })
    if spec.task == 'localization':
        reference_candidates = (
            spec.source_reference_checkpoint_path,
            spec.source_reference_output_dir / 'best_model.joblib',
        )
        reference_model_path = next(
            (candidate for candidate in reference_candidates if candidate.is_file()),
            None,
        )
        if reference_model_path is None:
            raise FileNotFoundError(
                'Localization requires a completed detection reference model. '
                f'Checked: {[str(path) for path in reference_candidates]}'
            )
        best_model = joblib.load(reference_model_path)
        save_json(spec.output_dir / 'reference_checkpoint.json', {
            'path': str(reference_model_path),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'policy': 'reuse_completed_detection_checkpoint_for_soz_localization',
        })
        target_test = load_manifest(target_root, 'test')
        localization_evidence = build_statistic_occlusion_localization_evidence(
            best_model,
            target_root,
            target_test,
            contract,
            spec,
        )
        dataset_contract = json.loads(
            (Path(target_root) / 'dataset_contract.json').read_text(encoding='utf-8')
        )
        raw_root = dataset_contract.get('source_root')
        if not raw_root:
            raise ValueError('Localization requires source_root in target dataset contract')
        summary = run_epilepsy_localization_task(
            localization_evidence,
            raw_root,
            spec.output_dir,
        )
        localization_evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        save_json(spec.output_dir / 'metrics.json', summary)
        save_json(spec.output_dir / 'run_summary.json', {
            'status': 'complete',
            'task': spec.task,
            'localization': summary,
            'reference_checkpoint': str(reference_model_path),
        })
        return
    source_dev = load_manifest(source_root, 'dev')
    selection_root = source_root
    selection_frame = source_dev
    threshold_source = 'source_dev_max_f1'
    if budget_selection is not None:
        selection_root = target_root
        selection_frame = filter_budget_manifest(load_manifest(target_root, 'dev'), budget_selection.clip_ids('dev'), 'dev')
        threshold_source = 'budgeted_target_dev_max_f1'
    target_test = load_manifest(target_root, 'test')
    x_dev, y_dev, dev_rows = load_features(selection_root, selection_frame, contract, 'Selection features')
    x_test, y_test, test_rows = load_features(target_root, target_test, contract, 'Target test features')
    if spec.is_in_domain:
        source_checkpoint, source_reference = resolve_in_domain_source_checkpoint(spec, 'best_model.joblib', model_arguments, contract)
        best_model = joblib.load(source_checkpoint)
        save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
    else:
        source_train = load_manifest(source_root, 'train')
        if budget_selection is None and spec.task == 'detection':
            source_train, _ = undersample_detection_training_frame(source_train, dataset=spec.source_dataset, split='train', seed=spec.undersample_seed, audit_root=spec.output_dir / 'sampling' / 'source_train')
        elif budget_selection is None and spec.task == 'prediction':
            source_train, _ = balance_prediction_training_frame(source_train, dataset=spec.source_dataset, split='train', seed=spec.undersample_seed, epoch=0, audit_root=spec.output_dir / 'sampling' / 'source_train')
        if budget_selection is None:
            x_train, y_train, _ = load_features(source_root, source_train, contract, 'Source train features')
        else:
            zero_checkpoint = spec.zero_shot_output_dir / 'best_model.joblib'
            validate_zero_shot_reference(spec, zero_checkpoint, model_arguments, contract)
            save_json(spec.output_dir / 'source_checkpoint_reference.json', {'path': str(zero_checkpoint), 'policy': 'validated_source_reference_then_budget_refit'})
            target_train = filter_budget_manifest(load_manifest(target_root, 'train'), budget_selection.clip_ids('train'), 'train')
            indices, report = patient_budget_rehearsal_indices(
                target_train,
                source_train,
                seed=spec.undersample_seed,
                epoch=0,
                rehearsal_fraction=spec.source_rehearsal_fraction,
                target_dataset=spec.target_dataset,
                source_dataset=spec.source_dataset,
                task=spec.task,
                prediction_negative_to_positive_ratio=1.0 if spec.task == 'prediction' else 2.0,
            )
            target_mask = indices < len(target_train)
            target_rows = target_train.iloc[indices[target_mask]].reset_index(drop=True)
            source_rows = source_train.iloc[indices[~target_mask] - len(target_train)].reset_index(drop=True)
            x_target, y_target, _ = load_features(target_root, target_rows, contract, 'Budgeted target train features')
            x_source, y_source, _ = load_features(source_root, source_rows, contract, 'Source rehearsal features')
            x_train = np.concatenate([x_target, x_source], axis=0)
            y_train = np.concatenate([y_target, y_source], axis=0)
            (spec.output_dir / 'sampling').mkdir(parents=True, exist_ok=True)
            save_json(spec.output_dir / 'sampling' / 'random_forest_budget_refit.json', report)
        order = np.random.default_rng(int(spec.seed)).permutation(len(y_train))
        x_train = x_train[order]
        y_train = y_train[order]
        best_model = RandomForestClassifier(
            n_estimators=int(args.n_estimators),
            max_depth=args.max_depth,
            max_features=args.max_features,
            min_samples_leaf=int(args.min_samples_leaf),
            class_weight=None,
            random_state=int(spec.seed),
            n_jobs=max(1, int(args.stats_num_workers)),
        )
        for _ in tqdm(range(1), desc='Train RandomForest', unit='fit', colour='green'):
            best_model.fit(x_train, y_train)
    dev_score = _score(best_model, x_dev)
    threshold = select_f1_threshold(y_dev, dev_score)
    test_score = _score(best_model, x_test)
    predictions = test_rows.copy()
    predictions['label'] = y_test
    predictions['score'] = test_score
    predictions['predicted_label'] = (test_score >= threshold).astype(np.int64)
    selection_predictions = dev_rows.copy()
    selection_predictions['label'] = y_dev
    selection_predictions['score'] = dev_score
    selection_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    metrics = evaluate_predictions(predictions, spec.task, threshold, spec.output_dir, bootstrap_seed=spec.seed, bootstrap_resamples=spec.bootstrap_resamples, bootstrap_workers=spec.stats_num_workers)
    predictions.to_csv(spec.output_dir / 'predictions.csv', index=False)
    joblib.dump(best_model, spec.output_dir / 'best_model.joblib')
    interpretability = finalize_precomputed_interpretability(spec, x_test, predictions[['clip_id', 'patient_id', 'label', 'score']].copy(), source_root, target_root, predictions)
    save_json(spec.output_dir / 'run_summary.json', {
        'status': 'complete',
        'selection_metric': 'validation_auroc',
        'validation_auroc': float(roc_auc_score(y_dev, dev_score)) if np.unique(y_dev).size == 2 else float('nan'),
        'threshold_source': threshold_source,
        'metrics': metrics,
        'interpretability': interpretability,
        'budget_interpretability': safe_refresh_budget_interpretability(spec),
    })
    logger.info('Completed RandomForest output=%s', spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='RandomForest benchmark baseline')
    parser.add_argument('--model', choices=['RandomForest'], default='RandomForest')
    add_mission_arguments(parser)
    parser.add_argument('--n-estimators', type=int, default=500)
    parser.add_argument('--max-depth', type=int, default=None)
    parser.add_argument('--max-features', default='sqrt')
    parser.add_argument('--min-samples-leaf', type=int, default=2)
    return parser


if __name__ == '__main__':
    run(build_parser().parse_args())
