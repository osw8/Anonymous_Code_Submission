from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.special import expit
from scipy.stats import kurtosis, skew
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from eeg_benchmark.engine import align_preprocessed_clip, build_channel_union
from eeg_benchmark.tasks.cross_dataset import patient_budget_rehearsal_indices
from eeg_benchmark.engine import evaluate_predictions, save_json, select_f1_threshold
from eeg_benchmark.engine import configure_reproducibility
from eeg_benchmark.tasks.cross_dataset import add_mission_arguments, build_spec, prepare_mission, resolve_in_domain_source_checkpoint, validate_mode_args, validate_zero_shot_reference
from eeg_benchmark.tasks.cross_dataset import balance_prediction_training_frame, mission_sampling_summary, undersample_detection_training_frame
from eeg_benchmark.tasks.cross_modal import (
    build_fixed_xai_cohort,
    canonical_contact_name,
    finalize_precomputed_interpretability,
    run_epilepsy_localization_task,
    safe_refresh_budget_interpretability,
)


BANDS = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 45.0),
    ("mid_gamma", 45.0, 55.0),
    ("high_gamma", 65.0, 95.0),
)


def trapezoid_integral(values: np.ndarray, coordinates: np.ndarray, axis: int = -1) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(coordinates, dtype=np.float64)
    normalized_axis = axis if axis >= 0 else y.ndim + axis
    output_shape = y.shape[:normalized_axis] + y.shape[normalized_axis + 1:]
    if y.shape[normalized_axis] == 0:
        return np.zeros(output_shape, dtype=np.float64)
    if y.shape[normalized_axis] == 1:
        return np.zeros(output_shape, dtype=np.float64)
    moved = np.moveaxis(y, axis, -1)
    widths = np.diff(x).reshape((1,) * (moved.ndim - 1) + (-1,))
    integrated = np.sum((moved[..., 1:] + moved[..., :-1]) * widths * 0.5, axis=-1)
    return integrated.astype(np.float64)


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("benchmark.task1.svm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def hjorth_features(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = np.diff(signal, axis=-1)
    second = np.diff(first, axis=-1)
    activity = np.var(signal, axis=-1)
    first_variance = np.var(first, axis=-1)
    second_variance = np.var(second, axis=-1)
    mobility = np.sqrt(first_variance / np.maximum(activity, 1e-12))
    second_mobility = np.sqrt(second_variance / np.maximum(first_variance, 1e-12))
    complexity = second_mobility / np.maximum(mobility, 1e-12)
    return activity, mobility, complexity


def extract_channel_features(signal: np.ndarray, sfreq: int) -> np.ndarray:
    frequencies, density = welch(signal, fs=sfreq, nperseg=min(1024, signal.shape[-1]), axis=-1)
    total_mask = (
        (frequencies >= 0.5)
        & (frequencies <= min(95.0, sfreq / 2.0 - 1e-6))
        & ~((frequencies >= 55.0) & (frequencies < 65.0))
    )
    total_power = trapezoid_integral(
        density[:, total_mask], frequencies[total_mask], axis=-1
    )
    spectral_probability = density[:, total_mask] / np.maximum(density[:, total_mask].sum(axis=-1, keepdims=True), 1e-12)
    spectral_entropy = -(spectral_probability * np.log(np.maximum(spectral_probability, 1e-12))).sum(axis=-1)
    activity, mobility, complexity = hjorth_features(signal)
    features = [
        np.log1p(activity),
        mobility,
        complexity,
        np.mean(np.abs(np.diff(signal, axis=-1)), axis=-1),
        np.sqrt(np.mean(np.square(signal), axis=-1)),
        np.nan_to_num(skew(signal, axis=-1, bias=False), nan=0.0),
        np.nan_to_num(kurtosis(signal, axis=-1, fisher=True, bias=False), nan=0.0),
        spectral_entropy,
    ]
    for _, low, high in BANDS:
        mask = (frequencies >= low) & (frequencies < min(high, sfreq / 2.0))
        power = (
            trapezoid_integral(density[:, mask], frequencies[mask], axis=-1)
            if mask.any()
            else np.zeros(signal.shape[0])
        )
        features.extend([np.log1p(power), power / np.maximum(total_power, 1e-12)])
    return np.stack(features, axis=1).astype(np.float32)


def aggregate_native_electrode_features(channel_features: np.ndarray) -> np.ndarray:
    if channel_features.ndim != 2 or channel_features.shape[0] == 0:
        raise ValueError('Native electrode feature aggregation requires non-empty channel features')
    values = np.asarray(channel_features, dtype=np.float32)
    quantiles = np.quantile(values, [0.25, 0.5, 0.75], axis=0).astype(np.float32)
    summary = np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            quantiles[0],
            quantiles[1],
            quantiles[2],
            values.min(axis=0),
            values.max(axis=0),
            np.asarray([float(values.shape[0])], dtype=np.float32),
        ]
    )
    if not np.isfinite(summary).all():
        raise ValueError('Native electrode aggregate features contain non-finite values')
    return summary.astype(np.float32)


def load_features(
    cache_root: Path,
    frame: pd.DataFrame,
    contract,
    description: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    vectors = []
    labels = []
    retained_rows = []
    progress = tqdm(frame.itertuples(index=False), total=len(frame), desc=description, unit="clip", colour="green")
    for row in progress:
        with np.load(cache_root / row.relative_path, allow_pickle=False) as archive:
            signal = np.asarray(archive["eeg"], dtype=np.float32)
            names = [str(value) for value in archive["channel_names"]]
            sfreq = int(archive["sfreq"])
            label = int(archive["label"])
            channel_mean_uv = np.asarray(archive['channel_mean_uv'], dtype=np.float32) if 'channel_mean_uv' in archive else None
            channel_std_uv = np.asarray(archive['channel_std_uv'], dtype=np.float32) if 'channel_std_uv' in archive else None
            metadata = json.loads(str(archive['metadata_json'].item())) if 'metadata_json' in archive else {}
        if signal.shape[0] != len(names):
            raise ValueError(f"Channel metadata mismatch for {row.relative_path}")
        signal, channel_mask = align_preprocessed_clip(
            signal,
            names,
            contract,
            channel_mean_uv,
            channel_std_uv,
            metadata.get('normalization'),
            str(row.dataset),
        )
        channel_features = extract_channel_features(signal, sfreq)
        if contract.policy == 'native_electrode_set_attention':
            vector = aggregate_native_electrode_features(channel_features)
        else:
            vector = np.concatenate(
                [channel_features.reshape(-1), channel_mask.astype(np.float32)]
            ).astype(np.float32)
        vectors.append(vector)
        labels.append(label)
        retained_rows.append(row._asdict())
    return np.stack(vectors), np.asarray(labels, dtype=np.int64), pd.DataFrame(retained_rows)


def build_localization_evidence(
    transformed_features: np.ndarray,
    labels: np.ndarray,
    rows: pd.DataFrame,
    contract,
    classifier,
) -> pd.DataFrame:
    channel_count = len(contract.channel_keys)
    feature_dim = transformed_features.shape[1]
    if feature_dim <= channel_count:
        raise ValueError('SVM feature vector is too small for localization evidence')
    per_channel_feature_dim = (feature_dim - channel_count) // channel_count
    if per_channel_feature_dim * channel_count + channel_count != feature_dim:
        raise ValueError('SVM feature vector cannot be reshaped into channel groups')
    coefficient = np.asarray(classifier.coef_, dtype=np.float32).reshape(-1)
    if coefficient.size != feature_dim:
        raise ValueError('SVM coefficient vector does not match transformed features')
    weighted = np.abs(transformed_features * coefficient)
    weighted = weighted[:, : channel_count * per_channel_feature_dim]
    channel_scores = weighted.reshape(
        transformed_features.shape[0], channel_count, per_channel_feature_dim
    ).sum(axis=2)
    evidence_rows = []
    for row_index, row in enumerate(rows.itertuples(index=False)):
        for channel_index, channel_key in enumerate(contract.channel_keys):
            evidence_rows.append({
                'patient_id': str(row.patient_id),
                'label': int(labels[row_index]),
                'contact_key': str(channel_key),
                'evidence': float(channel_scores[row_index, channel_index]),
            })
    return pd.DataFrame(evidence_rows)


def model_positive_score(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    if hasattr(model, 'decision_function'):
        return expit(np.asarray(model.decision_function(features), dtype=np.float64))
    raise ValueError('Traditional ML localization requires predict_proba or decision_function')


def statistic_feature_vector_from_native(
    signal: np.ndarray,
    names: list[str],
    contract,
    channel_mean_uv: np.ndarray | None,
    channel_std_uv: np.ndarray | None,
    normalization: str | None,
    dataset: str,
    sfreq: int,
) -> np.ndarray:
    aligned, channel_mask = align_preprocessed_clip(
        signal,
        names,
        contract,
        channel_mean_uv,
        channel_std_uv,
        normalization,
        dataset,
    )
    channel_features = extract_channel_features(aligned, sfreq)
    return np.concatenate(
        [channel_features.reshape(-1), channel_mask.astype(np.float32)]
    ).astype(np.float32)


def build_statistic_occlusion_localization_evidence(
    model,
    target_root: Path,
    target_frame: pd.DataFrame,
    contract,
    spec,
) -> pd.DataFrame:
    cohort = build_fixed_xai_cohort(
        target_frame,
        dataset=spec.target_dataset,
        task=spec.task,
        split='test',
        seed=spec.budget_seed,
        maximum_clips_per_patient_per_class=max(
            1, min(32, int(spec.interpretability_max_clips))
        ),
        require_both_classes_per_patient=True,
    )
    selected_ids = set(cohort.frame['clip_id'].astype(str))
    selected = target_frame[
        target_frame['clip_id'].astype(str).isin(selected_ids)
    ].reset_index(drop=True)
    rows: list[dict[str, object]] = []
    progress = tqdm(
        selected.itertuples(index=False),
        total=len(selected),
        desc='Native contact occlusion',
        unit='clip',
        colour='green',
    )
    for row in progress:
        with np.load(target_root / row.relative_path, allow_pickle=False) as archive:
            signal = np.asarray(archive['eeg'], dtype=np.float32)
            names = [str(value) for value in archive['channel_names']]
            sfreq = int(archive['sfreq'])
            label = int(archive['label'])
            channel_mean_uv = (
                np.asarray(archive['channel_mean_uv'], dtype=np.float32)
                if 'channel_mean_uv' in archive else None
            )
            channel_std_uv = (
                np.asarray(archive['channel_std_uv'], dtype=np.float32)
                if 'channel_std_uv' in archive else None
            )
            metadata = (
                json.loads(str(archive['metadata_json'].item()))
                if 'metadata_json' in archive else {}
            )
            channel_types = (
                [str(value) for value in archive['channel_types']]
                if 'channel_types' in archive else ['UNKNOWN'] * len(names)
            )
            channel_positions = (
                np.asarray(archive['channel_positions'], dtype=np.float32)
                if 'channel_positions' in archive
                else np.full((len(names), 3), np.nan, dtype=np.float32)
            )
        base_vector = statistic_feature_vector_from_native(
            signal,
            names,
            contract,
            channel_mean_uv,
            channel_std_uv,
            metadata.get('normalization'),
            str(row.dataset),
            sfreq,
        )[None]
        base_score = float(model_positive_score(model, base_vector)[0])
        evidence_values = []
        for channel_index in range(signal.shape[0]):
            keep = np.ones(signal.shape[0], dtype=bool)
            keep[channel_index] = False
            occluded_mean = (
                channel_mean_uv[keep]
                if channel_mean_uv is not None
                and channel_mean_uv.shape[0] == signal.shape[0]
                else channel_mean_uv
            )
            occluded_std = (
                channel_std_uv[keep]
                if channel_std_uv is not None
                and channel_std_uv.shape[0] == signal.shape[0]
                else channel_std_uv
            )
            vector = statistic_feature_vector_from_native(
                signal[keep],
                [name for index, name in enumerate(names) if keep[index]],
                contract,
                occluded_mean,
                occluded_std,
                metadata.get('normalization'),
                str(row.dataset),
                sfreq,
            )[None]
            occluded_score = float(model_positive_score(model, vector)[0])
            evidence_values.append(max(base_score - occluded_score, 0.0))
        evidence_array = np.asarray(evidence_values, dtype=np.float64)
        total = float(evidence_array.sum())
        if total <= 0.0:
            evidence_array = np.full(len(evidence_array), 1.0 / max(len(evidence_array), 1))
        else:
            evidence_array = evidence_array / total
        for channel_index, name in enumerate(names):
            position = channel_positions[channel_index]
            rows.append({
                'dataset': str(spec.target_dataset),
                'task': str(spec.task),
                'patient_id': str(row.patient_id),
                'clip_id': str(row.clip_id),
                'label': int(label),
                'source_relative_path': str(
                    getattr(row, 'source_relative_path', row.relative_path)
                ),
                'clip_start_seconds': float(row.clip_start_seconds),
                'clip_end_seconds': float(row.clip_end_seconds),
                'seizure_intervals_json': str(getattr(row, 'seizure_intervals_json', '[]')),
                'contact_name': str(name),
                'contact_key': canonical_contact_name(name),
                'channel_type': str(channel_types[channel_index]).upper(),
                'x': float(position[0]) if np.isfinite(position[0]) else np.nan,
                'y': float(position[1]) if np.isfinite(position[1]) else np.nan,
                'z': float(position[2]) if np.isfinite(position[2]) else np.nan,
                'evidence': float(evidence_array[channel_index]),
                'base_score': base_score,
                'attribution': 'native_contact_leave_one_out_positive_score_drop',
            })
    return pd.DataFrame(rows)


def load_manifest(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "manifest.csv", dtype={"patient_id": str, "clip_id": str})
    selected = frame[frame["split"] == split].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"No {split} clips in {root}")
    return selected


def filter_budget_manifest(
    frame: pd.DataFrame,
    clip_ids: tuple[str, ...],
    split: str,
) -> pd.DataFrame:
    selected = frame.loc[
        frame['clip_id'].astype(str).isin(set(clip_ids))
    ].reset_index(drop=True)
    if selected.empty or set(selected['label'].astype(int)) != {0, 1}:
        raise ValueError(f'Budgeted target {split} must retain both classes')
    return selected


def seed_training_order(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if len(labels) != len(features):
        raise ValueError('SVM feature and label counts differ before seeded ordering')
    generator = np.random.default_rng(int(seed))
    order = generator.permutation(len(labels))
    preview = [int(value) for value in order[: min(16, len(order))]]
    report = {
        'policy': (
            'seeded_order_only_sample_set_unchanged_for_linear_svm_solver'
        ),
        'seed': int(seed),
        'sample_count': int(len(labels)),
        'first_indices_after_permutation': preview,
    }
    return features[order], labels[order], report


def print_model_table(args: argparse.Namespace, spec, channel_count: int) -> None:
    print("| Field | Value |")
    print("|:--|:--|")
    rows = {
        "Model": "LinearSVM",
        'Mission': spec.mission_type,
        'In-domain source policy': (
            'reuse canonical completed source model'
            if spec.is_in_domain else 'train source model'
        ),
        "Mode": spec.mode,
        "Budget percent": spec.budget_percent,
        'Budget unit': spec.to_dict()['budget_unit'],
        "Window seconds": spec.window_seconds,
        "Task": spec.task,
        "Source": spec.source_dataset,
        "Target": spec.target_dataset,
        "Pretrained": False,
        "Channel policy": "Shared semantic channel contract with channel mask",
        "Union channels": channel_count,
        "C grid": args.c_grid,
        "Seed": spec.seed,
        "GPU": "CPU",
        'Statistics CPU workers': spec.stats_num_workers,
        'Bootstrap resamples': spec.bootstrap_resamples,
        'Early stopping min delta': 'not_applicable',
        'Training sampling': mission_sampling_summary(spec, single_fit=True),
        'Budget adaptation': (
            'budget_refit_with_source_rehearsal'
            if spec.budget_percent > 0.0 else 'none'
        ),
        'Undersample seed': spec.undersample_seed,
        'Interpretability': spec.generate_interpretability,
        "Output": spec.output_dir,
    }
    for key, value in rows.items():
        print(f"| {key} | {value} |")


def run(args: argparse.Namespace) -> None:
    validate_mode_args(args)
    if args.use_pretrained:
        raise ValueError('SVM does not use pretrained weights')
    spec = build_spec(args, "SVM")
    svm_random_seed = int(spec.seed)
    reproducibility = configure_reproducibility(svm_random_seed, spec.deterministic)
    logger = setup_logger(spec.output_dir)
    logger.info(
        'SVM_RANDOM_SEED_CONTRACT solver_seed=%s bootstrap_seed=%s '
        'budget_seed=%s undersample_seed=%s',
        svm_random_seed,
        svm_random_seed,
        spec.budget_seed,
        spec.undersample_seed,
    )
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(
        source_root / "manifest.csv",
        target_root / "manifest.csv",
        spec.source_dataset,
        spec.target_dataset,
        cross_modal_policy="inductive_permutation_invariant_native_channel_statistics",
    )
    contract.save(spec.output_dir / "channel_union.json")
    print_model_table(args, spec, len(contract.channel_keys))
    (spec.output_dir / "args.json").write_text(
        json.dumps({
            **spec.to_dict(),
            "c_grid": args.c_grid,
            "max_iter": args.max_iter,
            "reproducibility": reproducibility,
            'svm_random_seed_contract': {
                'solver_seed': svm_random_seed,
                'bootstrap_seed': svm_random_seed,
                'training_order_seed': svm_random_seed,
                'source_budget_seed': int(spec.budget_seed),
                'source_undersample_seed': int(spec.undersample_seed),
                'policy': (
                    'SVM solver, training row order, and evaluation bootstrap '
                    'follow main RAND_SEED via spec.seed; budget and '
                    'undersampling seeds remain fixed for fair sample '
                    'selection.'
                ),
            },
            'budget_finetune_strategy': (
                'budget_refit_with_source_rehearsal'
                if spec.budget_percent > 0.0 else 'none'
            ),
            'source_rehearsal_fraction': spec.source_rehearsal_fraction,
        }, indent=2),
        encoding="utf-8",
    )
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
        if (
            getattr(contract, 'policy', None)
            == 'inductive_permutation_invariant_native_channel_statistics'
        ):
            target_test = load_manifest(target_root, 'test')
            localization_evidence = build_statistic_occlusion_localization_evidence(
                best_model,
                target_root,
                target_test,
                contract,
                spec,
            )
        else:
            x_test, y_test, test_rows = load_features(target_root, load_manifest(target_root, "test"), contract, "Target test features")
            transformed_test = best_model.named_steps['scaler'].transform(x_test).astype(np.float32)
            localization_evidence = build_localization_evidence(
                transformed_test,
                y_test,
                test_rows,
                contract,
                best_model.named_steps['classifier'],
            )
        dataset_contract = json.loads(
            (Path(target_root) / 'dataset_contract.json').read_text(encoding='utf-8')
        )
        raw_root = dataset_contract.get('source_root')
        if not raw_root:
            raise ValueError('Localization requires source_root in target dataset contract')
        localization_summary = run_epilepsy_localization_task(
            localization_evidence,
            raw_root,
            spec.output_dir,
        )
        localization_evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        (spec.output_dir / 'metrics.json').write_text(
            json.dumps(localization_summary, indent=2, allow_nan=True),
            encoding='utf-8',
        )
        (spec.output_dir / 'run_summary.json').write_text(
            json.dumps({
                'status': 'complete',
                'task': spec.task,
                'localization': localization_summary,
                'reference_checkpoint': str(reference_model_path),
                'reference_output_dir': str(spec.source_reference_output_dir),
            }, indent=2, allow_nan=True),
            encoding='utf-8',
        )
        logger.info("Completed localization task with output %s", spec.output_dir)
        return
    source_dev = load_manifest(source_root, "dev")
    selection_root = source_root
    selection_frame = source_dev
    threshold_source = 'source_dev_max_f1'
    selection_metric = 'source_dev_auroc'
    if budget_selection is not None:
        selection_root = target_root
        selection_frame = filter_budget_manifest(
            load_manifest(target_root, 'dev'),
            budget_selection.clip_ids('dev'),
            'dev',
        )
        threshold_source = 'budgeted_target_dev_max_f1'
        selection_metric = 'budgeted_target_dev_auroc'
    target_test = load_manifest(target_root, "test")
    x_dev, y_dev, dev_rows = load_features(
        selection_root,
        selection_frame,
        contract,
        'Fixed target dev features'
        if budget_selection is not None else 'Source dev features',
    )
    x_test, y_test, test_rows = load_features(target_root, target_test, contract, "Target test features")
    if spec.is_in_domain:
        source_checkpoint, source_reference = resolve_in_domain_source_checkpoint(
            spec,
            'best_model.joblib',
            {'c_grid': args.c_grid, 'max_iter': args.max_iter},
            contract,
        )
        best_model = joblib.load(source_checkpoint)
        best_c = float(best_model.named_steps['classifier'].C)
        (spec.output_dir / 'source_checkpoint_reference.json').write_text(
            json.dumps(source_reference, indent=2),
            encoding='utf-8',
        )
        best_score = float(roc_auc_score(y_dev, expit(best_model.decision_function(x_dev))))
        logger.info('In-domain source model loaded without source retraining: %s', source_checkpoint)
    else:
        source_train = load_manifest(source_root, "train")
        if budget_selection is None and spec.task == 'detection':
            source_train, _ = undersample_detection_training_frame(
                source_train,
                dataset=spec.source_dataset,
                split='train',
                seed=spec.undersample_seed,
                audit_root=spec.output_dir / 'sampling' / 'source_train',
            )
        elif (
            budget_selection is None
            and spec.task == 'prediction'
        ):
            source_train, _ = balance_prediction_training_frame(
                source_train,
                dataset=spec.source_dataset,
                split='train',
                seed=spec.undersample_seed,
                epoch=0,
                audit_root=spec.output_dir / 'sampling' / 'source_train',
            )
        if budget_selection is None:
            x_train, y_train, _ = load_features(
                source_root, source_train, contract, 'Source train features'
            )
        else:
            zero_checkpoint = spec.zero_shot_output_dir / 'best_model.joblib'
            validate_zero_shot_reference(
                spec,
                zero_checkpoint,
                {'c_grid': args.c_grid, 'max_iter': args.max_iter},
                contract,
            )
            (spec.output_dir / 'source_checkpoint_reference.json').write_text(
                json.dumps({
                    'path': str(zero_checkpoint),
                    'policy': 'validated_source_reference_then_budget_refit',
                    'reason': 'LinearSVC_has_no_incremental_finetuning_interface',
                }, indent=2),
                encoding='utf-8',
            )
            target_train = filter_budget_manifest(
                load_manifest(target_root, 'train'),
                budget_selection.clip_ids('train'),
                'train',
            )
            rehearsal_indices, rehearsal_report = patient_budget_rehearsal_indices(
                target_train,
                source_train,
                seed=spec.undersample_seed,
                epoch=0,
                rehearsal_fraction=spec.source_rehearsal_fraction,
                target_dataset=spec.target_dataset,
                source_dataset=spec.source_dataset,
                task=spec.task,
                prediction_negative_to_positive_ratio=(
                    1.0
                    if spec.task == 'prediction'
                    else 2.0
                ),
            )
            target_mask = rehearsal_indices < len(target_train)
            target_rows = target_train.iloc[
                rehearsal_indices[target_mask]
            ].reset_index(drop=True)
            source_rows = source_train.iloc[
                rehearsal_indices[~target_mask] - len(target_train)
            ].reset_index(drop=True)
            x_target, y_target, _ = load_features(
                target_root,
                target_rows,
                contract,
                'Budgeted target train features',
            )
            x_source, y_source, _ = load_features(
                source_root,
                source_rows,
                contract,
                'Source rehearsal features',
            )
            x_train = np.concatenate([x_target, x_source], axis=0)
            y_train = np.concatenate([y_target, y_source], axis=0)
            (spec.output_dir / 'sampling').mkdir(parents=True, exist_ok=True)
            (spec.output_dir / 'sampling' / 'svm_budget_refit.json').write_text(
                json.dumps(rehearsal_report, indent=2), encoding='utf-8'
            )
        x_train, y_train, order_report = seed_training_order(
            x_train,
            y_train,
            svm_random_seed,
        )
        (spec.output_dir / 'sampling').mkdir(parents=True, exist_ok=True)
        (spec.output_dir / 'sampling' / 'svm_training_order.json').write_text(
            json.dumps(order_report, indent=2), encoding='utf-8'
        )
        logger.info(
            'Applied SVM seeded training order seed=%s sample_count=%s',
            svm_random_seed,
            len(y_train),
        )
        best_model = None
        best_c = None
        best_score = -np.inf
        for c_value in args.c_grid:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("classifier", LinearSVC(C=c_value, class_weight=None, random_state=svm_random_seed, max_iter=args.max_iter, dual=True)),
                ]
            )
            model.fit(x_train, y_train)
            dev_score = expit(model.decision_function(x_dev))
            dev_auroc = roc_auc_score(y_dev, dev_score)
            logger.info("Validation C=%s AUROC=%.8f", c_value, dev_auroc)
            if dev_auroc > best_score:
                best_score = float(dev_auroc)
                best_c = float(c_value)
                best_model = model
        if best_model is None:
            raise RuntimeError("SVM selection failed")
    dev_score = expit(best_model.decision_function(x_dev))
    selection_predictions = dev_rows.copy()
    selection_predictions['label'] = y_dev
    selection_predictions['score'] = dev_score
    selection_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    threshold = select_f1_threshold(y_dev, dev_score)
    test_score = expit(best_model.decision_function(x_test))
    predictions = test_rows.copy()
    predictions["label"] = y_test
    predictions["score"] = test_score
    predictions["predicted_label"] = (test_score >= threshold).astype(np.int64)
    metrics = evaluate_predictions(
        predictions,
        spec.task,
        threshold,
        spec.output_dir,
        bootstrap_seed=svm_random_seed,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_workers=spec.stats_num_workers,
    )
    transformed_test = best_model.named_steps['scaler'].transform(x_test).astype(np.float32)
    embedding_metadata = predictions[['clip_id', 'patient_id', 'label', 'score']].copy()
    interpretability = finalize_precomputed_interpretability(
        spec,
        transformed_test,
        embedding_metadata,
        source_root,
        target_root,
        predictions,
    )
    if spec.mission_type == 'eeg_ieeg_localization' and spec.target_dataset == 'epilepsy_ieeg':
        dataset_contract = json.loads(
            (Path(target_root) / 'dataset_contract.json').read_text(encoding='utf-8')
        )
        raw_root = dataset_contract.get('source_root')
        if raw_root:
            localization_evidence = build_localization_evidence(
                transformed_test,
                y_test,
                test_rows,
                contract,
                best_model.named_steps['classifier'],
            )
            localization_summary = run_epilepsy_localization_task(
                localization_evidence,
                raw_root,
                spec.output_dir,
            )
            interpretability['I4'] = localization_summary
        else:
            interpretability['I4'] = {'status': 'skipped_missing_source_root_contract'}
    joblib.dump(best_model, spec.output_dir / "best_model.joblib")
    summary = {
        "status": "complete",
        "selection_metric": selection_metric,
        "best_c": best_c,
        "source_dev_auroc": (
            best_score if budget_selection is None else None
        ),
        'budgeted_target_dev_auroc': (
            best_score if budget_selection is not None else None
        ),
        'validation_auroc': best_score,
        "threshold_source": threshold_source,
        "metrics": metrics,
        'interpretability': interpretability,
    }
    (spec.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    (spec.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    logger.info("Completed target test with output %s", spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task1 cross-dataset Linear SVM baseline")
    parser.add_argument("--model", choices=["SVM"], default="SVM")
    add_mission_arguments(parser)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--max-iter", type=int, default=10000)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
