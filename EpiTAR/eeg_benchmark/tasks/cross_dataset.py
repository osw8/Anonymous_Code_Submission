
from __future__ import annotations




import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


PATIENT_BUDGET_PROTOCOL = 'nested_complete_patients_train_dev_dynamic_class_balancing_lpft'
PATIENT_BUDGET_UNIT = 'patient'
DETECTION_TARGET_BUDGET_SAMPLING = (
    'selected_patients_all_ictal_dynamic_hard_far_one_to_two'
)
PREDICTION_TARGET_BUDGET_SAMPLING = (
    'selected_patients_all_preictal_dynamic_patient_balanced_interictal_one_to_one'
)
PREDICTION_SOURCE_TRAIN_SAMPLING = (
    'epoch_dynamic_keep_all_preictal_match_interictal_one_to_one'
)
PREDICTION_DATASET_ANCHORS = {
    'chbmit': 'preictal',
    'chbmit_historical': 'preictal',
    'chbmit_future': 'preictal',
    'siena': 'preictal',
    'tusz': 'preictal',
}


def prediction_anchor_label(dataset: str) -> int:
    key = str(dataset).strip().lower().replace('-', '').replace('_', '')
    return 0 if PREDICTION_DATASET_ANCHORS.get(key, 'preictal') == 'interictal' else 1


def prediction_anchor_policy(dataset: str) -> str:
    key = str(dataset).strip().lower().replace('-', '').replace('_', '')
    anchor = PREDICTION_DATASET_ANCHORS.get(key, 'preictal')
    matched = 'preictal' if anchor == 'interictal' else 'interictal'
    return f'epoch_dynamic_keep_all_{anchor}_match_{matched}_one_to_one'


def prediction_negative_to_positive_ratio(mission_type: str) -> float:
    del mission_type
    return 1.0


def prediction_budget_sampling_policy(negative_to_positive_ratio: float) -> str:
    if math.isclose(float(negative_to_positive_ratio), 1.0):
        return PREDICTION_TARGET_BUDGET_SAMPLING
    raise ValueError('Prediction sampling requires a one-to-one class ratio')


def prediction_budget_sampling_policy_for_dataset(dataset: str) -> str:
    anchor = 'preictal' if prediction_anchor_label(dataset) == 1 else 'interictal'
    matched = 'interictal' if anchor == 'preictal' else 'preictal'
    if anchor == 'preictal':
        return PREDICTION_TARGET_BUDGET_SAMPLING
    return f'selected_patients_all_{anchor}_dynamic_{matched}_one_to_one'


def source_rehearsal_sampling_policy(
    task: str,
    negative_to_positive_ratio: float,
) -> str:
    if task == 'prediction':
        if not math.isclose(float(negative_to_positive_ratio), 1.0):
            raise ValueError('Prediction rehearsal requires a one-to-one class ratio')
        return 'patient_class_balanced_one_to_one'
    return 'patient_class_balanced_one_to_two'


def round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


@dataclass(frozen=True)
class PatientSplitSelection:
    split: str
    total_patient_count: int
    selected_patient_count: int
    total_positive_clip_count: int
    total_negative_clip_count: int
    selected_positive_clip_count: int
    eligible_negative_clip_count: int
    selected_dev_negative_clip_count: int
    selected_patients: tuple[str, ...]
    selected_positive_clip_ids: tuple[str, ...]
    eligible_negative_clip_ids: tuple[str, ...]
    selected_train_negative_clip_ids: tuple[str, ...]
    selected_dev_negative_clip_ids: tuple[str, ...]

    @property
    def selected_clip_ids(self) -> tuple[str, ...]:
        if self.split == 'train':
            return tuple(sorted(
                self.selected_positive_clip_ids
                + (
                    self.selected_train_negative_clip_ids
                    if self.selected_train_negative_clip_ids
                    else self.eligible_negative_clip_ids
                )
            ))
        return tuple(sorted(
            self.selected_positive_clip_ids + self.selected_dev_negative_clip_ids
        ))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in (
            'selected_patients', 'selected_positive_clip_ids',
            'eligible_negative_clip_ids', 'selected_train_negative_clip_ids',
            'selected_dev_negative_clip_ids',
        ):
            result[key] = list(result[key])
        selected_negative_count = (
            len(self.selected_train_negative_clip_ids)
            if self.split == 'train' and self.selected_train_negative_clip_ids
            else len(self.eligible_negative_clip_ids)
            if self.split == 'train'
            else len(self.selected_dev_negative_clip_ids)
        )
        selected_total = len(self.selected_clip_ids)
        original_total = self.total_positive_clip_count + self.total_negative_clip_count
        result['selected_negative_clip_count'] = selected_negative_count
        result['selected_clip_count'] = selected_total
        result['achieved_patient_budget_fraction'] = float(
            self.selected_patient_count / self.total_patient_count
        )
        result['loader_pool_fraction'] = float(selected_total / original_total)
        result['original_positive_fraction'] = float(
            self.total_positive_clip_count / original_total
        )
        result['selected_positive_fraction'] = float(
            self.selected_positive_clip_count / selected_total
        )
        return result


@dataclass(frozen=True)
class PatientBudgetSelection:
    task: str
    dataset: str
    percentage: float
    budget_seed: int
    protocol: str
    negative_to_positive_ratio: float
    source_rehearsal_fraction: float
    train: PatientSplitSelection
    dev: PatientSplitSelection
    fingerprint: str

    def clip_ids(self, split: str) -> tuple[str, ...]:
        if split == 'train':
            return self.train.selected_clip_ids
        if split == 'dev':
            return self.dev.selected_clip_ids
        raise ValueError(f'Target label budget does not filter split: {split}')

    def to_dict(self) -> dict[str, object]:
        train_payload = self.train.to_dict()
        dev_payload = self.dev.to_dict()
        return {
            'budget_unit': PATIENT_BUDGET_UNIT,
            'task': self.task,
            'dataset': self.dataset,
            'percentage': self.percentage,
            'budget_seed': self.budget_seed,
            'protocol': self.protocol,
            'negative_to_positive_ratio': self.negative_to_positive_ratio,
            'budget_scope': 'target_train_and_dev_complete_patients',
            'dev_policy': 'same_percentage_nested_complete_patients',
            'patient_ordering': 'deterministic_patient_hash',
            'target_sampling': (
                DETECTION_TARGET_BUDGET_SAMPLING
                if self.task in {'detection', 'localization'}
                else prediction_budget_sampling_policy(
                    self.negative_to_positive_ratio
                )
            ),
            'source_rehearsal_sampling': source_rehearsal_sampling_policy(
                'detection' if self.task == 'localization' else self.task,
                self.negative_to_positive_ratio,
            ),
            'source_rehearsal_fraction': self.source_rehearsal_fraction,
            'train': train_payload,
            'dev': dev_payload,
            'fingerprint': self.fingerprint,
        }


def patient_rank_key(
    dataset: str,
    split: str,
    patient_id: str,
    budget_seed: int,
) -> str:
    return hashlib.sha256(
        f'{budget_seed}:{dataset}:{split}:patient:{patient_id}'.encode('utf-8')
    ).hexdigest()


def _patient_order(
    frame: pd.DataFrame,
    dataset: str,
    split: str,
    budget_seed: int,
) -> list[str]:
    patients = frame['patient_id'].astype(str).unique().tolist()
    return sorted(
        patients,
        key=lambda patient_id: patient_rank_key(
            dataset, split, patient_id, budget_seed,
        ),
    )


def _select_patient_split(
    frame: pd.DataFrame,
    task: str,
    dataset: str,
    split: str,
    percentage: float,
    budget_seed: int,
    negative_to_positive_ratio: float,
) -> tuple[PatientSplitSelection, pd.DataFrame]:
    del task, negative_to_positive_ratio
    split_frame = frame.loc[frame['split'].astype(str) == split].copy()
    labels = split_frame['label'].astype(int)
    positive = split_frame.loc[labels == 1].copy()
    negative = split_frame.loc[labels == 0].copy()
    if positive.empty or negative.empty:
        raise ValueError(f'Patient budget requires both classes in target {split}')
    ranked = _patient_order(
        split_frame, dataset, split, budget_seed,
    )
    selected_count = max(
        1,
        round_half_up(len(ranked) * float(percentage) / 100.0),
    )
    selected_patients = tuple(ranked[:selected_count])
    selected_patient_set = set(selected_patients)
    selected_frame = split_frame.loc[
        split_frame['patient_id'].astype(str).isin(selected_patient_set)
    ].copy()
    selected_positive = selected_frame.loc[
        selected_frame['label'].astype(int) == 1
    ].copy()
    selected_negative = selected_frame.loc[
        selected_frame['label'].astype(int) == 0
    ].copy()
    if selected_positive.empty or selected_negative.empty:
        raise ValueError(
            f'Patient budget selected a single-class target {split} subset'
        )
    selected_train_negative = selected_negative if split == 'train' else selected_negative.head(0)
    selected_dev_negative = selected_negative if split == 'dev' else selected_negative.head(0)
    selection = PatientSplitSelection(
        split=split,
        total_patient_count=len(ranked),
        selected_patient_count=selected_count,
        total_positive_clip_count=len(positive),
        total_negative_clip_count=len(negative),
        selected_positive_clip_count=len(selected_positive),
        eligible_negative_clip_count=len(selected_negative),
        selected_dev_negative_clip_count=(
            len(selected_dev_negative) if split == 'dev' else 0
        ),
        selected_patients=selected_patients,
        selected_positive_clip_ids=tuple(sorted(
            selected_positive['clip_id'].astype(str)
        )),
        eligible_negative_clip_ids=tuple(sorted(
            selected_negative['clip_id'].astype(str)
        )),
        selected_train_negative_clip_ids=tuple(sorted(
            selected_train_negative['clip_id'].astype(str)
        )),
        selected_dev_negative_clip_ids=(
            tuple(sorted(selected_dev_negative['clip_id'].astype(str)))
            if split == 'dev' else tuple()
        ),
    )
    patient_counts = split_frame.assign(
        patient_id=split_frame['patient_id'].astype(str),
        _label=split_frame['label'].astype(int),
    ).groupby('patient_id').agg(
        clip_count=('clip_id', 'size'),
        positive_clip_count=('_label', lambda values: int((values == 1).sum())),
        negative_clip_count=('_label', lambda values: int((values == 0).sum())),
    )
    audit = pd.DataFrame({
        'split': split,
        'patient_id': ranked,
        'rank': list(range(1, len(ranked) + 1)),
        'selected': [item in selected_patient_set for item in ranked],
        'budget_percent': float(percentage),
        'budget_seed': int(budget_seed),
    }).join(patient_counts, on='patient_id')
    return selection, audit


def select_budget_patients(
    manifest_path: str | Path,
    dataset: str,
    percentage: float,
    budget_seed: int,
    output_dir: str | Path | None = None,
    negative_to_positive_ratio: float | None = None,
    source_rehearsal_fraction: float = 0.25,
    task: str = 'detection',
) -> PatientBudgetSelection:
    manifest_path = Path(manifest_path)
    frame = pd.read_csv(
        manifest_path,
        dtype={'patient_id': str, 'clip_id': str, 'event_id': str},
    )
    required = {'patient_id', 'clip_id', 'split', 'label'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f'Manifest lacks patient budget columns {sorted(missing)}: '
            f'{manifest_path}'
        )
    if 'event_id' not in frame.columns:
        frame['event_id'] = ''
    duplicate_clips = frame['clip_id'].astype(str).duplicated(keep=False)
    if duplicate_clips.any():
        examples = sorted(
            frame.loc[duplicate_clips, 'clip_id'].astype(str).unique()
        )[:10]
        raise ValueError(f'Budgeting requires globally unique clip_id: {examples}')
    if percentage <= 0.0 or percentage > 100.0:
        raise ValueError('Patient budget percentage must be in (0, 100]')
    if task not in {'detection', 'prediction'}:
        raise ValueError(f'Unsupported budget task: {task}')
    if negative_to_positive_ratio is None:
        negative_to_positive_ratio = 2.0 if task == 'detection' else 1.0
    if negative_to_positive_ratio <= 0.0:
        raise ValueError('Negative-to-positive ratio must be positive')
    if not 0.0 < source_rehearsal_fraction < 1.0:
        raise ValueError('Source rehearsal fraction must be in (0, 1)')
    train, train_audit = _select_patient_split(
        frame, task, dataset, 'train', percentage, budget_seed,
        negative_to_positive_ratio,
    )
    dev, dev_audit = _select_patient_split(
        frame, task, dataset, 'dev', percentage, budget_seed,
        negative_to_positive_ratio,
    )
    fingerprint_payload = {
        'budget_unit': PATIENT_BUDGET_UNIT,
        'task': task,
        'dataset': dataset,
        'percentage': float(percentage),
        'budget_seed': int(budget_seed),
        'protocol': PATIENT_BUDGET_PROTOCOL,
        'negative_to_positive_ratio': float(negative_to_positive_ratio),
        'budget_scope': 'target_train_and_dev_complete_patients',
        'dev_policy': 'same_percentage_nested_complete_patients',
        'patient_ordering': 'deterministic_patient_hash',
        'target_sampling': (
            DETECTION_TARGET_BUDGET_SAMPLING
            if task == 'detection'
            else prediction_budget_sampling_policy_for_dataset(dataset)
        ),
        'source_rehearsal_sampling': source_rehearsal_sampling_policy(
            task, negative_to_positive_ratio,
        ),
        'source_rehearsal_fraction': float(source_rehearsal_fraction),
        'train_patients': list(train.selected_patients),
        'dev_patients': list(dev.selected_patients),
        'train_clip_ids': list(train.selected_clip_ids),
        'dev_clip_ids': list(dev.selected_clip_ids),
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload, sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
    ).hexdigest()
    selection = PatientBudgetSelection(
        task=task,
        dataset=dataset,
        percentage=float(percentage),
        budget_seed=int(budget_seed),
        protocol=PATIENT_BUDGET_PROTOCOL,
        negative_to_positive_ratio=float(negative_to_positive_ratio),
        source_rehearsal_fraction=float(source_rehearsal_fraction),
        train=train,
        dev=dev,
        fingerprint=fingerprint,
    )
    if output_dir is not None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        selected_by_split = {
            'train': set(train.selected_clip_ids),
            'dev': set(dev.selected_clip_ids),
        }
        clip_columns = [
            item for item in (
                'split', 'patient_id', 'clip_id', 'label', 'event_id',
                'source_relative_path', 'clip_start_seconds', 'clip_end_seconds',
            ) if item in frame.columns
        ]
        clip_audit = frame.loc[
            frame['split'].astype(str).isin({'train', 'dev'}),
            clip_columns,
        ].copy()
        clip_audit['selected_for_loader'] = [
            str(row.clip_id) in selected_by_split[str(row.split)]
            for row in clip_audit.itertuples(index=False)
        ]
        clip_audit['sampled_dynamically'] = (
            clip_audit['split'].astype(str).eq('train')
            & clip_audit['label'].astype(int).eq(0)
            & clip_audit['selected_for_loader']
        )
        clip_audit['budget_percent'] = float(percentage)
        clip_audit['budget_seed'] = int(budget_seed)
        clip_audit.to_csv(root / 'budget_clips.csv', index=False)
        pd.concat([train_audit, dev_audit], ignore_index=True).to_csv(
            root / 'budget_patients.csv', index=False
        )
        (root / 'budget_protocol.json').write_text(
            json.dumps(selection.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return selection


def select_temporal_budget_patients(
    manifest_path: str | Path,
    dataset: str,
    percentage: float,
    budget_seed: int,
    output_dir: str | Path | None = None,
    negative_to_positive_ratio: float | None = None,
    source_rehearsal_fraction: float = 0.25,
    task: str = 'detection',
) -> PatientBudgetSelection:
    return select_budget_patients(
        manifest_path=manifest_path,
        dataset=dataset,
        percentage=percentage,
        budget_seed=budget_seed,
        output_dir=output_dir,
        negative_to_positive_ratio=negative_to_positive_ratio,
        source_rehearsal_fraction=source_rehearsal_fraction,
        task=task,
    )


import numpy as np
import pandas as pd


SOURCE_REHEARSAL_FRACTION = 0.25


def _balanced_draws(
    frame: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    positive: bool,
) -> list[int]:
    labels = frame['label'].astype(int).to_numpy()
    eligible = frame.loc[labels == int(positive)].copy()
    if eligible.empty:
        raise ValueError('Balanced budget sampling requires both classes')
    eligible['_index'] = eligible.index.to_numpy(dtype=np.int64)
    patients = sorted(eligible['patient_id'].astype(str).unique())
    by_patient: dict[str, dict[str, np.ndarray] | np.ndarray] = {}
    for patient_id in patients:
        patient = eligible[eligible['patient_id'].astype(str) == patient_id]
        if positive and 'event_id' in patient.columns and not patient['event_id'].fillna('').astype(str).eq('').all():
            event_ids = patient['event_id'].fillna('').astype(str)
            if event_ids.eq('').any():
                raise ValueError(
                    f'Positive budget clips require event_id: {patient_id}'
                )
            by_patient[patient_id] = {
                event_id: group['_index'].to_numpy(dtype=np.int64)
                for event_id, group in patient.assign(_event=event_ids).groupby(
                    '_event', sort=True
                )
            }
        else:
            by_patient[patient_id] = patient['_index'].to_numpy(dtype=np.int64)
    selected: list[int] = []
    for draw in range(count):
        patient_id = patients[draw % len(patients)]
        pool = by_patient[patient_id]
        if positive and isinstance(pool, dict):
            events = sorted(pool)
            event_id = events[(draw // len(patients)) % len(events)]
            candidates = pool[event_id]
        else:
            candidates = pool
        selected.append(int(candidates[rng.integers(0, len(candidates))]))
    return selected


def _patient_class_draws(
    frame: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    positive: bool,
) -> list[int]:
    labels = frame['label'].astype(int).to_numpy()
    eligible = frame.loc[labels == int(positive)].copy()
    if eligible.empty:
        raise ValueError('Patient-class sampling requires both classes')
    eligible['_index'] = eligible.index.to_numpy(dtype=np.int64)
    patients = sorted(eligible['patient_id'].astype(str).unique())
    by_patient = {
        patient_id: eligible.loc[
            eligible['patient_id'].astype(str) == patient_id,
            '_index',
        ].to_numpy(dtype=np.int64)
        for patient_id in patients
    }
    selected: list[int] = []
    for draw in range(count):
        patient_id = patients[draw % len(patients)]
        candidates = by_patient[patient_id]
        selected.append(int(candidates[rng.integers(0, len(candidates))]))
    return selected


def _detection_negative_pools(
    frame: pd.DataFrame,
    context_seconds: float = 300.0,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    negative = frame.loc[frame['label'].astype(int) == 0].copy()
    required = {
        'patient_id', 'source_relative_path',
        'clip_start_seconds', 'clip_end_seconds',
    }
    if not required.issubset(frame.columns):
        ordered = negative.sort_index(kind='mergesort')
        hard = ordered.iloc[::2].copy()
        far = ordered.iloc[1::2].copy()
        return hard, far, 'deterministic_fallback_without_timing_columns'
    positive = frame.loc[frame['label'].astype(int) == 1].copy()
    spans = positive.groupby(
        ['patient_id', 'source_relative_path', 'event_id'], sort=False,
    ).agg(
        event_start=('clip_start_seconds', 'min'),
        event_end=('clip_end_seconds', 'max'),
    )
    span_map: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for key, span in spans.iterrows():
        patient_id, source_path, _ = key
        span_map.setdefault(
            (str(patient_id), str(source_path)), []
        ).append((float(span.event_start), float(span.event_end)))
    hard_mask = []
    for row in negative.itertuples(index=False):
        key = (str(row.patient_id), str(row.source_relative_path))
        if key not in span_map:
            hard_mask.append(False)
            continue
        distance = min(
            max(
                event_start - float(row.clip_end_seconds),
                float(row.clip_start_seconds) - event_end,
                0.0,
            )
            for event_start, event_end in span_map[key]
        )
        hard_mask.append(distance <= context_seconds)
    hard_mask = np.asarray(hard_mask, dtype=bool)
    return (
        negative.loc[hard_mask].copy(),
        negative.loc[~hard_mask].copy(),
        f'same_recording_within_{context_seconds:g}s_of_selected_event',
    )


def _draw_detection_negatives(
    frame: pd.DataFrame,
    positive_count: int,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, int | str]]:
    hard, far, context_policy = _detection_negative_pools(frame)
    hard_count = positive_count if not hard.empty else 0
    far_count = positive_count if not far.empty else 0
    if hard_count == 0 and far_count == 0:
        raise ValueError('Detection patient budget has no eligible negative clips')
    if hard_count == 0:
        far_count = 2 * positive_count
    elif far_count == 0:
        hard_count = 2 * positive_count
    selected = []
    if hard_count:
        selected.extend(_patient_class_draws(hard, hard_count, rng, positive=False))
    if far_count:
        selected.extend(_patient_class_draws(far, far_count, rng, positive=False))
    return selected, {
        'negative_context_policy': context_policy,
        'hard_negative_pool': int(len(hard)),
        'far_negative_pool': int(len(far)),
        'hard_negative_draws': int(hard_count),
        'far_negative_draws': int(far_count),
    }


def patient_budget_rehearsal_indices(
    target_frame: pd.DataFrame,
    source_frame: pd.DataFrame,
    seed: int,
    epoch: int,
    rehearsal_fraction: float = SOURCE_REHEARSAL_FRACTION,
    target_dataset: str | None = None,
    source_dataset: str | None = None,
    task: str | None = None,
    prediction_negative_to_positive_ratio: float = 1.0,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    if not 0.0 < rehearsal_fraction < 1.0:
        raise ValueError('Source rehearsal fraction must be in (0, 1)')
    target_positive_count = int((target_frame['label'].astype(int) == 1).sum())
    if target_positive_count <= 0:
        raise ValueError('Target label budget contains no positive clips')
    rng = np.random.default_rng(int(seed) + int(epoch))
    context_report: dict[str, int | str] = {}
    if str(task) == 'detection':
        target_positive = np.flatnonzero(
            target_frame['label'].astype(int).to_numpy() == 1
        ).tolist()
        target_negative, context_report = _draw_detection_negatives(
            target_frame, target_positive_count, rng,
        )
        target_policy = DETECTION_TARGET_BUDGET_SAMPLING
    else:
        negative_ratio = float(prediction_negative_to_positive_ratio)
        if negative_ratio <= 0.0:
            raise ValueError(
                'Prediction negative-to-positive ratio must be positive'
            )
        target_anchor_label = prediction_anchor_label(str(target_dataset))
        target_labels = target_frame['label'].astype(int).to_numpy()
        target_anchor = np.flatnonzero(target_labels == target_anchor_label).tolist()
        target_anchor_count = len(target_anchor)
        matched_count = int(round(target_anchor_count * negative_ratio))
        if target_anchor_label == 0:
            target_negative = target_anchor
            target_positive = _balanced_draws(
                target_frame, matched_count, rng, positive=True,
            )
        else:
            target_positive = target_anchor
            target_negative = _patient_class_draws(
                target_frame, matched_count, rng, positive=False,
            )
        target_policy = prediction_budget_sampling_policy_for_dataset(
            str(target_dataset)
        )
    target_count = len(target_positive) + len(target_negative)
    rehearsal_negative_ratio = (
        float(prediction_negative_to_positive_ratio)
        if str(task) == 'prediction'
        else 2.0
    )
    source_anchor_label = prediction_anchor_label(str(source_dataset or target_dataset))
    source_anchor_count = max(
        1, int(round(target_count * rehearsal_fraction / (1.0 + rehearsal_negative_ratio)))
    )
    source_matched_count = max(
        1, int(round(rehearsal_negative_ratio * source_anchor_count))
    )
    if str(task) == 'prediction' and source_anchor_label == 0:
        source_negative = _patient_class_draws(
            source_frame, source_anchor_count, rng, positive=False
        )
        source_positive = _balanced_draws(
            source_frame, source_matched_count, rng, positive=True
        )
    else:
        source_positive = _patient_class_draws(
            source_frame, source_anchor_count, rng, positive=True
        )
        source_negative = _patient_class_draws(
            source_frame, source_matched_count, rng, positive=False
        )
    source_offset = len(target_frame)
    indices = np.asarray(
        target_positive
        + target_negative
        + [source_offset + value for value in source_positive + source_negative],
        dtype=np.int64,
    )
    rng.shuffle(indices)
    report: dict[str, int | float | str] = {
        'policy': 'dataset_conditional_target_sampling_with_source_rehearsal',
        'target_policy': target_policy,
        'source_policy': (
            source_rehearsal_sampling_policy(
                str(task), rehearsal_negative_ratio,
            )
            + '_rehearsal'
        ),
        'epoch': int(epoch),
        'epoch_seed': int(seed) + int(epoch),
        'target_positive': len(target_positive),
        'target_negative': len(target_negative),
        'source_positive': len(source_positive),
        'source_negative': len(source_negative),
        'source_rehearsal_fraction_of_target': float(
            (len(source_positive) + len(source_negative)) / target_count
        ),
        'total': len(indices),
        **context_report,
    }
    return indices, report




import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


DETECTION_UNDERSAMPLE_DATASETS = frozenset({
    'tusz',
    'chbmit',
    'chbmit_historical',
    'chbmit_future',
    'siena',
})
DYNAMIC_NEGATIVE_TO_POSITIVE_RATIO = 2.0


def training_sampling_policy(
    task: str,
    dataset: str,
    single_fit: bool = False,
    mission_type: str = 'dataset_transfer',
) -> str:
    if task == 'localization':
        return 'localization-eval-only'
    if task == 'prediction':
        policy = prediction_anchor_policy(dataset)
        if single_fit:
            return f'single-fit {policy.removeprefix("epoch_dynamic_")}'
        return policy
    if task != 'detection' or dataset not in DETECTION_UNDERSAMPLE_DATASETS:
        return 'natural-distribution'
    if single_fit:
        return 'single-fit epoch-0 non-ictal-cap-2x-ictal'
    return 'epoch-dynamic non-ictal-cap-2x-ictal'


def mission_sampling_summary(spec, single_fit: bool = False) -> str:
    if spec.task == 'localization':
        return 'localization-eval-only'
    source = training_sampling_policy(
        spec.task,
        spec.source_dataset,
        single_fit=single_fit,
        mission_type=spec.mission_type,
    )
    if spec.budget_percent == 0.0:
        target = 'not-applicable'
    elif spec.task == 'detection':
        target = (
            'train nested complete-patient budget;all selected-patient ictal clips;'
            'dynamic hard-far negative one-to-two;fixed full target dev;'
            f'source-rehearsal={spec.source_rehearsal_fraction:g}x-target;'
            'source-rehearsal-policy=balanced-1:2;'
            'target-dev-excluded-from-budget'
        )
    else:
        prediction_ratio = prediction_negative_to_positive_ratio(
            spec.mission_type
        )
        prediction_anchor = (
            'preictal' if prediction_anchor_label(spec.target_dataset) == 1
            else 'interictal'
        )
        prediction_matched = (
            'interictal' if prediction_anchor == 'preictal' else 'preictal'
        )
        prediction_policy = (
            f'complete selected patients;all {prediction_anchor};'
            f'dynamic patient-balanced {prediction_matched} 1:1;'
            if math.isclose(prediction_ratio, 1.0)
            else 'patient-event-clip balanced 1:2;'
        )
        target = (
            'train nested complete-patient budget;'
            f'{prediction_policy}'
            'fixed full target dev;'
            f'source-rehearsal={spec.source_rehearsal_fraction:g}x-target;'
            f'source-rehearsal-policy=balanced-1:{prediction_ratio:g};'
            'target-dev-excluded-from-budget'
        )
    return f'source={source};target={target}'


@dataclass(frozen=True)
class DetectionSamplingReport:
    dataset: str
    split: str
    base_seed: int
    epoch: int
    epoch_seed: int
    target_negative_to_positive_ratio: float | None
    policy: str
    original_positive: int
    original_negative: int
    retained_positive: int
    retained_negative: int
    retained_total: int
    achieved_negative_pool_fraction: float
    achieved_negative_to_positive_ratio: float
    achieved_positive_fraction: float
    selected_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionSamplingReport:
    dataset: str
    split: str
    base_seed: int
    epoch: int
    epoch_seed: int
    policy: str
    original_positive: int
    original_negative: int
    retained_positive: int
    retained_negative: int
    retained_total: int
    achieved_negative_to_positive_ratio: float
    selected_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _rank(seed: int, dataset: str, clip_id: str) -> str:
    return hashlib.sha256(f'{seed}:{dataset}:{clip_id}'.encode('utf-8')).hexdigest()


def undersample_detection_training_frame(
    frame: pd.DataFrame,
    dataset: str,
    split: str,
    seed: int,
    epoch: int = 0,
    audit_root: str | Path | None = None,
) -> tuple[pd.DataFrame, DetectionSamplingReport | None]:
    required = {'clip_id', 'patient_id', 'label'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Detection sampling frame lacks columns: {sorted(missing)}')
    if split != 'train' or dataset not in DETECTION_UNDERSAMPLE_DATASETS:
        return frame.reset_index(drop=True), None
    labels = frame['label'].astype(int)
    invalid = sorted(set(labels) - {0, 1})
    if invalid:
        raise ValueError(f'Detection sampling requires binary labels: {invalid}')
    positive = frame.loc[labels == 1].copy()
    negative = frame.loc[labels == 0].copy()
    if positive.empty or negative.empty:
        raise ValueError(
            f'Training-only detection undersampling requires both classes for {dataset}'
        )
    if epoch < 0:
        raise ValueError('Detection sampling epoch must be non-negative')
    epoch_seed = int(seed) + int(epoch)
    requested_ratio = DYNAMIC_NEGATIVE_TO_POSITIVE_RATIO
    keep = min(
        len(negative),
        int(len(positive) * requested_ratio + 0.5),
    )
    selected_negative = negative.assign(
        _sampling_rank=[
            _rank(epoch_seed, dataset, str(clip_id))
            for clip_id in negative['clip_id']
        ]
    ).sort_values(['_sampling_rank', 'clip_id'], kind='mergesort').head(keep)
    selected_negative = selected_negative.drop(columns=['_sampling_rank'])
    policy = 'epoch_dynamic_keep_all_positive_global_hash_one_to_two_negative'
    retained = pd.concat([positive, selected_negative], ignore_index=False)
    retained = retained.sort_index(kind='mergesort').reset_index(drop=True)
    selected_ids = sorted(retained['clip_id'].astype(str))
    fingerprint = hashlib.sha256('\n'.join(selected_ids).encode('utf-8')).hexdigest()
    report = DetectionSamplingReport(
        dataset=dataset,
        split=split,
        base_seed=int(seed),
        epoch=int(epoch),
        epoch_seed=epoch_seed,
        target_negative_to_positive_ratio=requested_ratio,
        policy=policy,
        original_positive=int(len(positive)),
        original_negative=int(len(negative)),
        retained_positive=int(len(positive)),
        retained_negative=int(len(selected_negative)),
        retained_total=int(len(retained)),
        achieved_negative_pool_fraction=float(len(selected_negative) / len(negative)),
        achieved_negative_to_positive_ratio=float(len(selected_negative) / len(positive)),
        achieved_positive_fraction=float(len(positive) / len(retained)),
        selected_fingerprint=fingerprint,
    )
    if audit_root is not None:
        audit_mode = os.environ.get('SAMPLING_AUDIT_MODE', 'compact').strip().lower()
        if audit_mode not in {'compact', 'full', 'none'}:
            raise ValueError(f'Unsupported SAMPLING_AUDIT_MODE: {audit_mode}')
        root = Path(audit_root)
        root.mkdir(parents=True, exist_ok=True)
        if audit_mode == 'full':
            epoch_root = root / f'epoch_{epoch:04d}'
            epoch_root.mkdir(parents=True, exist_ok=True)
            audit = frame[['clip_id', 'patient_id', 'label']].copy()
            audit['selected'] = audit['clip_id'].astype(str).isin(set(selected_ids))
            audit['base_sampling_seed'] = int(seed)
            audit['epoch'] = int(epoch)
            audit['epoch_sampling_seed'] = epoch_seed
            audit['target_negative_to_positive_ratio'] = requested_ratio
            audit.to_csv(epoch_root / 'clip_selection.csv', index=False)
            report_path = epoch_root / 'sampling_protocol.json'
        else:
            report_path = root / f'epoch_{epoch:04d}.json'
        if audit_mode != 'none':
            report_path.write_text(
                json.dumps({
                    **report.to_dict(),
                    'audit_mode': audit_mode,
                    'replay_contract': 'sha256_epoch_seed_dataset_clip_id_rank',
                }, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )
    return retained, report


def balance_prediction_training_frame(
    frame: pd.DataFrame,
    dataset: str,
    split: str,
    seed: int,
    epoch: int = 0,
    audit_root: str | Path | None = None,
) -> tuple[pd.DataFrame, PredictionSamplingReport]:
    required = {'clip_id', 'patient_id', 'label'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f'Prediction sampling frame lacks columns {sorted(missing)}'
        )
    if split != 'train':
        raise ValueError('Prediction balancing is training-only')
    labels = frame['label'].astype(int)
    invalid = sorted(set(labels) - {0, 1})
    if invalid:
        raise ValueError(f'Prediction sampling requires binary labels {invalid}')
    positive = frame.loc[labels == 1].copy()
    negative = frame.loc[labels == 0].copy()
    if positive.empty or negative.empty:
        raise ValueError(
            f'Prediction balancing requires both classes for {dataset}'
        )
    epoch_seed = int(seed) + int(epoch)
    rng = np.random.default_rng(epoch_seed)
    indexed = frame.reset_index(drop=True)
    anchor_label = prediction_anchor_label(dataset)
    anchor = positive if anchor_label == 1 else negative
    matched_label = 1 - anchor_label
    if matched_label == 1:
        matched_indices = _balanced_draws(
            indexed, len(anchor), rng, positive=True,
        )
    else:
        matched_indices = _patient_class_draws(
            indexed, len(anchor), rng, positive=False,
        )
    matched = indexed.iloc[matched_indices].copy()
    selected_positive = anchor.copy() if anchor_label == 1 else matched
    selected_negative = anchor.copy() if anchor_label == 0 else matched
    retained = pd.concat(
        [selected_positive.reset_index(drop=True), selected_negative.reset_index(drop=True)],
        ignore_index=True,
    )
    retained = retained.iloc[
        rng.permutation(len(retained))
    ].reset_index(drop=True)
    selected_ids = sorted(retained['clip_id'].astype(str).tolist())
    fingerprint = hashlib.sha256(
        '\n'.join(selected_ids).encode('utf-8')
    ).hexdigest()
    report = PredictionSamplingReport(
        dataset=str(dataset),
        split=str(split),
        base_seed=int(seed),
        epoch=int(epoch),
        epoch_seed=epoch_seed,
        policy=prediction_anchor_policy(dataset),
        original_positive=int(len(positive)),
        original_negative=int(len(negative)),
        retained_positive=int(len(selected_positive)),
        retained_negative=int(len(selected_negative)),
        retained_total=int(len(retained)),
        achieved_negative_to_positive_ratio=float(
            len(selected_negative) / len(selected_positive)
        ),
        selected_fingerprint=fingerprint,
    )
    if audit_root is not None and os.environ.get(
        'SAMPLING_AUDIT_MODE', 'compact'
    ) != 'none':
        root = Path(audit_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / f'prediction_epoch_{epoch:04d}.json').write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return retained, report




import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_ROOT = Path(
    os.environ.get(
        'BENCHMARK_ROOT',
        str(Path(__file__).resolve().parents[2]),
    )
)
DEFAULT_EEG_ROOT = Path(
    os.environ.get('EEG_ROOT', str(DEFAULT_BENCHMARK_ROOT))
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get(
        'DATA_ROOT',
        str(DEFAULT_BENCHMARK_ROOT / 'data'),
    )
)
DEFAULT_PREPROCESS_ROOT = Path(
    os.environ.get(
        'PREPROCESS_ROOT',
        str(DEFAULT_DATA_ROOT / 'PreprocessResults_eeg_cross_dataset'),
    )
)
DEFAULT_CROSS_MODAL_PREPROCESS_ROOT = Path(
    os.environ.get(
        'CROSS_MODAL_PREPROCESS_ROOT',
        str(DEFAULT_DATA_ROOT / 'PreprocessResults_eeg_ieeg_cross_modal'),
    )
)
DEFAULT_CROSS_TIME_PREPROCESS_ROOT = Path(
    os.environ.get(
        'CROSS_TIME_PREPROCESS_ROOT',
        str(DEFAULT_DATA_ROOT / 'PreprocessResult_cross_time'),
    )
)
DEFAULT_RESULT_ROOT = Path(
    os.environ.get('RESULT_ROOT', str(DEFAULT_DATA_ROOT / 'Results'))
)
DEFAULT_CROSS_MODAL_RESULT_ROOT = Path(
    os.environ.get(
        'CROSS_MODAL_RESULT_ROOT',
        str(DEFAULT_DATA_ROOT / 'results' / 'cross_modal'),
    )
)
DEFAULT_LOCALIZATION_REFERENCE_RESULT_ROOT = Path(
    os.environ.get(
        'LOCALIZATION_REFERENCE_RESULT_ROOT',
        os.environ.get(
            'CROSS_MODAL_RESULT_ROOT',
            str(DEFAULT_CROSS_MODAL_RESULT_ROOT),
        ),
    )
)
DEFAULT_LOCALIZATION_RESULT_ROOT = Path(
    os.environ.get(
        'LOCALIZATION_RESULT_ROOT',
        str(DEFAULT_DATA_ROOT / 'results' / 'localization'),
    )
)
DEFAULT_CROSS_TIME_RESULT_ROOT = Path(
    os.environ.get(
        'CROSS_TIME_RESULT_ROOT',
        str(DEFAULT_DATA_ROOT / 'Results_cross_time'),
    )
)


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    'BENDR': {
        'script': 'models/baselines/BENDR/benchmark_adapter.py',
        'pretrained': 'models/baselines/BENDR/model.safetensors',
        'budget_ft': True,
        'lr': 1e-5, 'target_lr': 1e-5, 'target_backbone_lr': 1e-6,
        'weight_decay': 1e-2, 'max_grad_norm': 1.0,
    },
    'BIOT': {
        'script': 'models/baselines/BIOT/benchmark_adapter.py',
        'pretrained': 'models/baselines/BIOT/pretrained-models/EEG-six-datasets-18-channels.ckpt',
        'budget_ft': True,
        'lr': 1e-3, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 1e-5,
    },
    'CBraMod': {
        'script': 'models/baselines/CBraMod/benchmark_adapter.py',
        'pretrained': 'models/baselines/CBraMod/pretrained_weights/pretrained_weights.pth',
        'budget_ft': True,
        'lr': 1e-4, 'target_lr': 1e-5, 'target_backbone_lr': 1e-6,
        'weight_decay': 5e-2,
    },
    'CST': {
        'script': 'models/baselines/CST/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 1e-3, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 0.0, 'max_grad_norm': 1.0,
    },
    'EEGPT': {
        'script': 'models/baselines/EEGPT/benchmark_adapter.py',
        'pretrained': 'models/baselines/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt',
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'LaBraM': {
        'script': 'models/baselines/LaBraM/benchmark_adapter.py',
        'pretrained': 'models/baselines/LaBraM/checkpoints/labram-base.pth',
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'STEEGFormer': {
        'script': 'models/baselines/STEEGFormer/benchmark_adapter.py',
        'pretrained': 'models/baselines/STEEGFormer/checkpoint-300.pth',
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'RIVER': {
        'script': 'models/RIVER/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 3e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'EEGNet': {
        'script': 'models/baselines/EEGNet/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 1e-3, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
    },
    'EvoBrain': {
        'script': 'models/baselines/EvoBrain/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 3e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-4, 'max_grad_norm': 5.0,
    },
    'SVM': {
        'script': 'models/baselines/SVM/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
    },
    'ScatterFormer': {
        'script': 'models/baselines/ScatterFormer/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'CMMN': {
        'script': 'models/baselines/CMMN/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
    },
    'TSMNet': {
        'script': 'models/baselines/TSMNet/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'BF-EML': {
        'script': 'models/baselines/BF-EML/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
        'lr': 5e-4, 'target_lr': 1e-4, 'target_backbone_lr': 1e-5,
        'weight_decay': 5e-2, 'max_grad_norm': 1.0,
    },
    'RandomForest': {
        'script': 'models/baselines/RandomForest/benchmark_adapter.py',
        'pretrained': None,
        'budget_ft': True,
    },
}


TRADITIONAL_ML_MODELS = frozenset({'SVM', 'CMMN', 'RandomForest'})


DATASET_DISPLAY_NAMES = {
    'tusz': 'TUSZ',
    'siena': 'Siena',
    'chbmit': 'CHB-MIT',
    'chbmit_historical': 'CHBMIT-Historical',
    'chbmit_future': 'CHBMIT-Future',
    'epilepsy_ieeg': 'Epilepsy-iEEG',
    'hup_ieeg': 'HUP-iEEG',
    'thalamocortical_ieeg': 'Thalamocortical-iEEG',
}
SCALP_EEG_DATASETS = frozenset({
    'tusz', 'siena', 'chbmit', 'chbmit_historical', 'chbmit_future',
})
IEEG_DATASETS = frozenset({'epilepsy_ieeg', 'hup_ieeg', 'thalamocortical_ieeg'})
MISSION_TYPES = frozenset({
    'dataset_transfer', 'eeg_ieeg_transfer', 'eeg_ieeg_localization',
    'cross_time', 'ieeg_indomain',
})
BUDGET_FINETUNE_STRATEGY = 'linear_probe_then_full_model_discriminative_lr'
BUDGET_REFIT_STRATEGY = 'budget_refit_with_source_rehearsal'
NEW_BASELINE_MODELS = frozenset({
    'ScatterFormer', 'CMMN', 'TSMNet', 'BF-EML', 'RandomForest',
})
DEFAULT_NEW_BASELINE_RESULT_ROOT = Path(
    os.environ.get(
        'NEW_BASELINE_RESULT_ROOT',
        str(DEFAULT_DATA_ROOT / 'results' / 'baselines'),
    )
)
NEW_BASELINE_TRACK_FOLDERS = {
    'dataset_transfer': 'cross_dataset',
    'eeg_ieeg_transfer': 'cross_modality',
    'cross_time': 'cross_time',
}


def budget_finetune_strategy(model: str) -> str:
    return (
        BUDGET_REFIT_STRATEGY
        if model in TRADITIONAL_ML_MODELS
        else BUDGET_FINETUNE_STRATEGY
    )


def result_root_for_model(args: argparse.Namespace, model: str, mission_type: str) -> Path:
    if model in NEW_BASELINE_MODELS and mission_type in NEW_BASELINE_TRACK_FOLDERS:
        return args.new_baseline_result_root
    return args.result_root
def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f'{value:g}'


@dataclass(frozen=True)
class MissionSpec:
    model: str
    task: str
    source_dataset: str
    target_dataset: str
    window_seconds: float
    mission_type: str = 'dataset_transfer'
    budget_percent: float = 0.0
    budget_seed: int = 1
    use_pretrained: bool = True
    epochs: int = 100
    patience: int = 20
    min_delta: float = 0.001
    seed: int = 1
    batch_size: int = 128
    num_workers: int = 8
    stats_num_workers: int = 8
    bootstrap_resamples: int = 2000
    gpu: int = 0
    physical_gpu: int = 0
    deterministic: bool = True
    undersample_seed: int = 1
    source_rehearsal_fraction: float = 0.25
    generate_interpretability: bool = False
    interpretability_max_clips: int = 2000
    mri_template_path: Path | None = None
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT
    preprocess_root: Path = DEFAULT_PREPROCESS_ROOT
    result_root: Path = DEFAULT_RESULT_ROOT
    cross_modal_result_root: Path = DEFAULT_LOCALIZATION_REFERENCE_RESULT_ROOT
    localization_reference_window_seconds: float | None = None
    new_baseline_result_root: Path = DEFAULT_NEW_BASELINE_RESULT_ROOT
    def __post_init__(self) -> None:
        if self.model not in MODEL_REGISTRY:
            raise ValueError(f'Unsupported model: {self.model}')
        if self.task not in {'detection', 'prediction', 'localization'}:
            raise ValueError(f'Unsupported task: {self.task}')
        if self.task == 'localization' and self.mission_type != 'eeg_ieeg_localization':
            raise ValueError('Localization task is only valid for EEG iEEG localization')
        if (
            self.mission_type == 'eeg_ieeg_localization'
            and self.model in TRADITIONAL_ML_MODELS
        ):
            raise ValueError(
                'EEG iEEG localization requires native contact evidence; '
                f'{self.model} is excluded by protocol'
            )
        if self.source_dataset not in DATASET_DISPLAY_NAMES:
            raise ValueError(f'Unsupported source dataset: {self.source_dataset}')
        if self.target_dataset not in DATASET_DISPLAY_NAMES:
            raise ValueError(f'Unsupported target dataset: {self.target_dataset}')
        if self.mission_type not in MISSION_TYPES:
            raise ValueError(f'Unsupported mission type: {self.mission_type}')
        self._validate_dataset_pair()
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0.0:
            raise ValueError('window_seconds must be positive')
        if not math.isfinite(self.budget_percent) or not 0.0 <= self.budget_percent <= 100.0:
            raise ValueError('budget_percent must be between 0 and 100')
        if self.epochs <= 0:
            raise ValueError('epochs must be positive')
        if self.patience <= 0:
            raise ValueError('patience must be positive')
        if self.num_workers < 0:
            raise ValueError('num_workers must be non-negative')
        if self.stats_num_workers <= 0:
            raise ValueError('stats_num_workers must be positive')
        if self.bootstrap_resamples <= 0:
            raise ValueError('bootstrap_resamples must be positive')
        if self.gpu < 0 or self.physical_gpu < 0:
            raise ValueError('GPU indices must be non-negative')
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError('min_delta must be finite and non-negative')
        if not math.isfinite(self.source_rehearsal_fraction) or not 0.0 < self.source_rehearsal_fraction < 1.0:
            raise ValueError('source_rehearsal_fraction must be in (0, 1)')
        if self.budget_percent > 0.0 and not MODEL_REGISTRY[self.model]['budget_ft']:
            raise ValueError(f'{self.model} does not support BudgetFT')
        if self.generate_interpretability and self.interpretability_max_clips <= 1:
            raise ValueError('interpretability_max_clips must exceed one')
        available = MODEL_REGISTRY[self.model]['pretrained'] is not None
        if self.use_pretrained and not available:
            raise ValueError(f'{self.model} has no official pretrained weight')

    def _validate_dataset_pair(self) -> None:
        if self.mission_type == 'ieeg_indomain':
            if self.source_dataset != self.target_dataset:
                raise ValueError('iEEG in-domain requires identical source and target datasets')
            if self.source_dataset not in IEEG_DATASETS:
                raise ValueError('iEEG in-domain requires an iEEG dataset')
            if self.budget_percent != 0.0:
                raise ValueError('iEEG in-domain currently supports zero budget only')
            if self.task == 'detection' and self.source_dataset != 'epilepsy_ieeg':
                raise ValueError('iEEG in-domain detection currently uses Epilepsy-iEEG')
            if self.task == 'prediction' and self.source_dataset != 'thalamocortical_ieeg':
                raise ValueError('iEEG in-domain prediction currently uses Thalamocortical-iEEG')
            if self.task not in {'detection', 'prediction'}:
                raise ValueError('iEEG in-domain supports detection and prediction')
            return
        if self.source_dataset == self.target_dataset:
            if self.mission_type != 'dataset_transfer':
                raise ValueError('In-domain evaluation is only valid for dataset transfer')
            if self.budget_percent != 0.0:
                raise ValueError('In-domain evaluation only supports zero budget')
            if self.source_dataset not in SCALP_EEG_DATASETS:
                raise ValueError('In-domain evaluation requires a scalp EEG dataset')
            return
        if self.mission_type == 'cross_time':
            if (self.source_dataset, self.target_dataset) != (
                'chbmit_historical', 'chbmit_future',
            ):
                raise ValueError(
                    'Cross-time transfer requires chbmit_historical:chbmit_future'
                )
            if self.task not in {'detection', 'prediction'}:
                raise ValueError('Cross-time transfer supports detection and prediction')
            return
        if self.mission_type == 'dataset_transfer':
            if self.source_dataset not in SCALP_EEG_DATASETS:
                raise ValueError('Dataset transfer source must be scalp EEG')
            if self.target_dataset not in SCALP_EEG_DATASETS:
                raise ValueError('Dataset transfer target must be scalp EEG')
            return
        if self.source_dataset not in SCALP_EEG_DATASETS:
            raise ValueError('EEG to iEEG transfer source must be scalp EEG')
        if self.mission_type == 'eeg_ieeg_localization':
            if self.task != 'localization':
                raise ValueError('EEG iEEG localization requires localization task')
            if self.target_dataset != 'epilepsy_ieeg':
                raise ValueError('EEG iEEG localization currently targets Epilepsy-iEEG only')
            return
        if self.task == 'detection':
            allowed = {'epilepsy_ieeg'}
        else:
            allowed = {'thalamocortical_ieeg'}
        if self.target_dataset not in allowed:
            raise ValueError(
                f'Unsupported {self.task} target for EEG to iEEG transfer: '
                f'{self.target_dataset}'
            )

    @property
    def mode(self) -> str:
        return 'SourceOnly' if self.budget_percent == 0.0 else 'FT'

    @property
    def is_in_domain(self) -> bool:
        return (
            self.mission_type == 'dataset_transfer'
            and self.source_dataset == self.target_dataset
        )

    @property
    def protocol(self) -> str:
        return 'zero_shot' if self.budget_percent == 0.0 else 'budget_ft'

    @property
    def direction(self) -> str:
        source = DATASET_DISPLAY_NAMES[self.source_dataset]
        target = DATASET_DISPLAY_NAMES[self.target_dataset]
        return f'{source}_to_{target}'

    @property
    def window_name(self) -> str:
        return f'{format_number(self.window_seconds)}s'

    @property
    def budget_name(self) -> str:
        value = format_number(self.budget_percent)
        suffix = f'{value}%_budget' if self.budget_percent > 0.0 else '0_budget'
        return f'{self.model}_{suffix}'

    @property
    def cache_root(self) -> Path:
        return self.preprocess_root / self.window_name

    def cache_dir(self, dataset: str) -> Path:
        task_folder = 'detection' if self.mission_type == 'eeg_ieeg_localization' else self.task
        return self.cache_root / task_folder / DATASET_DISPLAY_NAMES[dataset]

    @property
    def reference_output_dir(self) -> Path:
        return self.output_dir

    @property
    def source_reference_output_dir(self) -> Path:
        if self.mission_type != 'eeg_ieeg_localization':
            return self.zero_shot_output_dir
        reference_budget_name = self.budget_name
        reference_window_seconds = (
            self.window_seconds
            if self.localization_reference_window_seconds is None
            else self.localization_reference_window_seconds
        )
        reference_window_name = f'{format_number(reference_window_seconds)}s'
        if self.model in NEW_BASELINE_MODELS:
            return (
                self.new_baseline_result_root
                / 'cross_modality'
                / reference_window_name
                / 'detection'
                / self.direction
                / reference_budget_name
                / f'seed_{self.seed}'
            )
        return (
            self.cross_modal_result_root
            / 'eeg_ieeg_transfer'
            / reference_window_name
            / 'detection'
            / self.direction
            / reference_budget_name
            / f'seed_{self.seed}'
        )

    @property
    def source_reference_artifact_name(self) -> str:
        return {
            'BIOT': 'best.ckpt',
            'CBraMod': 'best.pth',
            'EvoBrain': 'best.pth.tar',
            'EEGNet': 'best.weights.h5',
            'SVM': 'best_model.joblib',
        }.get(self.model, 'best.pt')

    @property
    def source_reference_checkpoint_path(self) -> Path:
        if self.mission_type == 'eeg_ieeg_localization':
            stage = 'source' if self.budget_percent == 0.0 else 'target'
            return self.source_reference_output_dir / stage / self.source_reference_artifact_name
        return self.source_reference_output_dir / 'source' / self.source_reference_artifact_name

    @property
    def output_dir(self) -> Path:
        task_folder = 'localization' if self.mission_type == 'eeg_ieeg_localization' else self.task
        track_folder = (
            NEW_BASELINE_TRACK_FOLDERS[self.mission_type]
            if self.model in NEW_BASELINE_MODELS
            and self.mission_type in NEW_BASELINE_TRACK_FOLDERS
            else self.mission_type
        )
        return (
            self.result_root
            / track_folder
            / self.window_name
            / task_folder
            / self.direction
            / self.budget_name
            / f'seed_{self.seed}'
        )

    @property
    def zero_shot_output_dir(self) -> Path:
        task_folder = 'localization' if self.mission_type == 'eeg_ieeg_localization' else self.task
        track_folder = (
            NEW_BASELINE_TRACK_FOLDERS[self.mission_type]
            if self.model in NEW_BASELINE_MODELS
            and self.mission_type in NEW_BASELINE_TRACK_FOLDERS
            else self.mission_type
        )
        return (
            self.result_root
            / track_folder
            / self.window_name
            / task_folder
            / self.direction
            / f'{self.model}_0_budget'
            / f'seed_{self.seed}'
        )

    def pretrained_path(self) -> Path | None:
        relative = MODEL_REGISTRY[self.model]['pretrained']
        return None if relative is None else self.benchmark_root / relative

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            'benchmark_root', 'preprocess_root', 'result_root',
            'cross_modal_result_root', 'new_baseline_result_root',
            'mri_template_path',
        ):
            if result[key] is None:
                continue
            result[key] = str(result[key])
        result.update({
            'source_training_sampling': training_sampling_policy(
                self.task,
                self.source_dataset,
                single_fit=self.model in TRADITIONAL_ML_MODELS,
                mission_type=self.mission_type,
            ),
            'budget_target_sampling': (
                DETECTION_TARGET_BUDGET_SAMPLING
                if self.budget_percent > 0.0 and self.task == 'detection'
                else prediction_budget_sampling_policy_for_dataset(
                    self.target_dataset
                )
                if self.budget_percent > 0.0
                else 'none'
            ),
            'source_rehearsal_sampling': (
                source_rehearsal_sampling_policy(
                    self.task,
                    prediction_negative_to_positive_ratio(self.mission_type),
                )
                if self.budget_percent > 0.0 else 'none'
            ),
            'budget_unit': (
                PATIENT_BUDGET_UNIT if self.budget_percent > 0.0 else 'none'
            ),
            'budget_finetune_strategy': (
                budget_finetune_strategy(self.model)
                if self.budget_percent > 0.0 else 'none'
            ),
            'mode': self.mode,
            'is_in_domain': self.is_in_domain,
            'protocol': self.protocol,
            'direction': self.direction,
            'cache_root': str(self.cache_root),
            'output_dir': str(self.output_dir),
            'zero_shot_output_dir': str(self.zero_shot_output_dir),
            'source_reference_output_dir': str(self.source_reference_output_dir),
            'source_reference_checkpoint_path': str(self.source_reference_checkpoint_path),
            'pretrained_path': str(self.pretrained_path()) if self.pretrained_path() else None,
        })
        return result




import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np



SCALP_PREPROCESSING_PROTOCOL = 'scalp_common_16_bipolar_eahs'


def _same_value(left: object, right: object) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_in_domain_source_checkpoint(
    spec: MissionSpec,
    checkpoint_name: str,
    expected_arguments: dict[str, object],
    current_channel_contract: object,
) -> tuple[Path, dict[str, object]]:
    if not spec.is_in_domain or spec.budget_percent != 0.0:
        raise ValueError('Source checkpoint reuse requires a zero-budget in-domain mission')
    source_display = {
        'tusz': 'TUSZ', 'siena': 'Siena', 'chbmit': 'CHB-MIT',
    }[spec.source_dataset]
    search_root = (
        spec.result_root / 'dataset_transfer' / spec.window_name / spec.task
    )
    candidates = []
    stable_protocol = {
        'model': spec.model,
        'task': spec.task,
        'source_dataset': spec.source_dataset,
        'window_seconds': spec.window_seconds,
        'use_pretrained': spec.use_pretrained,
        'epochs': spec.epochs,
        'patience': spec.patience,
        'min_delta': spec.min_delta,
        'seed': spec.seed,
        'batch_size': spec.batch_size,
        'undersample_seed': spec.undersample_seed,
        'deterministic': spec.deterministic,
    }
    current_contract = {
        key: getattr(current_channel_contract, key)
        for key in (
            'policy', 'channel_keys', 'source_dataset',
            'source_channel_keys', 'source_native_channel_keys',
        )
    }
    canonical_reference_target = {
        'tusz': 'CHB-MIT',
        'chbmit': 'TUSZ',
        'siena': 'CHB-MIT',
    }[spec.source_dataset]
    direction_roots = [
        search_root / f'{source_display}_to_{canonical_reference_target}'
    ]
    audit_attempts = []
    for direction_root in direction_roots:
        run_root = direction_root / f'{spec.model}_0_budget' / f'seed_{spec.seed}'
        checkpoint = (
            run_root / checkpoint_name
            if checkpoint_name == 'best_model.joblib'
            else run_root / 'source' / checkpoint_name
        )
        required = [
            checkpoint,
            run_root / 'protocol.json',
            run_root / 'args.json',
            run_root / 'channel_union.json',
        ]
        missing = [
            str(path)
            for path in required
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            audit_attempts.append({
                'run_root': str(run_root),
                'status': 'missing_required_source_artifacts',
                'missing': missing,
            })
            continue
        protocol = json.loads((run_root / 'protocol.json').read_text(encoding='utf-8'))
        arguments = json.loads((run_root / 'args.json').read_text(encoding='utf-8'))
        channel_contract = json.loads((run_root / 'channel_union.json').read_text(encoding='utf-8'))
        mismatches = {}
        for key, expected in stable_protocol.items():
            actual = protocol.get(key)
            if not _same_value(actual, expected):
                mismatches[f'protocol.{key}'] = {'expected': expected, 'actual': actual}
        for key, expected in expected_arguments.items():
            actual = arguments.get(key)
            if not _same_value(actual, expected):
                mismatches[f'args.{key}'] = {'expected': expected, 'actual': actual}
        for key, expected in current_contract.items():
            actual = channel_contract.get(key)
            normalized_expected = list(expected) if isinstance(expected, tuple) else expected
            if actual != normalized_expected:
                mismatches[f'channel_union.{key}'] = {
                    'expected': normalized_expected,
                    'actual': actual,
                }
        if not mismatches:
            candidates.append({
                'path': checkpoint,
                'run_root': run_root,
                'direction': direction_root.name,
                'sha256': _sha256(checkpoint),
            })
            audit_attempts.append({
                'run_root': str(run_root),
                'status': 'compatible',
            })
        else:
            audit_attempts.append({
                'run_root': str(run_root),
                'status': 'contract_mismatch',
                'mismatches': mismatches,
            })
    if not candidates:
        raise RuntimeError(
            'No compatible source-stage checkpoint was found. Audit: '
            + json.dumps(audit_attempts, ensure_ascii=False, sort_keys=True)
        )
    hashes = {item['sha256'] for item in candidates}
    if len(hashes) != 1:
        raise RuntimeError(
            'Compatible source checkpoint candidates have different SHA256 values. '
            'Specify one source run only after auditing the discrepancy.'
        )
    selected = candidates[0]
    report = {
        'policy': 'reuse_source_stage_checkpoint_for_in_domain_evaluation',
        'path': str(selected['path']),
        'source_run': str(selected['run_root']),
        'source_direction': selected['direction'],
        'sha256': selected['sha256'],
        'compatible_candidate_count': len(candidates),
        'selection_rule': 'lexicographically_first_after_identical_sha256_verification',
        'canonical_reference_direction': f'{source_display}_to_{canonical_reference_target}',
        'threshold_policy': 'recompute_on_in_domain_source_dev',
        'test_policy': 'evaluate_once_on_in_domain_source_test',
        'source_stage_requirement': 'non_empty_best_checkpoint_selected_by_source_dev_auroc',
        'original_transfer_metrics_required': False,
        'strict_fields': [
            'model', 'task', 'source_dataset', 'window_seconds',
            'use_pretrained', 'seed', 'batch_size', 'undersample_seed',
            'deterministic', 'model_native_arguments', 'channel_keys',
            'source_channel_keys', 'source_native_channel_keys',
        ],
    }
    return selected['path'], report


def build_spec(args: argparse.Namespace, model: str) -> MissionSpec:
    return MissionSpec(
        model=model,
        task=args.task,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        window_seconds=args.window_seconds,
        mission_type=args.mission_type,
        budget_percent=args.budget_percent,
        budget_seed=args.budget_seed,
        use_pretrained=bool(args.use_pretrained),
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stats_num_workers=getattr(args, 'stats_num_workers', 8),
        bootstrap_resamples=getattr(args, 'bootstrap_resamples', 2000),
        gpu=args.gpu,
        physical_gpu=getattr(args, 'physical_gpu', args.gpu),
        deterministic=bool(args.deterministic),
        undersample_seed=args.undersample_seed,
        source_rehearsal_fraction=getattr(args, 'source_rehearsal_fraction', 0.25),
        generate_interpretability=bool(args.generate_interpretability),
        interpretability_max_clips=args.interpretability_max_clips,
        mri_template_path=args.mri_template_path,
        benchmark_root=args.benchmark_root,
        preprocess_root=args.preprocess_root,
        result_root=args.result_root,
        cross_modal_result_root=args.cross_modal_result_root,
        localization_reference_window_seconds=getattr(
            args, 'localization_reference_window_seconds', None
        ),
    )


def validate_cache(task_root: Path, spec: MissionSpec, dataset_key: str) -> dict[str, object]:
    required = [
        task_root / 'manifest.csv',
        task_root / 'dataset_contract.json',
        task_root / 'status.json',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f'Preprocessing cache is incomplete: {missing}')
    status = json.loads((task_root / 'status.json').read_text(encoding='utf-8'))
    if not status.get('ready_for_training', False):
        raise RuntimeError(f'Preprocessing cache is not ready for training: {task_root}')
    contract = json.loads((task_root / 'dataset_contract.json').read_text(encoding='utf-8'))
    expected_track = (
        'eeg_cross_dataset'
        if spec.mission_type in {'dataset_transfer', 'cross_time'}
        else 'eeg_ieeg_cross_modal'
    )
    expected_bandpass = (
        [0.5, 45.0]
        if spec.mission_type in {'dataset_transfer', 'cross_time'}
        else [0.5, 95.0]
    )
    actual_bandpass = [
        float(value) for value in contract.get('requested_bandpass_hz', [])
    ]
    line_frequencies = [
        float(value) for value in contract.get('requested_notch_freqs_hz', [])
    ]
    frozen_contract = {
        'protocol_track': contract.get('protocol_track') == expected_track,
        'requested_bandpass_hz': actual_bandpass == expected_bandpass,
        'target_sfreq': int(contract.get('target_sfreq', -1)) == 256,
        'line_frequency': (
            len(line_frequencies) == 1
            and line_frequencies[0] in {50.0, 60.0}
        ),
        'line_noise_policy': (
            contract.get('line_noise_policy') == 'dataset_native_spectrum_fit'
        ),
        'notch_method': contract.get('notch_method') == 'spectrum_fit',
        'bandpass_method': contract.get('bandpass_method') == 'fir',
        'bandpass_phase': contract.get('bandpass_phase') == 'zero',
        'resampling_method': contract.get('resampling_method') == 'fft',
        'normalization': (
            contract.get('normalization') == 'per_clip_per_channel_zscore'
        ),
    }
    failed_contract_fields = [
        field for field, valid in frozen_contract.items() if not valid
    ]
    if failed_contract_fields:
        raise ValueError(
            f'Cache does not match final {expected_track} preprocessing contract '
            f'for {task_root}: {failed_contract_fields}'
        )
    if dataset_key in {
        'tusz', 'siena', 'chbmit', 'chbmit_historical', 'chbmit_future',
    }:
        cache_protocol = contract.get('preprocessing_protocol')
        if cache_protocol != SCALP_PREPROCESSING_PROTOCOL:
            raise ValueError(
                f'Scalp cache protocol is stale for {task_root}: {cache_protocol} '
                f'vs required {SCALP_PREPROCESSING_PROTOCOL}'
            )
    actual_window = float(contract.get('window_seconds', -1.0))
    if abs(actual_window - spec.window_seconds) > 1e-9:
        raise ValueError(
            f'Cache window mismatch for {task_root}: {actual_window} vs {spec.window_seconds}'
        )
    manifest = pd.read_csv(
        task_root / 'manifest.csv',
        dtype={'patient_id': str, 'clip_id': str},
    )
    required_columns = {
        'patient_id', 'clip_id', 'split', 'label', 'relative_path',
        'dataset', 'montage', 'channel_names', 'source_relative_path',
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(f'Manifest lacks required columns: {sorted(missing_columns)}')
    if manifest.empty or manifest['clip_id'].duplicated().any():
        raise ValueError(f'Manifest is empty or has duplicate clip identifiers: {task_root}')
    expected_splits = {'train', 'dev', 'test'}
    actual_splits = set(manifest['split'].dropna().astype(str))
    if not expected_splits.issubset(actual_splits):
        raise ValueError(f'Manifest lacks train, dev, or test split: {task_root}')
    patient_split_count = manifest.groupby('patient_id')['split'].nunique()
    leaked = patient_split_count[patient_split_count > 1].index.astype(str).tolist()
    if leaked and spec.mission_type != 'cross_time':
        raise ValueError(f'Patient-level split leakage in {task_root}: {leaked[:20]}')
    invalid_labels = sorted(set(manifest['label'].dropna().astype(int)) - {0, 1})
    if invalid_labels:
        raise ValueError(f'Non-binary labels in {task_root}: {invalid_labels}')
    for split in ('train', 'dev', 'test'):
        labels = set(manifest.loc[manifest['split'] == split, 'label'].astype(int))
        if labels != {0, 1}:
            raise ValueError(f'{split} split must contain both classes in {task_root}')
    sample_rows = (
        manifest.sort_values('clip_id', kind='mergesort')
        .groupby(['split', 'label'], sort=True, as_index=False)
        .head(1)
    )
    required_archive_keys = {'eeg', 'label', 'sfreq', 'channel_names'}
    if dataset_key in {
        'tusz', 'siena', 'chbmit', 'chbmit_historical', 'chbmit_future',
    }:
        required_archive_keys |= {'channel_mean_uv', 'channel_std_uv', 'metadata_json'}
    if dataset_key.endswith('_ieeg'):
        required_archive_keys |= {
            'channel_types', 'channel_positions', 'metadata_json',
        }
    checked = []
    for row in sample_rows.itertuples(index=False):
        clip_path = task_root / row.relative_path
        if not clip_path.exists():
            raise FileNotFoundError(f'Manifest clip does not exist: {clip_path}')
        with np.load(clip_path, allow_pickle=False) as archive:
            missing_keys = required_archive_keys - set(archive.files)
            if missing_keys:
                raise ValueError(f'Clip archive lacks keys {sorted(missing_keys)}: {clip_path}')
            eeg = np.asarray(archive['eeg'])
            names = np.asarray(archive['channel_names'])
            if eeg.ndim != 2 or eeg.shape[0] != len(names):
                raise ValueError(f'Clip signal and channel metadata mismatch: {clip_path}')
            if int(archive['label']) != int(row.label):
                raise ValueError(f'Clip label and manifest label mismatch: {clip_path}')
            if not np.isfinite(eeg).all():
                raise ValueError(f'Clip contains non-finite values: {clip_path}')
            if dataset_key in {
                'tusz', 'siena', 'chbmit', 'chbmit_historical', 'chbmit_future',
            }:
                means = np.asarray(archive['channel_mean_uv'])
                stds = np.asarray(archive['channel_std_uv'])
                if means.shape != (eeg.shape[0],) or stds.shape != (eeg.shape[0],):
                    raise ValueError(f'Clip channel scale metadata mismatch: {clip_path}')
                metadata = json.loads(str(archive['metadata_json'].item()))
                if metadata.get('normalization') not in {'per_clip_per_channel_zscore', 'none'}:
                    raise ValueError(f'Clip normalization contract is unsupported: {clip_path}')
        checked.append(str(row.clip_id))
    return {
        'dataset': dataset_key,
        'root': str(task_root),
        'fingerprint': str(contract.get('fingerprint', '')),
        'protocol_track': str(contract.get('protocol_track', '')),
        'sample_count': int(len(manifest)),
        'patient_count': int(manifest['patient_id'].nunique()),
        'split_counts': {
            str(key): int(value)
            for key, value in manifest['split'].value_counts().sort_index().items()
        },
        'label_counts': {
            str(key): int(value)
            for key, value in manifest['label'].astype(int).value_counts().sort_index().items()
        },
        'checked_clip_ids': checked,
        'prediction_rule': contract.get('prediction_rule'),
    }


def prepare_mission(spec: MissionSpec) -> tuple[Path, Path, PatientBudgetSelection | None]:
    source_root = spec.cache_dir(spec.source_dataset)
    target_root = spec.cache_dir(spec.target_dataset)
    source_audit = validate_cache(source_root, spec, spec.source_dataset)
    target_audit = validate_cache(target_root, spec, spec.target_dataset)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    (spec.output_dir / 'protocol.json').write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    (spec.output_dir / 'cache_validation.json').write_text(
        json.dumps({'source': source_audit, 'target': target_audit}, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    if spec.mission_type in {'eeg_ieeg_transfer', 'eeg_ieeg_localization'}:
        from eeg_benchmark.tasks.cross_modal import write_cross_modal_audit

        write_cross_modal_audit(
            spec.output_dir / 'cross_modal_audit.json',
            source_root,
            target_root,
            spec.source_dataset,
            spec.target_dataset,
            spec.task,
            source_rehearsal_fraction=spec.source_rehearsal_fraction,
        )
    if spec.budget_percent == 0.0:
        return source_root, target_root, None
    budget_task = 'detection' if spec.task == 'localization' else spec.task
    selector = (
        select_temporal_budget_patients
        if spec.mission_type == 'cross_time'
        else select_budget_patients
    )
    selection = selector(
        target_root / 'manifest.csv',
        spec.target_dataset,
        spec.budget_percent,
        spec.budget_seed,
        spec.output_dir,
        source_rehearsal_fraction=spec.source_rehearsal_fraction,
        task=budget_task,
        negative_to_positive_ratio=(
            2.0
            if budget_task == 'detection'
            else prediction_negative_to_positive_ratio(spec.mission_type)
        ),
    )
    return source_root, target_root, selection


def validate_zero_shot_reference(
    spec: MissionSpec,
    checkpoint_path: Path,
    expected_arguments: dict[str, object] | None = None,
    current_contract: object | None = None,
    reference_output_dir: Path | None = None,
    reference_task: str | None = None,
    reference_mission_type: str | None = None,
) -> dict[str, object]:
    reference_root = reference_output_dir or spec.zero_shot_output_dir
    protocol_path = reference_root / 'protocol.json'
    metrics_path = reference_root / 'metrics.json'
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f'BudgetFT requires a reference metrics file: {metrics_path}'
        )
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f'BudgetFT requires a non-empty reference checkpoint: {checkpoint_path}'
        )
    zero_cache_path = reference_root / 'cache_validation.json'
    zero_args_path = reference_root / 'args.json'
    required_reference_files = [protocol_path, zero_cache_path, zero_args_path]
    missing_reference_files = [
        str(path)
        for path in required_reference_files
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_reference_files:
        raise FileNotFoundError(
            f'BudgetFT reference is incomplete: {missing_reference_files}'
        )
    try:
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        zero_cache = json.loads(zero_cache_path.read_text(encoding='utf-8'))
        zero_args = json.loads(zero_args_path.read_text(encoding='utf-8'))
        current_source = json.loads(
            (spec.cache_dir(spec.source_dataset) / 'dataset_contract.json').read_text(
                encoding='utf-8'
            )
        )
        current_target = json.loads(
            (spec.cache_dir(spec.target_dataset) / 'dataset_contract.json').read_text(
                encoding='utf-8'
            )
        )
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError('BudgetFT reference metadata is invalid') from exc
    reference_warnings: dict[str, object] = {}
    fingerprint_mismatches = {}
    for role, current in (('source', current_source), ('target', current_target)):
        previous = zero_cache.get(role, {}).get('fingerprint')
        if not previous or previous != current.get('fingerprint'):
            fingerprint_mismatches[role] = {
                'zero_shot': previous,
                'current': current.get('fingerprint'),
            }
    if fingerprint_mismatches:
        reference_warnings['preprocessing_fingerprint_mismatches'] = fingerprint_mismatches
        print(
            'WARNING_BUDGET_REFERENCE_PREPROCESSING_FINGERPRINT_MISMATCH '
            f'{json.dumps(fingerprint_mismatches, ensure_ascii=False, sort_keys=True)}',
            flush=True,
        )
    reference_expected_arguments = dict(expected_arguments or {})
    if (
        spec.mission_type == 'eeg_ieeg_localization'
        and spec.model == 'RIVER'
    ):
        structural_keys = {
            'river_version',
            'river_architecture_version',
            'river_sampling_frequency',
            'river_patch_points',
            'river_descriptor_dim',
            'river_use_full_clip',
            'river_descriptor_hidden_dim',
            'river_descriptor_window_seconds',
            'river_descriptor_slow_seconds',
            'river_descriptor_channel_chunk_size',
            'river_latent_electrodes',
            'river_embed_dim',
            'river_depth',
            'river_heads',
            'river_ff_dim',
            'river_fourier_modes',
            'river_fourier_rank',
            'river_local_kernel_size',
            'river_sinkhorn_iterations',
            'river_transport_epsilon',
            'river_electrode_mass_relaxation',
            'river_electrode_mass_temperature',
            'river_pooling_temperature',
            'river_transport_mode',
            'river_temporal_mixer',
            'river_dropout',
        }
        skipped = {
            key: {
                'expected': value,
                'actual': zero_args.get(key),
                'reason': 'not_required_for_localization_checkpoint_reuse',
            }
            for key, value in reference_expected_arguments.items()
            if key not in structural_keys
        }
        if skipped:
            reference_warnings['relaxed_river_localization_reference_arguments'] = skipped
            print(
                'WARNING_RIVER_LOCALIZATION_REFERENCE_ARGUMENTS_RELAXED '
                f'{json.dumps(skipped, ensure_ascii=False, sort_keys=True)}',
                flush=True,
            )
        reference_expected_arguments = {
            key: value
            for key, value in reference_expected_arguments.items()
            if key in structural_keys and key in zero_args
        }
    stable_protocol = {
        'model': spec.model,
        'task': reference_task or spec.task,
        'source_dataset': spec.source_dataset,
        'target_dataset': spec.target_dataset,
        'mission_type': reference_mission_type or spec.mission_type,
        'window_seconds': spec.window_seconds,
        'budget_percent': (
            spec.budget_percent
            if spec.mission_type == 'eeg_ieeg_localization'
            else 0.0
        ),
        'budget_seed': spec.budget_seed,
        'use_pretrained': spec.use_pretrained,
        'seed': spec.seed,
        'batch_size': spec.batch_size,
        'undersample_seed': spec.undersample_seed,
        'deterministic': spec.deterministic,
        'preprocess_root': str(spec.preprocess_root),
        'cache_root': str(spec.cache_root),
    }
    reference_mismatches = {}
    for field, expected in stable_protocol.items():
        actual = protocol.get(field)
        if not _same_value(actual, expected):
            reference_mismatches[f'protocol.{field}'] = {
                'expected': expected, 'actual': actual,
            }
    for field, expected in reference_expected_arguments.items():
        actual = zero_args.get(field)
        if not _same_value(actual, expected):
            reference_mismatches[f'args.{field}'] = {
                'expected': expected, 'actual': actual,
            }
    if reference_mismatches:
        raise ValueError(
            'BudgetFT reference checkpoint contract mismatch: '
            f'{reference_mismatches}'
        )
    if current_contract is not None:
        contract_path = reference_root / 'channel_union.json'
        if not contract_path.is_file():
            raise FileNotFoundError(
                f'BudgetFT requires reference channel contract: {contract_path}'
            )
        try:
            zero_contract = json.loads(contract_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f'Reference channel contract is invalid: {contract_path}'
            ) from exc
        current_fingerprint = getattr(current_contract, 'fingerprint', None)
        if (
            not current_fingerprint
            or zero_contract.get('fingerprint') != current_fingerprint
        ):
            reference_warnings['channel_contract_fingerprint_mismatch'] = {
                'zero_shot': zero_contract.get('fingerprint'),
                'current': current_fingerprint,
            }
            print(
                'WARNING_BUDGET_REFERENCE_CHANNEL_FINGERPRINT_MISMATCH '
                + json.dumps(
                    reference_warnings['channel_contract_fingerprint_mismatch'],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    expected_source_sampling = training_sampling_policy(
        reference_task or spec.task,
        spec.source_dataset,
        single_fit=spec.model in TRADITIONAL_ML_MODELS,
        mission_type=reference_mission_type or spec.mission_type,
    )
    actual_source_sampling = protocol.get('source_training_sampling')
    if actual_source_sampling != expected_source_sampling:
        legacy_source_sampling = training_sampling_policy(
            reference_task or spec.task,
            spec.source_dataset,
            single_fit=False,
            mission_type=reference_mission_type or spec.mission_type,
        )
        if (
            spec.model in {'CMMN', 'RandomForest'}
            and actual_source_sampling == legacy_source_sampling
        ):
            reference_warnings['legacy_traditional_ml_source_sampling_metadata'] = {
                'expected': expected_source_sampling,
                'actual': actual_source_sampling,
                'reason': (
                    'CMMN_and_RandomForest_zero_budget_runs_before_2026_09_01_'
                    'used_single_fit_training_but_wrote_deep_dynamic_sampling_'
                    'metadata_from_the_shared_MissionSpec_contract'
                ),
            }
            print(
                'WARNING_LEGACY_TRADITIONAL_ML_SOURCE_SAMPLING_METADATA '
                + json.dumps(
                    reference_warnings[
                        'legacy_traditional_ml_source_sampling_metadata'
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            raise ValueError(
                'BudgetFT reference source checkpoint does not use the current '
                f'source sampling protocol: expected {expected_source_sampling}'
            )
    if reference_warnings:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        (spec.output_dir / 'budget_reference_warnings.json').write_text(
            json.dumps(reference_warnings, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    return protocol


def add_mission_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        '--mission-type',
        choices=[
            'dataset_transfer',
            'eeg_ieeg_transfer',
            'eeg_ieeg_localization',
            'cross_time',
            'ieeg_indomain',
        ],
        default='dataset_transfer',
    )
    parser.add_argument('--task', choices=['detection', 'prediction', 'localization'], required=True)
    parser.add_argument('--mode', choices=['SourceOnly', 'FT'], required=True)
    parser.add_argument(
        '--source-dataset',
        choices=[
            'tusz', 'siena', 'chbmit', 'chbmit_historical',
            'epilepsy_ieeg', 'hup_ieeg', 'thalamocortical_ieeg',
        ],
        required=True,
    )
    parser.add_argument(
        '--target-dataset',
        choices=[
            'tusz', 'siena', 'chbmit', 'chbmit_future',
            'epilepsy_ieeg', 'hup_ieeg', 'thalamocortical_ieeg',
        ],
        required=True,
    )
    parser.add_argument('--use-pretrained', type=int, choices=[0, 1], default=1)
    parser.add_argument('--window-seconds', type=float, default=60.0)
    parser.add_argument('--budget-percent', type=float, default=0.0)
    parser.add_argument('--budget-seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--budget-epochs', type=int, default=50)
    parser.add_argument('--budget-patience', type=int, default=10)
    parser.add_argument('--budget-head-only-epochs', type=int, default=5)
    parser.add_argument('--target-backbone-lr', type=float, default=1e-5)
    parser.add_argument('--min-delta', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--stats-num-workers', type=int, default=8)
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--physical-gpu', type=int, default=0)
    parser.add_argument('--deterministic', type=int, choices=[0, 1], default=1)
    parser.add_argument('--undersample-seed', type=int, default=1)
    parser.add_argument('--source-rehearsal-fraction', type=float, default=0.25)
    parser.add_argument('--generate-interpretability', type=int, choices=[0, 1], default=0)
    parser.add_argument('--interpretability-max-clips', type=int, default=2000)
    parser.add_argument('--mri-template-path', type=Path)
    parser.add_argument('--benchmark-root', type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument('--preprocess-root', type=Path, default=DEFAULT_PREPROCESS_ROOT)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        '--cross-modal-result-root',
        type=Path,
        default=DEFAULT_LOCALIZATION_REFERENCE_RESULT_ROOT,
        help='Completed EEG-iEEG detection result root used by localization as reference.',
    )
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    expected = 'SourceOnly' if args.budget_percent == 0.0 else 'FT'
    if args.mode != expected:
        raise ValueError(
            f'Mode {args.mode} conflicts with budget {args.budget_percent:g}; expected {expected}'
        )
    if args.budget_epochs <= 0 or args.budget_patience <= 0:
        raise ValueError('budget epochs and patience must be positive')
    if args.budget_head_only_epochs < 0:
        raise ValueError('budget_head_only_epochs must be non-negative')
    target_lr = getattr(args, 'target_lr', None)
    if target_lr is not None and target_lr <= 0.0:
        raise ValueError('target_lr must be positive')
    target_backbone_lr = getattr(args, 'target_backbone_lr', None)
    if target_backbone_lr is not None and target_backbone_lr <= 0.0:
        raise ValueError('target_backbone_lr must be positive')
    if (
        target_lr is not None
        and target_backbone_lr is not None
        and target_backbone_lr > target_lr
    ):
        raise ValueError('target_backbone_lr must not exceed target_lr')




import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path



def pretrained_value(model: str, requested: str) -> int:
    available = MODEL_REGISTRY[model]['pretrained'] is not None
    if requested == 'auto':
        return int(available)
    value = int(requested)
    if value and not available:
        raise ValueError(f'{model} has no official pretrained weight')
    return value


def parse_direction(value: str) -> tuple[str, str]:
    parts = value.split(':')
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError('Direction must use source:target')
    return parts[0], parts[1]


def sanitize_process_token(value: object) -> str:
    token = re.sub(r'[^A-Za-z0-9_.%+-]+', '_', str(value)).strip('_')
    return token or 'unknown'


def process_name_from_python_executable(python_executable: str) -> str:
    executable_path = Path(python_executable)
    if executable_path.parent.name == 'bin':
        return sanitize_process_token(executable_path.parent.parent.name)
    return sanitize_process_token(executable_path.stem or executable_path.name)


def process_name_for_spec(process_name: str, python_executable: str) -> str:
    if process_name != 'auto':
        return sanitize_process_token(process_name)
    return process_name_from_python_executable(python_executable)


def anonymized_process_invocation(
    command: list[str], process_name: str, spec: MissionSpec | None = None,
) -> tuple[list[str], str]:
    if not command:
        raise ValueError('Cannot launch an empty command')
    del spec
    resolved_process_name = process_name_for_spec(process_name, command[0])
    if not resolved_process_name or any(character.isspace() for character in resolved_process_name):
        raise ValueError('Process name must be one non-empty token')
    return [resolved_process_name, *command[1:]], command[0]


def run_mission_process(
    command: list[str],
    executable: str,
    spec: MissionSpec,
    environment: dict[str, str],
) -> int:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = spec.output_dir / 'launcher.log'
    with log_path.open('w', encoding='utf-8', buffering=1) as stream:
        real_command = [executable, *command[1:]]
        stream.write(f'COMMAND={shlex.join(command)}\n')
        stream.write(f'REAL_COMMAND={shlex.join(real_command)}\n')
        stream.write(f'EXECUTABLE={executable}\n')
        process = subprocess.Popen(
            command,
            executable=executable,
            cwd=spec.benchmark_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError('Failed to capture child process output')
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
        returncode = int(process.wait())
    signal_name = None
    if returncode < 0:
        try:
            signal_name = signal.Signals(-returncode).name
        except ValueError:
            signal_name = f'UNKNOWN_SIGNAL_{-returncode}'
    status = {
        'status': 'complete' if returncode == 0 else 'failed',
        'returncode': returncode,
        'signal': signal_name,
        'command': command,
        'real_command': real_command,
        'process_name': command[0],
        'executable': executable,
        'launcher_log': str(log_path),
        'likely_system_oom': signal_name == 'SIGKILL',
    }
    (spec.output_dir / 'launcher_status.json').write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return returncode


def build_command(args: argparse.Namespace, spec: MissionSpec) -> list[str]:
    registry = MODEL_REGISTRY[spec.model]
    python_executable = (
        args.eegnet_python if spec.model == 'EEGNet' else Path(sys.executable)
    )
    learning_rate = args.lr if args.lr is not None else registry.get('lr')
    target_learning_rate = args.target_lr if args.target_lr is not None else registry.get('target_lr')
    target_backbone_learning_rate = (
        args.target_backbone_lr
        if args.target_backbone_lr is not None
        else registry.get('target_backbone_lr')
    )
    weight_decay = args.weight_decay if args.weight_decay is not None else registry.get('weight_decay')
    maximum_gradient_norm = args.max_grad_norm if args.max_grad_norm is not None else registry.get('max_grad_norm')
    command = [
        str(python_executable),
        str(spec.benchmark_root / MODEL_REGISTRY[spec.model]['script']),
        '--model', spec.model,
        '--mission-type', spec.mission_type,
        '--task', spec.task,
        '--mode', spec.mode,
        '--source-dataset', spec.source_dataset,
        '--target-dataset', spec.target_dataset,
        '--use-pretrained', str(int(spec.use_pretrained)),
        '--window-seconds', str(spec.window_seconds),
        '--budget-percent', str(spec.budget_percent),
        '--budget-seed', str(spec.budget_seed),
        '--epochs', str(spec.epochs),
        '--patience', str(spec.patience),
        '--min-delta', str(spec.min_delta),
        '--seed', str(spec.seed),
        '--batch-size', str(spec.batch_size),
        '--num-workers', str(spec.num_workers),
        '--stats-num-workers', str(spec.stats_num_workers),
        '--bootstrap-resamples', str(spec.bootstrap_resamples),
        '--gpu', str(spec.gpu),
        '--physical-gpu', str(spec.physical_gpu),
        '--deterministic', str(int(spec.deterministic)),
        '--undersample-seed', str(spec.undersample_seed),
        '--source-rehearsal-fraction', str(spec.source_rehearsal_fraction),
        '--generate-interpretability', str(int(spec.generate_interpretability)),
        '--interpretability-max-clips', str(spec.interpretability_max_clips),
        '--benchmark-root', str(spec.benchmark_root),
        '--preprocess-root', str(spec.preprocess_root),
        '--result-root', str(spec.result_root),
        '--cross-modal-result-root', str(spec.cross_modal_result_root),
    ]
    if spec.localization_reference_window_seconds is not None:
        command.extend([
            '--localization-reference-window-seconds',
            str(spec.localization_reference_window_seconds),
        ])
    if spec.mri_template_path is not None:
        command.extend(['--mri-template-path', str(spec.mri_template_path)])
    traditional_models = TRADITIONAL_ML_MODELS
    if spec.model not in traditional_models and learning_rate is not None:
        command.extend(['--lr', str(learning_rate)])
    if spec.model not in traditional_models and target_learning_rate is not None:
        command.extend(['--target-lr', str(target_learning_rate)])
        command.extend([
            '--target-backbone-lr', str(target_backbone_learning_rate),
        ])
        command.extend([
            '--budget-epochs', str(args.budget_epochs),
            '--budget-patience', str(args.budget_patience),
            '--budget-head-only-epochs', str(args.budget_head_only_epochs),
        ])
    if spec.model in {
        'BENDR', 'CST', 'EEGPT', 'LaBraM', 'STEEGFormer',
        'RIVER', 'ScatterFormer', 'TSMNet', 'BF-EML',
    }:
        command.extend(['--weight-decay', str(weight_decay)])
        command.extend(['--max-grad-norm', str(maximum_gradient_norm)])
        command.extend([
            '--reuse-incomplete-source-checkpoint',
            str(args.reuse_incomplete_source_checkpoint),
        ])
    elif spec.model in {'BIOT', 'CBraMod', 'EvoBrain'}:
        command.extend(['--weight-decay', str(weight_decay)])
    if spec.model == 'BIOT':
        command.extend([
            '--token-size', str(args.biot_token_size),
            '--hop-length', str(args.biot_hop_length),
        ])
    elif spec.model == 'CBraMod':
        command.extend([
            '--optimizer', args.cbramod_optimizer,
            '--clip-value', str(args.cbramod_clip_value),
            '--multi-lr', str(args.cbramod_multi_lr),
            '--prefetch-factor', str(args.cbramod_prefetch_factor),
        ])
    elif spec.model == 'BENDR':
        command.extend([
            '--prefetch-factor', str(args.bendr_prefetch_factor),
            '--progress-update-interval',
            str(args.bendr_progress_update_interval),
        ])
    elif spec.model == 'EEGPT':
        command.extend([
            '--prefetch-factor', str(getattr(args, 'eegpt_prefetch_factor', 4)),
        ])
    elif spec.model == 'STEEGFormer':
        command.extend([
            '--prefetch-factor', str(getattr(args, 'steegformer_prefetch_factor', 4)),
        ])
    elif spec.model == 'CST':
        command.extend([
            '--cst-resize-heads', str(args.cst_resize_heads),
            '--cst-resize-layers', str(args.cst_resize_layers),
            '--cst-resize-feedforward', str(args.cst_resize_feedforward),
            '--cst-kd-weight', str(args.cst_kd_weight),
            '--cst-mcc-weight', str(args.cst_mcc_weight),
            '--cst-kd-temperature', str(args.cst_kd_temperature),
            '--cst-mcc-temperature', str(args.cst_mcc_temperature),
        ])
    elif spec.model == 'EvoBrain':
        command.extend([
            '--max-grad-norm', str(
                args.max_grad_norm if args.max_grad_norm is not None
                else args.evobrain_max_grad_norm
            ),
            '--rnn-units', str(args.evobrain_rnn_units),
            '--agg', args.evobrain_agg,
            '--top-k', str(args.evobrain_top_k),
        ])
    elif spec.model == 'RIVER':
        alignment_weight = args.river_alignment_weight
        command.extend([
            '--river-descriptor-hidden-dim',
            str(args.river_descriptor_hidden_dim),
            '--river-descriptor-window-seconds',
            str(args.river_descriptor_window_seconds),
            '--river-descriptor-channel-chunk-size',
            str(args.river_descriptor_channel_chunk_size),
            '--river-latent-electrodes', str(args.river_latent_electrodes),
            '--river-embed-dim', str(args.river_embed_dim),
            '--river-depth', str(args.river_depth),
            '--river-heads', str(args.river_heads),
            '--river-ff-dim', str(args.river_ff_dim),
            '--river-fourier-modes', str(args.river_fourier_modes),
            '--river-fourier-rank', str(args.river_fourier_rank),
            '--river-local-kernel-size', str(args.river_local_kernel_size),
            '--river-evidence-smoothing-kernel',
            str(args.river_evidence_smoothing_kernel),
            '--river-sinkhorn-iterations',
            str(args.river_sinkhorn_iterations),
            '--river-transport-epsilon', str(args.river_transport_epsilon),
            '--river-electrode-mass-relaxation',
            str(args.river_electrode_mass_relaxation),
            '--river-electrode-mass-temperature',
            str(args.river_electrode_mass_temperature),
            '--river-pooling-temperature',
            str(args.river_pooling_temperature),
            '--river-transport-mode', args.river_transport_mode,
            '--river-temporal-mixer', args.river_temporal_mixer,
            '--river-alignment-weight', str(alignment_weight),
            '--river-channel-drop-enabled',
            str(args.river_channel_drop_enabled),
            '--river-fuse-augmentation-views',
            str(args.river_fuse_augmentation_views),
            '--river-channel-drop-ratio', str(args.river_channel_drop_ratio),
            '--river-prototype-momentum', str(args.river_prototype_momentum),
            '--river-dropout', str(args.river_dropout),
        ])
    elif spec.model == 'SVM':
        command.extend([
            '--c-grid', *[str(value) for value in args.svm_c_grid],
            '--max-iter', str(args.svm_max_iter),
        ])
    elif spec.model == 'RandomForest':
        command.extend([
            '--n-estimators', str(args.rf_n_estimators),
            '--max-features', str(args.rf_max_features),
            '--min-samples-leaf', str(args.rf_min_samples_leaf),
        ])
        if args.rf_max_depth is not None:
            command.extend(['--max-depth', str(args.rf_max_depth)])
    elif spec.model == 'CMMN':
        command.extend([
            '--c', str(args.cmmn_c),
            '--max-iter', str(args.cmmn_max_iter),
        ])
    if args.interpretability_only:
        if spec.model != 'EEGNet':
            raise ValueError('Interpretability-only mode is currently supported only for EEGNet')
        command.append('--interpretability-only')
    return command


def expected_model_arguments(args: argparse.Namespace, spec: MissionSpec) -> dict[str, object]:
    if spec.model == 'SVM':
        return {
            'c_grid': [float(value) for value in args.svm_c_grid],
            'max_iter': int(args.svm_max_iter),
        }
    if spec.model == 'RandomForest':
        return {
            'n_estimators': int(args.rf_n_estimators),
            'max_depth': args.rf_max_depth,
            'max_features': args.rf_max_features,
            'min_samples_leaf': int(args.rf_min_samples_leaf),
            'random_seed': int(spec.seed),
        }
    if spec.model == 'CMMN':
        return {
            'classifier_c': float(args.cmmn_c),
            'max_iter': int(args.cmmn_max_iter),
            'cmmn_feature_transform': 'train_domain_mean_variance_frequency_feature_normalization',
            'random_seed': int(spec.seed),
        }
    registry = MODEL_REGISTRY[spec.model]
    learning_rate = args.lr if args.lr is not None else registry.get('lr')
    target_learning_rate = args.target_lr if args.target_lr is not None else registry.get('target_lr')
    target_backbone_learning_rate = (
        args.target_backbone_lr
        if args.target_backbone_lr is not None
        else registry.get('target_backbone_lr')
    )
    weight_decay = args.weight_decay if args.weight_decay is not None else registry.get('weight_decay')
    maximum_gradient_norm = args.max_grad_norm if args.max_grad_norm is not None else registry.get('max_grad_norm')
    common = {
        'learning_rate': learning_rate,
        'target_learning_rate': target_learning_rate,
        'target_backbone_learning_rate': target_backbone_learning_rate,
    }
    if spec.model == 'BIOT':
        return {
            **common, 'weight_decay': weight_decay,
            'token_size': args.biot_token_size, 'hop_length': args.biot_hop_length,
        }
    if spec.model == 'CBraMod':
        return {
            **common, 'weight_decay': weight_decay,
            'optimizer': args.cbramod_optimizer,
            'clip_value': args.cbramod_clip_value,
            'multi_lr': args.cbramod_multi_lr,
            'prefetch_factor': args.cbramod_prefetch_factor,
        }
    if spec.model == 'BENDR':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'pretrained_channels': 20,
            'encoder_width': 512,
            'transformer_layers': 8,
            'transformer_heads': 8,
            'view_seconds': 4.0,
            'view_sampling_frequency': 250,
            'channel_projection': 'identity_initialized_learnable_1x1',
        }
    if spec.model == 'EEGPT':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
        }
    if spec.model == 'STEEGFormer':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
        }
    if spec.model == 'CST':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'resize_heads': args.cst_resize_heads,
            'resize_layers': args.cst_resize_layers,
            'resize_feedforward': args.cst_resize_feedforward,
            'kd_weight': args.cst_kd_weight,
            'mcc_weight': args.cst_mcc_weight,
            'kd_temperature': args.cst_kd_temperature,
            'mcc_temperature': args.cst_mcc_temperature,
            'feature_extractor': 'EEGNet_F1_8_D_2_F2_16',
            'input_sampling_frequency': 128,
            'view_seconds': 1.0,
            'ea_policy': 'shared_per_clip_channel_zscore_no_target_test_fit',
            'zero_budget_target_access': 'none',
        }
    if spec.model == 'EvoBrain':
        return {
            **common, 'weight_decay': weight_decay,
            'maximum_gradient_norm': (
                args.max_grad_norm
                if args.max_grad_norm is not None
                else args.evobrain_max_grad_norm
            ),
            'rnn_units': args.evobrain_rnn_units,
            'aggregation': args.evobrain_agg,
            'top_k': args.evobrain_top_k,
        }
    if spec.model == 'RIVER':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'river_version': spec.model,
            'river_architecture_version': 'RIVER_Final',
            'river_sampling_frequency': 256,
            'river_patch_points': 256,
            'river_descriptor_dim': 16,
            'river_use_full_clip': True,
            'river_descriptor_hidden_dim': args.river_descriptor_hidden_dim,
            'river_descriptor_window_seconds': args.river_descriptor_window_seconds,
            'river_descriptor_slow_seconds': 16.0,
            'river_descriptor_channel_chunk_size': args.river_descriptor_channel_chunk_size,
            'river_latent_electrodes': args.river_latent_electrodes,
            'river_embed_dim': args.river_embed_dim,
            'river_depth': args.river_depth,
            'river_heads': args.river_heads,
            'river_ff_dim': args.river_ff_dim,
            'river_fourier_modes': args.river_fourier_modes,
            'river_fourier_rank': args.river_fourier_rank,
            'river_local_kernel_size': args.river_local_kernel_size,
            'river_sinkhorn_iterations': args.river_sinkhorn_iterations,
            'river_transport_epsilon': args.river_transport_epsilon,
            'river_electrode_mass_relaxation': args.river_electrode_mass_relaxation,
            'river_electrode_mass_temperature': args.river_electrode_mass_temperature,
            'river_pooling_temperature': args.river_pooling_temperature,
            'river_transport_mode': args.river_transport_mode,
            'river_temporal_mixer': args.river_temporal_mixer,
            'river_alignment_weight': args.river_alignment_weight,
            'river_channel_drop_enabled': bool(args.river_channel_drop_enabled),
            'river_fuse_augmentation_views': bool(
                args.river_fuse_augmentation_views
            ),
            'river_channel_drop_ratio': args.river_channel_drop_ratio,
            'river_prototype_momentum': args.river_prototype_momentum,
            'river_dropout': args.river_dropout,
        }
    if spec.model in {'EEGPT', 'LaBraM', 'STEEGFormer'}:
        return {
            **common, 'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
        }
    if spec.model == 'ScatterFormer':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'scatterformer_input_adapter': 'steegformer_continuous_views_to_three_channel_time_channel_image',
            'scatterformer_image_size': [64, 64],
        }
    if spec.model == 'TSMNet':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'tsmnet_temporal_filters': 24,
            'tsmnet_subspace_dim': 24,
            'tsmnet_input_adapter': 'steegformer_continuous_views_to_temporal_spatial_covariance',
            'tsmnet_covariance_features': 'upper_triangular_plus_log_diagonal',
        }
    if spec.model == 'BF-EML':
        return {
            **common,
            'weight_decay': weight_decay,
            'maximum_gradient_norm': maximum_gradient_norm,
            'bfeml_input_adapter': 'steegformer_continuous_views_to_original_bfeml_views',
            'bfeml_view0_shape': [23, 256],
            'bfeml_view1_shape': [23, 27],
            'bfeml_view2_shape': [23, 14, 256],
            'bfeml_original_loss_reused': False,
            'bfeml_runtime_loss': 'binary_cross_entropy_with_logits',
        }
    if spec.model == 'EEGNet':
        return common
    raise ValueError(f'Unsupported completion argument contract: {spec.model}')


def expected_checkpoint(spec: MissionSpec) -> Path:
    if spec.model in TRADITIONAL_ML_MODELS:
        return spec.output_dir / 'best_model.joblib'
    stage = 'source' if spec.budget_percent == 0.0 else 'target'
    names = {
        'BENDR': 'best.pt',
        'BIOT': 'best.ckpt',
        'CBraMod': 'best.pth',
        'CST': 'best.pt',
        'EEGNet': 'best.weights.h5',
        'EvoBrain': 'best.pth.tar',
        'EEGPT': 'best.pt',
        'LaBraM': 'best.pt',
        'STEEGFormer': 'best.pt',
        'RIVER': 'best.pt',
    }
    return spec.output_dir / stage / names[spec.model]


def completed_result_status(
    args: argparse.Namespace,
    spec: MissionSpec,
) -> tuple[bool, str]:
    metrics_path = spec.output_dir / 'metrics.json'
    if metrics_path.is_file():
        return True, 'metrics_file_present'
    return False, 'metrics_file_missing'


def build_specs(args: argparse.Namespace) -> tuple[list[MissionSpec], list[str]]:
    specs = []
    skipped = []
    budgets = list(dict.fromkeys(float(value) for value in args.budget_percent))
    if any(value > 0.0 for value in budgets) and 0.0 not in budgets:
        budgets.insert(0, 0.0)
    directions = getattr(args, 'directions', None)
    if not directions:
        directions = [(args.source_dataset, args.target_dataset)]
    for source_dataset, target_dataset in directions:
        for window_seconds in args.window_seconds:
            for task in args.tasks:
                for model in args.models:
                    for budget_percent in budgets:
                        if budget_percent > 0.0 and not MODEL_REGISTRY[model]['budget_ft']:
                            skipped.append(
                                f'{model} {budget_percent:g}% budget is not applicable and was skipped'
                            )
                            continue
                        use_pretrained = bool(pretrained_value(model, args.use_pretrained))
                        try:
                            specs.append(MissionSpec(
                            model=model,
                            task=task,
                            source_dataset=source_dataset,
                            target_dataset=target_dataset,
                            window_seconds=window_seconds,
                            mission_type=args.mission_type,
                            budget_percent=budget_percent,
                            budget_seed=args.budget_seed,
                            use_pretrained=use_pretrained,
                            epochs=args.epochs,
                            patience=args.patience,
                            min_delta=args.min_delta,
                            seed=args.seed,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            stats_num_workers=getattr(args, 'stats_num_workers', 8),
                            bootstrap_resamples=getattr(args, 'bootstrap_resamples', 2000),
                            gpu=args.gpu,
                            physical_gpu=getattr(args, 'physical_gpu', args.gpu),
                            deterministic=bool(args.deterministic),
                            undersample_seed=args.undersample_seed,
                            source_rehearsal_fraction=getattr(
                                args, 'source_rehearsal_fraction', 0.25
                            ),
                            generate_interpretability=bool(args.generate_interpretability),
                            interpretability_max_clips=args.interpretability_max_clips,
                            mri_template_path=args.mri_template_path,
                            benchmark_root=args.benchmark_root,
                            preprocess_root=args.preprocess_root,
                            result_root=result_root_for_model(
                                args,
                                model,
                                args.mission_type,
                            ),
                            cross_modal_result_root=args.cross_modal_result_root,
                            localization_reference_window_seconds=(
                                args.localization_reference_window_seconds
                            ),
                            new_baseline_result_root=args.new_baseline_result_root,
                            ))
                        except ValueError as exc:
                            skipped.append(f'{model} {task} {budget_percent:g}% skipped: {exc}')
    return specs, skipped


def print_mission_table(specs: list[MissionSpec]) -> None:
    print('|Index|Mission|Model|Window|Task|Direction|Budget|Pretrained|GPU|Seed|Budget seed|Sampling seed|Min delta|Bootstrap resamples|Statistics CPU workers|Training sampling|')
    print('|--:|:--|:--|--:|:--|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|:--|')
    for index, spec in enumerate(specs, start=1):
        sampling = mission_sampling_summary(spec, single_fit=spec.model in TRADITIONAL_ML_MODELS)
        print(
            f'|{index}|{spec.mission_type}|{spec.model}|{spec.window_name}|{spec.task}|{spec.direction}|'
            f'{spec.budget_percent:g}%|{spec.use_pretrained}|{spec.physical_gpu}|{spec.seed}|'
            f'{spec.budget_seed}|{spec.undersample_seed}|{spec.min_delta:g}|'
            f'{spec.bootstrap_resamples}|{spec.stats_num_workers}|{sampling}|'
        )


def environment_for_spec(
    base_environment: dict[str, str],
    args: argparse.Namespace,
    spec: MissionSpec,
) -> dict[str, str]:
    environment = base_environment.copy()
    if spec.model != 'EEGNet':
        return environment
    if not args.eegnet_python.is_file():
        raise FileNotFoundError(f'EEGNet Python does not exist: {args.eegnet_python}')
    roots = []
    if args.eegnet_cuda_root is not None:
        roots.append(args.eegnet_cuda_root)
    roots.extend((args.eegnet_python.parent.parent / 'lib').glob('python*/site-packages'))
    library_dirs = sorted({
        str(path)
        for root in roots
        for path in root.glob('nvidia/*/lib')
        if path.is_dir()
    })
    current = environment.get('LD_LIBRARY_PATH')
    if library_dirs:
        if current:
            library_dirs.append(current)
        environment['LD_LIBRARY_PATH'] = ':'.join(library_dirs)
    environment['TF_ENABLE_ONEDNN_OPTS'] = '0'
    environment['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    return environment


def run_preflight(
    args: argparse.Namespace,
    specs: list[MissionSpec],
    base_environment: dict[str, str],
) -> None:
    checked_models: set[str] = set()
    checked_caches: set[tuple[str, str, str, str, float]] = set()
    torch_gpu_checked = False
    tensorflow_gpu_checked = False
    for spec in specs:
        script = spec.benchmark_root / MODEL_REGISTRY[spec.model]['script']
        if not script.is_file():
            raise FileNotFoundError(f'Baseline entrypoint does not exist: {script}')
        pretrained_path = spec.pretrained_path()
        if spec.use_pretrained and (
            pretrained_path is None or not pretrained_path.is_file()
        ):
            raise FileNotFoundError(
                f'Pretrained weight does not exist: {pretrained_path}'
            )
        cache_key = (
            spec.mission_type,
            spec.task,
            spec.source_dataset,
            spec.target_dataset,
            spec.window_seconds,
        )
        if cache_key not in checked_caches:
            source_root = spec.cache_dir(spec.source_dataset)
            target_root = spec.cache_dir(spec.target_dataset)
            validate_cache(source_root, spec, spec.source_dataset)
            validate_cache(target_root, spec, spec.target_dataset)
            if spec.mission_type in {'eeg_ieeg_transfer', 'eeg_ieeg_localization'}:
                from eeg_benchmark.tasks.cross_modal import audit_cross_modal_cache

                audit_cross_modal_cache(
                    source_root,
                    target_root,
                    spec.source_dataset,
                    spec.target_dataset,
                    spec.task,
                    source_rehearsal_fraction=spec.source_rehearsal_fraction,
                )
            checked_caches.add(cache_key)
            print(
                f'PREFLIGHT_CACHE_OK mission={spec.mission_type} task={spec.task} '
                f'direction={spec.direction} window={spec.window_name}'
            )
        if spec.model in checked_models:
            continue
        registry = MODEL_REGISTRY[spec.model]
        python_executable = (
            args.eegnet_python if spec.model == 'EEGNet' else Path(sys.executable)
        )
        mission_environment = environment_for_spec(
            base_environment, args, spec
        )
        if spec.model not in {'EEGNet'} | set(TRADITIONAL_ML_MODELS) and not torch_gpu_checked:
            gpu_probe = subprocess.run(
                [
                    str(python_executable), '-c',
                    'import torch; assert torch.cuda.is_available(); '
                    'assert torch.cuda.device_count() == 1; '
                    'print(torch.__version__, torch.cuda.get_device_name(0))',
                ],
                cwd=spec.benchmark_root,
                env=mission_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if gpu_probe.returncode != 0:
                raise RuntimeError(
                    f'PyTorch GPU preflight failed: {gpu_probe.stdout[-4000:]}'
                )
            torch_gpu_checked = True
            print(f'PREFLIGHT_PYTORCH_GPU_OK {gpu_probe.stdout.strip()}')
        if spec.model == 'EEGNet' and not tensorflow_gpu_checked:
            gpu_probe = subprocess.run(
                [
                    str(python_executable), '-c',
                    'import tensorflow as tf; '
                    'gpus=tf.config.list_physical_devices(\'GPU\'); '
                    'assert len(gpus) == 1, gpus; '
                    'print(tf.__version__, gpus[0])',
                ],
                cwd=spec.benchmark_root,
                env=mission_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if gpu_probe.returncode != 0:
                raise RuntimeError(
                    f'TensorFlow GPU preflight failed: {gpu_probe.stdout[-4000:]}'
                )
            tensorflow_gpu_checked = True
            print(f'PREFLIGHT_TENSORFLOW_GPU_OK {gpu_probe.stdout.strip()}')
        probe = subprocess.run(
            [str(python_executable), str(script), '--help'],
            cwd=spec.benchmark_root,
            env=mission_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            output = probe.stdout[-4000:] if probe.stdout else 'no output'
            raise RuntimeError(
                f'Baseline import preflight failed for {spec.model}: {output}'
            )
        checked_models.add(spec.model)
        print(
            f'PREFLIGHT_MODEL_OK model={spec.model} python={python_executable} '
            f'entrypoint={script}'
        )
    print(
        f'PREFLIGHT_COMPLETE models={len(checked_models)} '
        f'cache_contracts={len(checked_caches)} physical_gpu={specs[0].physical_gpu}'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Public interface for EEG benchmark transfer missions'
    )
    subparsers = parser.add_subparsers(dest='mission', required=True)
    dataset_transfer = subparsers.add_parser('dataset-transfer')
    eeg_ieeg_transfer = subparsers.add_parser('eeg-ieeg-transfer')
    eeg_ieeg_localization = subparsers.add_parser('eeg-ieeg-localization')
    cross_time = subparsers.add_parser('cross-time')
    ieeg_indomain = subparsers.add_parser('ieeg-indomain')
    for transfer, mission_type in (
        (dataset_transfer, 'dataset_transfer'),
        (eeg_ieeg_transfer, 'eeg_ieeg_transfer'),
        (eeg_ieeg_localization, 'eeg_ieeg_localization'),
        (cross_time, 'cross_time'),
        (ieeg_indomain, 'ieeg_indomain'),
    ):
        transfer.set_defaults(mission_type=mission_type)
        transfer.add_argument('--models', nargs='+', choices=sorted(MODEL_REGISTRY), required=True)
        transfer.add_argument('--tasks', nargs='+', choices=['detection', 'prediction', 'localization'], required=True)
        transfer.add_argument(
            '--source-dataset',
            choices=[
                'tusz', 'siena', 'chbmit', 'chbmit_historical',
                'epilepsy_ieeg', 'hup_ieeg', 'thalamocortical_ieeg',
            ],
        )
        transfer.add_argument(
            '--target-dataset',
            choices=[
                'tusz', 'siena', 'chbmit', 'chbmit_future',
                'epilepsy_ieeg', 'hup_ieeg', 'thalamocortical_ieeg',
            ],
        )
        transfer.add_argument(
            '--directions', nargs='+', type=parse_direction,
            help='One or more source:target pairs. Overrides the single source and target arguments.',
        )
        transfer.add_argument('--window-seconds', type=float, nargs='+', default=[60.0])
        transfer.add_argument('--budget-percent', type=float, nargs='+', default=[0.0])
        transfer.add_argument('--budget-seed', type=int, default=1)
        transfer.add_argument('--use-pretrained', choices=['auto', '0', '1'], default='auto')
        transfer.add_argument('--epochs', type=int, default=100)
        transfer.add_argument('--patience', type=int, default=20)
        transfer.add_argument('--min-delta', type=float, default=0.001)
        transfer.add_argument('--seed', type=int, default=1)
        transfer.add_argument('--batch-size', type=int, default=128)
        transfer.add_argument('--num-workers', type=int, default=8)
        transfer.add_argument('--stats-num-workers', type=int, default=8)
        transfer.add_argument('--bootstrap-resamples', type=int, default=2000)
        transfer.add_argument('--gpu', type=int, default=0)
        transfer.add_argument('--physical-gpu', type=int, default=0)
        transfer.add_argument('--deterministic', type=int, choices=[0, 1], default=1)
        transfer.add_argument('--undersample-seed', type=int, default=1)
        transfer.add_argument('--source-rehearsal-fraction', type=float, default=0.25)
        transfer.add_argument('--generate-interpretability', type=int, choices=[0, 1], default=0)
        transfer.add_argument('--interpretability-max-clips', type=int, default=2000)
        transfer.add_argument('--mri-template-path', type=Path)
        transfer.add_argument('--lr', type=float)
        transfer.add_argument('--target-lr', type=float)
        transfer.add_argument('--weight-decay', type=float)
        transfer.add_argument('--max-grad-norm', type=float)
        transfer.add_argument(
            '--reuse-incomplete-source-checkpoint', type=int, choices=[0, 1], default=0
        )
        transfer.add_argument('--biot-token-size', type=int, default=200)
        transfer.add_argument('--biot-hop-length', type=int, default=100)
        transfer.add_argument('--cbramod-optimizer', choices=['AdamW', 'SGD'], default='AdamW')
        transfer.add_argument('--cbramod-clip-value', type=float, default=1.0)
        transfer.add_argument('--cbramod-multi-lr', type=int, choices=[0, 1], default=1)
        transfer.add_argument('--cbramod-prefetch-factor', type=int, default=4)
        transfer.add_argument('--bendr-prefetch-factor', type=int, default=4)
        transfer.add_argument('--bendr-progress-update-interval', type=int, default=10)
        transfer.add_argument('--eegpt-prefetch-factor', type=int, default=4)
        transfer.add_argument('--steegformer-prefetch-factor', type=int, default=4)
        transfer.add_argument('--cst-resize-heads', type=int, default=2)
        transfer.add_argument('--cst-resize-layers', type=int, default=2)
        transfer.add_argument('--cst-resize-feedforward', type=int, default=128)
        transfer.add_argument('--cst-kd-weight', type=float, default=1.0)
        transfer.add_argument('--cst-mcc-weight', type=float, default=1.0)
        transfer.add_argument('--cst-kd-temperature', type=float, default=4.0)
        transfer.add_argument('--cst-mcc-temperature', type=float, default=2.5)
        transfer.add_argument('--evobrain-max-grad-norm', type=float, default=5.0)
        transfer.add_argument('--evobrain-rnn-units', type=int, default=64)
        transfer.add_argument('--evobrain-agg', choices=['max', 'mean', 'sum', 'concat'], default='max')
        transfer.add_argument('--evobrain-top-k', type=int, default=3)
        transfer.add_argument('--river-descriptor-hidden-dim', type=int, default=64)
        transfer.add_argument('--river-descriptor-window-seconds', type=float, default=4.0)
        transfer.add_argument('--river-descriptor-slow-seconds', type=float, default=16.0)
        transfer.add_argument('--river-descriptor-channel-chunk-size', type=int, default=256)
        transfer.add_argument('--river-latent-electrodes', type=int, default=12)
        transfer.add_argument('--river-embed-dim', type=int, default=128)
        transfer.add_argument('--river-depth', type=int, default=6)
        transfer.add_argument('--river-heads', type=int, default=8)
        transfer.add_argument('--river-ff-dim', type=int, default=384)
        transfer.add_argument('--river-fourier-modes', type=int, default=32)
        transfer.add_argument('--river-fourier-rank', type=int, default=8)
        transfer.add_argument('--river-local-kernel-size', type=int, default=5)
        transfer.add_argument('--river-slot-temperature', type=float, default=0.50)
        transfer.add_argument('--river-evidence-smoothing-kernel', type=int, default=3)
        transfer.add_argument('--river-sinkhorn-iterations', type=int, default=8)
        transfer.add_argument('--river-transport-epsilon', type=float, default=0.20)
        transfer.add_argument('--river-electrode-mass-relaxation', type=float, default=0.80)
        transfer.add_argument('--river-electrode-mass-temperature', type=float, default=0.50)
        transfer.add_argument('--river-pooling-temperature', type=float, default=0.70)
        transfer.add_argument(
            '--river-transport-mode',
            choices=['temporal_uot', 'static_uot', 'query_attention'],
            default='temporal_uot',
        )
        transfer.add_argument(
            '--river-state-encoder-mode',
            choices=['descriptor', 'descriptor_token_gate'],
            default='descriptor',
        )
        transfer.add_argument(
            '--river-temporal-mixer',
            choices=['spectral_local', 'axial_attention'],
            default='spectral_local',
        )
        transfer.add_argument('--river-diversity-weight', type=float, default=0.01)
        transfer.add_argument('--river-alignment-weight', type=float, default=0.05)
        transfer.add_argument(
            '--river-channel-drop-enabled', type=int, choices=[0, 1], default=1
        )
        transfer.add_argument(
            '--river-fuse-augmentation-views', type=int, choices=[0, 1], default=1
        )
        transfer.add_argument(
            '--river-montage-consistency-weight', type=float, default=0.02
        )
        transfer.add_argument('--river-channel-drop-ratio', type=float, default=0.15)
        transfer.add_argument('--river-channel-drop-start-ratio', type=float, default=0.05)
        transfer.add_argument(
            '--river-temporal-patch-mask-enabled',
            type=int,
            choices=[0, 1],
            default=1,
        )
        transfer.add_argument('--river-temporal-patch-mask-ratio', type=float, default=0.10)
        transfer.add_argument('--river-temporal-patch-mask-blocks', type=int, default=1)
        transfer.add_argument(
            '--river-temporal-patch-mask-min-keep-ratio',
            type=float,
            default=0.75,
        )
        transfer.add_argument('--river-prototype-momentum', type=float, default=0.95)
        transfer.add_argument(
            '--river-mass-aware-transport', type=int, choices=[0, 1], default=0
        )
        transfer.add_argument(
            '--river-residual-bottleneck-dim', type=int, default=32
        )
        transfer.add_argument(
            '--river-slot-competition-strength', type=float, default=0.65
        )
        transfer.add_argument(
            '--river-slot-competition-temperature', type=float, default=0.35
        )
        transfer.add_argument('--river-dropout', type=float, default=0.0)
        transfer.add_argument(
            '--river-rdrop-enabled', type=int, choices=[0, 1], default=1
        )
        transfer.add_argument('--river-rdrop-weight', type=float, default=0.1)
        transfer.add_argument('--river-ema-decay', type=float, default=0.999)
        transfer.add_argument(
            '--river-precision', choices=['fp32', 'bf16'], default='fp32'
        )
        transfer.add_argument('--river-warmup-ratio', type=float, default=0.05)
        transfer.add_argument('--river-min-lr-ratio', type=float, default=0.01)
        transfer.add_argument('--target-backbone-lr', type=float)
        transfer.add_argument('--budget-epochs', type=int, default=50)
        transfer.add_argument('--budget-patience', type=int, default=10)
        transfer.add_argument('--budget-head-only-epochs', type=int, default=5)
        transfer.add_argument('--svm-c-grid', type=float, nargs='+', default=[0.01, 0.1, 1.0, 10.0])
        transfer.add_argument('--svm-max-iter', type=int, default=10000)
        transfer.add_argument('--rf-n-estimators', type=int, default=500)
        transfer.add_argument('--rf-max-depth', type=int)
        transfer.add_argument('--rf-max-features', default='sqrt')
        transfer.add_argument('--rf-min-samples-leaf', type=int, default=2)
        transfer.add_argument('--cmmn-c', type=float, default=1.0)
        transfer.add_argument('--cmmn-max-iter', type=int, default=1000)
        transfer.add_argument('--dry-run', action='store_true')
        transfer.add_argument('--preflight-only', action='store_true')
        transfer.add_argument('--interpretability-only', action='store_true')
        transfer.add_argument('--stop-on-error', type=int, choices=[0, 1], default=0)
        transfer.add_argument('--skip-completed', type=int, choices=[0, 1], default=1)
        transfer.add_argument('--eegnet-cuda-root', type=Path)
        transfer.add_argument(
            '--eegnet-python', type=Path,
            default=Path(
                os.environ.get(
                    'EEGNET_PYTHON',
                    str(DEFAULT_EEG_ROOT / 'envs/conda_envs/EEGNet_env/bin/python'),
                )
            ),
        )
        transfer.add_argument(
            '--process-name',
            default='auto',
            help=(
                'Process argv0 shown by ps. Use auto to show the environment '
                'name resolved from the selected Python executable.'
            ),
        )
        transfer.add_argument('--benchmark-root', type=Path, default=DEFAULT_BENCHMARK_ROOT)
        default_preprocess_root = (
            DEFAULT_PREPROCESS_ROOT
            if mission_type == 'dataset_transfer'
            else DEFAULT_CROSS_TIME_PREPROCESS_ROOT
            if mission_type == 'cross_time'
            else DEFAULT_CROSS_MODAL_PREPROCESS_ROOT
            if mission_type in {'eeg_ieeg_transfer', 'eeg_ieeg_localization', 'ieeg_indomain'}
            else DEFAULT_CROSS_MODAL_PREPROCESS_ROOT
        )
        default_result_root = (
            DEFAULT_RESULT_ROOT
            if mission_type == 'dataset_transfer'
            else DEFAULT_CROSS_TIME_RESULT_ROOT
            if mission_type == 'cross_time'
            else DEFAULT_CROSS_MODAL_RESULT_ROOT
            if mission_type == 'eeg_ieeg_transfer'
            else DEFAULT_DATA_ROOT / 'Results_ieeg_indomain'
            if mission_type == 'ieeg_indomain'
            else DEFAULT_LOCALIZATION_RESULT_ROOT
            )
        transfer.add_argument(
            '--preprocess-root', type=Path, default=default_preprocess_root
        )
        transfer.add_argument('--result-root', type=Path, default=default_result_root)
        transfer.add_argument(
            '--new-baseline-result-root',
            type=Path,
            default=DEFAULT_NEW_BASELINE_RESULT_ROOT,
        )
        transfer.add_argument(
            '--cross-modal-result-root',
            type=Path,
            default=DEFAULT_LOCALIZATION_REFERENCE_RESULT_ROOT,
            help='Completed EEG-iEEG detection result root used by localization as reference.',
        )
        transfer.add_argument(
            '--localization-reference-window-seconds',
            type=float,
            default=None,
            help='Window length of completed detection checkpoints used by localization.',
        )
    return parser


def run_parsed_args(args: argparse.Namespace) -> None:
    if not args.directions and (args.source_dataset is None or args.target_dataset is None):
        raise ValueError('Provide source and target datasets or use directions')
    if args.directions and (args.source_dataset is not None or args.target_dataset is not None):
        raise ValueError('Use either directions or the single source and target arguments')
    specs, skipped = build_specs(args)
    if not specs:
        if skipped:
            reason_text = '; '.join(skipped)
            raise ValueError(
                f'No applicable missions were selected: {reason_text}'
            )
        raise ValueError('No applicable missions were selected')
    print_mission_table(specs)
    for message in skipped:
        print(f'SKIPPED: {message}')
    environment = os.environ.copy()
    environment.update({
        'PYTHONHASHSEED': str(args.seed),
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'DATA_AUGMENTATION': '0',
        'DROPOUT': '0',
        'PDFORMER_ATTN_DROPOUT': '0',
        'PDFORMER_DROP_PATH': '0',
        'PYTHONFAULTHANDLER': '1',
        'PYTHONUNBUFFERED': '1',
    })
    if args.preflight_only:
        run_preflight(args, specs, environment)
        return
    failures = []
    for index, spec in enumerate(specs, start=1):
        command = build_command(args, spec)
        display_command, executable = anonymized_process_invocation(
            command, args.process_name, spec
        )
        if args.skip_completed:
            completed, completion_reason = completed_result_status(args, spec)
            if completed:
                print(f'SKIPPED_COMPLETED index={index} output={spec.output_dir}')
                continue
            print(
                f'RUN_REQUIRED index={index} reason={completion_reason} '
                f'output={spec.output_dir}'
            )
        print(f'Launching mission {index}/{len(specs)}: {spec.output_dir}')
        if args.dry_run:
            print(shlex.join(display_command))
        else:
            mission_environment = environment_for_spec(environment, args, spec)
            returncode = run_mission_process(
                display_command,
                executable,
                spec,
                mission_environment,
            )
            if returncode != 0:
                failures.append((index, str(spec.output_dir), returncode))
                print(
                    f'MISSION_FAILED index={index} exit_status={returncode} '
                    f'output={spec.output_dir}'
                )
                if args.stop_on_error:
                    break
    if failures:
        print('|Index|Exit status|Output|')
        print('|--:|--:|:--|')
        for index, output, returncode in failures:
            print(f'|{index}|{returncode}|{output}|')
        raise SystemExit(1)


def main() -> None:
    run_parsed_args(build_parser().parse_args())


if __name__ == '__main__':
    main()
