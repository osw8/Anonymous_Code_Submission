
# Defines cross-modal transfer and localization orchestration.
from __future__ import annotations




import hashlib
import json
import os
import zipfile
from dataclasses import asdict, dataclass
from xml.etree import ElementTree as ET


INTERPRETABILITY_TASKS = ('I1', 'I2', 'I3', 'I4')


@dataclass(frozen=True)
class InterpretabilityContract:
    tasks: tuple[str, ...] = INTERPRETABILITY_TASKS
    neural_attribution: str = 'input_x_gradient_positive_pre_sigmoid_logit'
    svm_attribution: str = 'feature_permutation_importance_separate_analysis'
    xai_sampling_seed: int = 2026
    maximum_clips_per_patient_per_class: int = 32
    patient_is_statistical_unit: bool = True
    patient_bootstrap_resamples: int = 1000
    fixed_cohort_across_models: bool = True
    fixed_cohort_across_budgets: bool = True
    fixed_cohort_across_seeds: bool = True
    source_split: str = 'test'
    target_split: str = 'test'
    frequency_bands_hz: tuple[tuple[str, float, float], ...] = (
        ('delta', 0.5, 4.0),
        ('theta', 4.0, 8.0),
        ('alpha', 8.0, 13.0),
        ('beta', 13.0, 30.0),
        ('low_gamma', 30.0, 45.0),
        ('mid_gamma', 45.0, 55.0),
        ('high_gamma', 65.0, 95.0),
    )
    i1_endpoint: str = 'epilepsy_ieeg_soz_contact_alignment'
    i2_endpoint: str = 'thalamocortical_preictal_recruitment_trajectory'
    i3_endpoint: str = 'budget_performance_and_clinical_evidence_recovery'
    clinical_labels_used_for_training: bool = False
    native_electrode_identity_required: bool = True
    electrode_coordinates_are_auxiliary: bool = True
    i4_enabled: bool = True

    def __post_init__(self) -> None:
        unsupported = set(self.tasks) - set(INTERPRETABILITY_TASKS)
        if unsupported:
            raise ValueError(f'Unsupported interpretability tasks: {sorted(unsupported)}')
        if self.maximum_clips_per_patient_per_class <= 0:
            raise ValueError('maximum_clips_per_patient_per_class must be positive')
        if self.patient_bootstrap_resamples <= 0:
            raise ValueError('patient_bootstrap_resamples must be positive')
        if not self.i4_enabled:
            raise ValueError('I4 is required by the current benchmark contract')

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload['fingerprint'] = self.fingerprint
        return payload




import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_NAMES = {
    'BENDR': ('source/best.pt', 'target/best.pt'),
    'BIOT': ('source/best.ckpt', 'target/best.ckpt'),
    'CBraMod': ('source/best.pth', 'target/best.pth'),
    'CST': ('source/best.pt', 'target/best.pt'),
    'EEGNet': ('source/best.weights.h5', 'target/best.weights.h5'),
    'EvoBrain': ('source/best.pth.tar', 'target/best.pth.tar'),
    'EEGPT': ('source/best.pt', 'target/best.pt'),
    'LaBraM': ('source/best.pt', 'target/best.pt'),
    'STEEGFormer': ('source/best.pt', 'target/best.pt'),
    'SVM': ('best_model.joblib', 'best_model.joblib'),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunEligibility:
    eligible: bool
    run_root: Path
    model: str
    checkpoint: Path | None
    missing: tuple[str, ...]
    reasons: tuple[str, ...]
    checkpoint_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            'eligible': self.eligible,
            'run_root': str(self.run_root),
            'model': self.model,
            'checkpoint': None if self.checkpoint is None else str(self.checkpoint),
            'missing': list(self.missing),
            'reasons': list(self.reasons),
            'checkpoint_sha256': self.checkpoint_sha256,
        }


def audit_run_eligibility(run_root: str | Path, model: str) -> RunEligibility:
    root = Path(run_root)
    if model not in CHECKPOINT_NAMES:
        raise ValueError(f'Unknown model for XAI eligibility: {model}')
    required = ('metrics.json', 'predictions.csv', 'protocol.json', 'args.json')
    missing = [name for name in required if not (root / name).is_file()]
    budget_percent = None
    protocol_path = root / 'protocol.json'
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        budget_percent = float(protocol.get('budget_percent', 0.0))
    checkpoint_name = CHECKPOINT_NAMES[model][0 if budget_percent == 0.0 else 1]
    checkpoint = root / checkpoint_name
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        missing.append(checkpoint_name)
        checkpoint_value = None
        checkpoint_hash = None
    else:
        checkpoint_value = checkpoint
        checkpoint_hash = sha256_file(checkpoint)
    reasons = []
    if model == 'SVM':
        reasons.append('SVM uses separate feature permutation importance')
    return RunEligibility(
        eligible=not missing,
        run_root=root,
        model=model,
        checkpoint=checkpoint_value,
        missing=tuple(missing),
        reasons=tuple(reasons),
        checkpoint_sha256=checkpoint_hash,
    )




from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class AttributionBatch:
    logits: np.ndarray
    raw_attribution: np.ndarray
    normalized_attribution: np.ndarray
    channel_attribution: np.ndarray
    temporal_attribution: np.ndarray


def normalize_signed_attribution(values: Any, epsilon: float = 1e-12):
    denominator = values.abs().flatten(start_dim=1).sum(dim=1)
    shape = (values.shape[0],) + (1,) * (values.ndim - 1)
    return values / denominator.clamp_min(epsilon).reshape(shape)


class TorchInputGradientBackend:
    method = 'input_x_gradient_positive_pre_sigmoid_logit'

    def explain(
        self,
        model: Any,
        model_input: Any,
        forward_positive_logit: Callable[[Any, Any], Any],
    ) -> AttributionBatch:
        import torch

        model.eval()
        values = model_input.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = forward_positive_logit(model, values).reshape(-1)
        if logits.shape[0] != values.shape[0]:
            raise ValueError('Positive logit count does not match attribution batch size')
        gradient = torch.autograd.grad(
            logits.sum(), values, retain_graph=False, create_graph=False
        )[0]
        raw = values * gradient
        normalized = normalize_signed_attribution(raw)
        if raw.ndim < 3:
            raise ValueError('Attribution tensor must contain channel and time dimensions')
        channel = raw.abs().flatten(start_dim=2).sum(dim=2)
        temporal = raw.abs().sum(dim=1)
        return AttributionBatch(
            logits=logits.detach().cpu().numpy().astype(np.float32),
            raw_attribution=raw.detach().cpu().numpy().astype(np.float32),
            normalized_attribution=(
                normalized.detach().cpu().numpy().astype(np.float32)
            ),
            channel_attribution=channel.detach().cpu().numpy().astype(np.float32),
            temporal_attribution=temporal.detach().cpu().numpy().astype(np.float32),
        )


class TensorFlowInputGradientBackend:
    method = 'input_x_gradient_positive_pre_softmax_logit'

    def explain(
        self,
        model: Any,
        model_input: Any,
        forward_positive_logit: Callable[[Any, Any], Any],
    ) -> AttributionBatch:
        import tensorflow as tf

        values = tf.convert_to_tensor(model_input)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(values)
            logits = tf.reshape(forward_positive_logit(model, values), (-1,))
        gradient = tape.gradient(tf.reduce_sum(logits), values)
        if gradient is None:
            raise ValueError('TensorFlow attribution graph is disconnected from the input')
        raw = values * gradient
        axes = tuple(range(1, len(raw.shape)))
        denominator = tf.reduce_sum(tf.abs(raw), axis=axes, keepdims=True)
        normalized = raw / tf.maximum(denominator, tf.cast(1e-12, raw.dtype))
        if len(raw.shape) < 3:
            raise ValueError('Attribution tensor must contain channel and time dimensions')
        channel = tf.reduce_sum(tf.abs(raw), axis=tuple(range(2, len(raw.shape))))
        temporal = tf.reduce_sum(tf.abs(raw), axis=1)
        return AttributionBatch(
            logits=np.asarray(logits, dtype=np.float32),
            raw_attribution=np.asarray(raw, dtype=np.float32),
            normalized_attribution=np.asarray(normalized, dtype=np.float32),
            channel_attribution=np.asarray(channel, dtype=np.float32),
            temporal_attribution=np.asarray(temporal, dtype=np.float32),
        )


def svm_permutation_importance(
    estimator: Any,
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 2026,
    repeats: int = 20,
) -> dict[str, np.ndarray | float | int]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator,
        np.asarray(features),
        np.asarray(labels),
        scoring='roc_auc',
        n_repeats=int(repeats),
        random_state=int(seed),
        n_jobs=1,
    )
    return {
        'importance_mean': result.importances_mean.astype(np.float32),
        'importance_std': result.importances_std.astype(np.float32),
        'importance_repeats': result.importances.astype(np.float32),
        'seed': int(seed),
        'repeats': int(repeats),
    }




import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


COHORT_COLUMNS = (
    'dataset', 'task', 'split', 'patient_id', 'clip_id', 'label', 'relative_path'
)


def _stable_group_seed(base_seed: int, patient_id: str, label: int) -> int:
    digest = hashlib.sha256(
        f'{base_seed}|{patient_id}|{int(label)}'.encode('utf-8')
    ).digest()
    return int.from_bytes(digest[:8], 'little', signed=False)


@dataclass(frozen=True)
class XaiCohort:
    frame: pd.DataFrame
    sha256: str
    sampling_seed: int
    maximum_clips_per_patient_per_class: int

    def save(self, output_dir: str | Path) -> None:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(root / 'xai_cohort.csv', index=False)
        payload = {
            'sha256': self.sha256,
            'sampling_seed': self.sampling_seed,
            'maximum_clips_per_patient_per_class': (
                self.maximum_clips_per_patient_per_class
            ),
            'clip_count': int(len(self.frame)),
            'patient_count': int(self.frame['patient_id'].astype(str).nunique()),
            'class_counts': {
                str(key): int(value)
                for key, value in self.frame['label'].astype(int).value_counts().items()
            },
        }
        (root / 'xai_cohort.json').write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
        )


def build_fixed_xai_cohort(
    manifest: pd.DataFrame,
    dataset: str,
    task: str,
    split: str = 'test',
    seed: int = 2026,
    maximum_clips_per_patient_per_class: int = 32,
    require_both_classes_per_patient: bool = True,
) -> XaiCohort:
    required = {'split', 'patient_id', 'clip_id', 'label', 'relative_path'}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f'Manifest lacks XAI cohort columns: {sorted(missing)}')
    eligible = manifest.loc[manifest['split'].astype(str) == split].copy()
    if eligible.empty:
        raise ValueError(f'No clips are available for XAI split {split}')
    eligible['patient_id'] = eligible['patient_id'].astype(str)
    eligible['clip_id'] = eligible['clip_id'].astype(str)
    eligible['label'] = eligible['label'].astype(int)
    if require_both_classes_per_patient and set(eligible['label'].unique()) != {0, 1}:
        raise ValueError('The fixed XAI cohort requires both classes')
    parts = []
    for patient_id, patient_frame in eligible.groupby('patient_id', sort=True):
        by_label = {
            int(label): group.sort_values('clip_id', kind='mergesort').reset_index(drop=True)
            for label, group in patient_frame.groupby('label', sort=True)
        }
        if require_both_classes_per_patient:
            if set(by_label) != {0, 1}:
                continue
            quota = min(
                maximum_clips_per_patient_per_class,
                len(by_label[0]),
                len(by_label[1]),
            )
            selected_labels = (0, 1)
        else:
            if 1 not in by_label:
                continue
            quota = min(maximum_clips_per_patient_per_class, len(by_label[1]))
            selected_labels = (1,)
        if quota <= 0:
            continue
        for label in selected_labels:
            ordered = by_label[label]
            rng = np.random.default_rng(
                _stable_group_seed(seed, patient_id, int(label))
            )
            indices = np.arange(len(ordered))
            rng.shuffle(indices)
            parts.append(ordered.iloc[indices[:quota]].copy())
    if not parts:
        requirement = (
            'both classes' if require_both_classes_per_patient else 'positive clips'
        )
        raise ValueError(f'No patient has {requirement} for the fixed XAI cohort')
    selected = pd.concat(parts, ignore_index=True)
    selected['task'] = str(task)
    selected['dataset'] = str(dataset)
    selected = selected.sort_values(
        ['patient_id', 'label', 'clip_id'], kind='mergesort'
    ).reset_index(drop=True)
    selected = selected.loc[:, list(COHORT_COLUMNS)]
    csv_bytes = selected.to_csv(index=False, lineterminator='\n').encode('utf-8')
    return XaiCohort(
        frame=selected,
        sha256=hashlib.sha256(csv_bytes).hexdigest(),
        sampling_seed=int(seed),
        maximum_clips_per_patient_per_class=int(
            maximum_clips_per_patient_per_class
        ),
    )




import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def canonical_contact_name(value: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', str(value).upper())


def canonical_patient_id(value: str) -> str:
    compact = str(value).strip().lower()
    compact = re.sub(r'^sub-', '', compact)
    matched = re.fullmatch(r'pt0*(\d+)', compact)
    if matched is not None:
        return f'sub-pt{int(matched.group(1))}'
    return f'sub-{compact}'


CONTACT_RANGE = re.compile(r'^([A-Z]+)(\d+)-(\d+)$', flags=re.IGNORECASE)


def expand_soz_contact_expression(value: str) -> tuple[str, ...]:
    normalized = str(value).replace(':', ',').replace(';', ',')
    contacts: list[str] = []
    for token in normalized.split(','):
        compact = re.sub(r'\s+', '', token)
        if not compact:
            continue
        matched = CONTACT_RANGE.match(compact)
        if matched is None:
            contacts.append(canonical_contact_name(compact))
            continue
        prefix, first_text, last_text = matched.groups()
        first = int(first_text)
        last = int(last_text)
        step = 1 if last >= first else -1
        contacts.extend(
            canonical_contact_name(f'{prefix}{index}')
            for index in range(first, last + step, step)
        )
    return tuple(dict.fromkeys(contact for contact in contacts if contact))


def load_epilepsy_ieeg_soz_labels(
    workbook: str | Path,
) -> dict[str, tuple[str, ...]]:
    try:
        frame = pd.read_excel(workbook, usecols=['dataset_id', 'soz_contacts'])
    except ImportError:
        frame = _load_xlsx_table(workbook, ('dataset_id', 'soz_contacts'))
    output: dict[str, tuple[str, ...]] = {}
    for row in frame.dropna(subset=['dataset_id', 'soz_contacts']).itertuples(index=False):
        patient_id = canonical_patient_id(str(row.dataset_id))
        output[patient_id] = expand_soz_contact_expression(str(row.soz_contacts))
    return output


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read('xl/sharedStrings.xml')
    except KeyError:
        return []
    root = ET.fromstring(payload)
    namespace = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    values: list[str] = []
    for node in root.findall('.//a:si', namespace):
        text_parts = [text.text or '' for text in node.findall('.//a:t', namespace)]
        values.append(''.join(text_parts))
    return values


def _xlsx_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read('xl/workbook.xml'))
    namespace = {
        'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }
    sheet = workbook.find('.//a:sheets/a:sheet', namespace)
    if sheet is None:
        raise ValueError('XLSX workbook does not contain a visible sheet')
    rel_id = sheet.attrib.get(f'{{{namespace["r"]}}}id')
    if not rel_id:
        raise ValueError('XLSX workbook sheet lacks relationship id')
    relationships = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
    rel_namespace = {'a': 'http://schemas.openxmlformats.org/package/2006/relationships'}
    for relation in relationships.findall('.//a:Relationship', rel_namespace):
        if relation.attrib.get('Id') == rel_id:
            target = relation.attrib.get('Target')
            if not target:
                break
            if target.startswith('/'):
                return target.lstrip('/')
            return f'xl/{target}'
    raise ValueError('XLSX workbook sheet relationship could not be resolved')


def _load_xlsx_table(
    workbook: str | Path,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    with zipfile.ZipFile(workbook) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheet_path = _xlsx_sheet_path(archive)
        sheet = ET.fromstring(archive.read(sheet_path))
    namespace = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows: list[list[str | None]] = []
    for row in sheet.findall('.//a:sheetData/a:row', namespace):
        values: list[str | None] = []
        for cell in row.findall('a:c', namespace):
            ref = cell.attrib.get('r', 'A1')
            column_index = 0
            for char in ref:
                if char.isalpha():
                    column_index = column_index * 26 + (ord(char.upper()) - 64)
                else:
                    break
            column_index -= 1
            while len(values) <= column_index:
                values.append(None)
            cell_type = cell.attrib.get('t')
            value_node = cell.find('a:v', namespace)
            inline_node = cell.find('a:is/a:t', namespace)
            if cell_type == 's' and value_node is not None:
                index = int(value_node.text or '0')
                values[column_index] = shared_strings[index] if index < len(shared_strings) else None
            elif inline_node is not None:
                values[column_index] = inline_node.text
            elif value_node is not None:
                values[column_index] = value_node.text
        rows.append(values)
    if not rows:
        raise ValueError(f'XLSX workbook is empty: {workbook}')
    header = [str(value).strip() if value is not None else '' for value in rows[0]]
    index_map = {name: position for position, name in enumerate(header) if name}
    missing = [name for name in required_columns if name not in index_map]
    if missing:
        raise ValueError(f'XLSX workbook lacks columns: {missing}')
    selected_rows: list[dict[str, object]] = []
    for values in rows[1:]:
        row = {
            column: (
                values[index_map[column]]
                if index_map[column] < len(values)
                else None
            )
            for column in required_columns
        }
        selected_rows.append(row)
    return pd.DataFrame(selected_rows, columns=list(required_columns))


def load_hup_soz_labels(channels_tsv: str | Path) -> tuple[str, ...]:
    frame = pd.read_csv(channels_tsv, sep='\t')
    required = {'name', 'status_description'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'HUP channels table lacks columns: {sorted(missing)}')
    descriptions = frame['status_description'].fillna('').astype(str)
    selected = frame.loc[
        descriptions.str.contains(r'(^|,)\s*soz\s*(,|$)', case=False, regex=True),
        'name',
    ]
    return tuple(canonical_contact_name(value) for value in selected)


def load_thalamocortical_soz_labels(
    channel_groups_json: str | Path,
) -> tuple[str, ...]:
    payload = json.loads(Path(channel_groups_json).read_text(encoding='utf-8'))
    onset = payload.get('onset', [])
    labeled = [
        key for key, value in payload.items()
        if isinstance(value, str) and value.strip().upper() == 'SOZ'
    ]
    return tuple(
        dict.fromkeys(
            canonical_contact_name(value)
            for value in [*onset, *labeled]
            if canonical_contact_name(value)
        )
    )


@dataclass(frozen=True)
class PatientSozMetrics:
    patient_id: str
    contact_count: int
    soz_contact_count: int
    auroc: float
    average_precision: float
    recall_at_k: float
    enrichment: float


@dataclass(frozen=True)
class PatientLocalizationMetrics:
    patient_id: str
    contact_count: int
    soz_contact_count: int
    auroc: float
    average_precision: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    ndcg_at_1: float
    ndcg_at_3: float
    ndcg_at_5: float
    ndcg_at_10: float
    enrichment: float


def compute_patient_soz_metrics(
    patient_id: str,
    contact_names: Iterable[str],
    attribution: np.ndarray,
    soz_contacts: Iterable[str],
    epsilon: float = 1e-12,
) -> PatientSozMetrics:
    names = [canonical_contact_name(value) for value in contact_names]
    scores = np.asarray(attribution, dtype=np.float64).reshape(-1)
    if len(names) != len(scores):
        raise ValueError('Contact names and attribution scores have different lengths')
    soz = {canonical_contact_name(value) for value in soz_contacts}
    labels = np.asarray([int(name in soz) for name in names], dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError('SOZ metrics require both SOZ and non-SOZ contacts')
    magnitude = np.abs(scores)
    k = int(labels.sum())
    ranking = np.argsort(-magnitude, kind='mergesort')[:k]
    recall = float(labels[ranking].sum() / k)
    enrichment = float(
        magnitude[labels == 1].mean()
        / (magnitude[labels == 0].mean() + epsilon)
    )
    return PatientSozMetrics(
        patient_id=str(patient_id),
        contact_count=len(labels),
        soz_contact_count=k,
        auroc=float(roc_auc_score(labels, magnitude)),
        average_precision=float(average_precision_score(labels, magnitude)),
        recall_at_k=recall,
        enrichment=enrichment,
    )


def _ranking_metrics_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError('Localization labels and scores have different lengths')
    if labels.size == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-scores, kind='mergesort')
    topk = order[: min(int(k), labels.size)]
    hit = float(labels[topk].sum() > 0)
    recall = float(labels[topk].sum() / labels.sum()) if labels.sum() > 0 else 0.0
    gains = labels[order][: min(int(k), labels.size)].astype(np.float64)
    if gains.size == 0:
        return hit, recall, 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum((2.0 ** gains - 1.0) * discounts))
    ideal = np.sort(labels)[::-1][:gains.size].astype(np.float64)
    ideal_dcg = float(np.sum((2.0 ** ideal - 1.0) * discounts))
    ndcg = float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0
    return hit, recall, ndcg


def compute_patient_localization_metrics(
    patient_id: str,
    contact_names: Iterable[str],
    attribution: np.ndarray,
    soz_contacts: Iterable[str],
    epsilon: float = 1e-12,
) -> PatientLocalizationMetrics:
    names = [canonical_contact_name(value) for value in contact_names]
    scores = np.asarray(attribution, dtype=np.float64).reshape(-1)
    if len(names) != len(scores):
        raise ValueError('Contact names and attribution scores have different lengths')
    soz = {canonical_contact_name(value) for value in soz_contacts}
    labels = np.asarray([int(name in soz) for name in names], dtype=np.int64)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError('SOZ metrics require both SOZ and non-SOZ contacts')
    k = int(labels.sum())
    auc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    hit1, recall1, ndcg1 = _ranking_metrics_at_k(labels, scores, 1)
    hit3, recall3, ndcg3 = _ranking_metrics_at_k(labels, scores, 3)
    hit5, recall5, ndcg5 = _ranking_metrics_at_k(labels, scores, 5)
    hit10, recall10, ndcg10 = _ranking_metrics_at_k(labels, scores, 10)
    enrichment = float(
        scores[labels == 1].mean() / (scores[labels == 0].mean() + epsilon)
    )
    return PatientLocalizationMetrics(
        patient_id=str(patient_id),
        contact_count=len(labels),
        soz_contact_count=k,
        auroc=auc,
        average_precision=ap,
        hit_at_1=hit1,
        hit_at_3=hit3,
        hit_at_5=hit5,
        hit_at_10=hit10,
        recall_at_1=recall1,
        recall_at_3=recall3,
        recall_at_5=recall5,
        recall_at_10=recall10,
        ndcg_at_1=ndcg1,
        ndcg_at_3=ndcg3,
        ndcg_at_5=ndcg5,
        ndcg_at_10=ndcg10,
        enrichment=enrichment,
    )




import numpy as np
from scipy.signal import butter, sosfiltfilt



def deterministic_bandstop_before_adapter(
    signal: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    nyquist = float(sfreq) / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f'Bandstop range {low_hz:g} to {high_hz:g}Hz is invalid for {sfreq:g}Hz'
        )
    sos = butter(
        int(order), [low_hz / nyquist, high_hz / nyquist],
        btype='bandstop', output='sos',
    )
    filtered = sosfiltfilt(sos, values, axis=-1)
    return np.asarray(filtered, dtype=np.float32)


def default_frequency_bands() -> tuple[tuple[str, float, float], ...]:
    return InterpretabilityContract().frequency_bands_hz




import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd



CLINICAL_BOOTSTRAP_SEED = 2026
CLINICAL_BOOTSTRAP_RESAMPLES = 2000


def _patient_bootstrap_mean_ci(
    values: Any,
    seed: int = CLINICAL_BOOTSTRAP_SEED,
    resamples: int = CLINICAL_BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            'mean': None,
            'ci95_low': None,
            'ci95_high': None,
            'patient_count': 0,
            'resamples': int(resamples),
            'seed': int(seed),
        }
    generator = np.random.default_rng(int(seed))
    draws = generator.integers(0, array.size, size=(int(resamples), array.size))
    bootstrap = array[draws].mean(axis=1)
    return {
        'mean': float(array.mean()),
        'ci95_low': float(np.quantile(bootstrap, 0.025)),
        'ci95_high': float(np.quantile(bootstrap, 0.975)),
        'patient_count': int(array.size),
        'resamples': int(resamples),
        'seed': int(seed),
    }


def _native_channel_axis(layout: str) -> int:
    return 3 if layout == 'time_channel_patch' else 2


def _subset_loader(
    loader: Any,
    indices: list[int],
    batch_size: int | None = None,
) -> Any:
    from torch.utils.data import DataLoader, Subset

    return DataLoader(
        Subset(loader.dataset, indices),
        batch_size=loader.batch_size if batch_size is None else int(batch_size),
        shuffle=False,
        num_workers=loader.num_workers,
        pin_memory=getattr(loader, 'pin_memory', False),
        persistent_workers=loader.num_workers > 0,
        collate_fn=loader.collate_fn,
    )


def collect_native_contact_evidence(
    model: Any,
    loader: Any,
    device: Any,
    forward_clip: Callable[[Any, Any, Any], Any],
    dataset: str,
    task: str,
    maximum_per_patient_class: int = 32,
    seed: int = 2026,
    cohort_root: str | Path | None = None,
    evidence_batch_size: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch

    dataset_root = getattr(loader, 'dataset', None)
    if dataset_root is None:
        raise ValueError('Localization evidence loader lacks a dataset attribute')
    manifest = getattr(dataset_root, 'manifest', None)
    if manifest is None:
        nested_dataset = getattr(dataset_root, 'dataset', None)
        manifest = getattr(nested_dataset, 'manifest', None)
    if manifest is None:
        raise AttributeError('Localization evidence loader dataset does not expose a manifest')
    model.to(device).eval()
    cohort = build_fixed_xai_cohort(
        manifest,
        dataset=dataset,
        task=task,
        split='test',
        seed=seed,
        maximum_clips_per_patient_per_class=maximum_per_patient_class,
        require_both_classes_per_patient=not (
            dataset == 'thalamocortical_ieeg' and task == 'prediction'
        ),
    )
    if cohort_root is not None:
        shared_root = Path(cohort_root) / str(dataset) / str(task)
        existing_csv = shared_root / 'xai_cohort.csv'
        existing_json = shared_root / 'xai_cohort.json'
        if existing_csv.is_file() and existing_json.is_file():
            existing_metadata = json.loads(existing_json.read_text(encoding='utf-8'))
            if existing_metadata.get('sha256') != cohort.sha256:
                shared_root.mkdir(parents=True, exist_ok=True)
                (shared_root / 'xai_cohort_conflict.json').write_text(
                    json.dumps(
                        {
                            'status': 'rebuilt_shared_xai_cohort',
                            'reason': 'existing_shared_cohort_differs_from_current_contract',
                            'existing_sha256': existing_metadata.get('sha256'),
                            'current_sha256': cohort.sha256,
                            'existing_sampling_seed': existing_metadata.get('sampling_seed'),
                            'current_sampling_seed': cohort.sampling_seed,
                            'policy': 'continue_with_current_fixed_budget_seed_cohort',
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding='utf-8',
                )
                cohort.save(shared_root)
        else:
            cohort.save(shared_root)
    selected_ids = set(cohort.frame['clip_id'].astype(str))
    indices = [
        index for index, clip_id in enumerate(manifest['clip_id'].astype(str))
        if clip_id in selected_ids
    ]
    selected_loader = _subset_loader(loader, indices, evidence_batch_size)
    rows: list[dict[str, Any]] = []
    model.eval()
    for batch in selected_loader:
        values = batch['eeg'].to(device).detach().requires_grad_(True)
        mask = batch['channel_mask'].to(device)
        model.zero_grad(set_to_none=True)
        deterministic_enabled = torch.are_deterministic_algorithms_enabled()
        if deterministic_enabled:
            torch.use_deterministic_algorithms(False)
        try:
            logits = forward_clip(model, values, mask).reshape(-1)
            gradient = torch.autograd.grad(logits.sum(), values)[0]
        finally:
            if deterministic_enabled:
                torch.use_deterministic_algorithms(True)
        evidence = (values * gradient).abs()
        channel_axis = _native_channel_axis(str(loader.dataset.spec.layout))
        reduce_axes = tuple(
            axis for axis in range(1, evidence.ndim) if axis != channel_axis
        )
        contact_scores = evidence.sum(dim=reduce_axes)
        contact_scores = contact_scores / contact_scores.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        scores = contact_scores.detach().cpu().numpy()
        positions = batch['channel_positions'].detach().cpu().numpy()
        for sample_index, names in enumerate(batch['channel_names']):
            channel_types = batch['channel_types'][sample_index]
            for channel_index, name in enumerate(names):
                position = positions[sample_index, channel_index]
                rows.append({
                    'dataset': str(dataset),
                    'task': str(task),
                    'patient_id': str(batch['patient_id'][sample_index]),
                    'clip_id': str(batch['clip_id'][sample_index]),
                    'label': int(batch['label'][sample_index]),
                    'source_relative_path': str(batch['source_relative_path'][sample_index]),
                    'clip_start_seconds': float(batch['clip_start_seconds'][sample_index]),
                    'clip_end_seconds': float(batch['clip_end_seconds'][sample_index]),
                    'seizure_intervals_json': str(batch['seizure_intervals_json'][sample_index]),
                    'contact_name': str(name),
                    'contact_key': canonical_contact_name(name),
                    'channel_type': str(channel_types[channel_index]).upper(),
                    'evidence': float(scores[sample_index, channel_index]),
                    'x': float(position[0]),
                    'y': float(position[1]),
                    'z': float(position[2]),
                    'coordinate_available': bool(np.isfinite(position).all()),
                })
    return pd.DataFrame(rows), {
        'cohort_sha256': cohort.sha256,
        'clip_count': int(len(indices)),
        'patient_count': int(cohort.frame['patient_id'].nunique()),
        'maximum_clips_per_patient_per_class': int(maximum_per_patient_class),
        'sampling_seed': int(seed),
        'evidence_batch_size': (
            int(loader.batch_size)
            if evidence_batch_size is None
            else int(evidence_batch_size)
        ),
    }


def run_i1_epilepsy_soz(
    evidence: pd.DataFrame,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    workbook = Path(raw_root) / 'sourcedata' / 'clinical_data_summary.xlsx'
    labels = load_epilepsy_ieeg_soz_labels(workbook)
    averaged = evidence.groupby(
        ['patient_id', 'label', 'contact_key'], as_index=False
    )['evidence'].mean()
    contrast_rows = []
    metric_rows = []
    for patient_id, patient in averaged.groupby('patient_id', sort=True):
        pivot = patient.pivot_table(
            index='contact_key', columns='label', values='evidence', aggfunc='mean'
        ).dropna(subset=[0, 1])
        if pivot.empty or patient_id not in labels:
            continue
        contrast = (pivot[1] - pivot[0]).clip(lower=0.0)
        for contact, value in contrast.items():
            contrast_rows.append({
                'patient_id': str(patient_id),
                'contact_key': str(contact),
                'positive_minus_negative_evidence': float(value),
                'is_soz': int(contact in set(labels[patient_id])),
            })
        try:
            metric_rows.append(asdict(compute_patient_soz_metrics(
                str(patient_id), pivot.index, contrast.to_numpy(), labels[patient_id]
            )))
        except ValueError:
            continue
    metrics = pd.DataFrame(metric_rows)
    contacts = pd.DataFrame(contrast_rows)
    root = Path(output_dir) / 'interpretability' / 'I1_epilepsy_soz'
    root.mkdir(parents=True, exist_ok=True)
    contacts.to_csv(root / 'contact_evidence.csv', index=False)
    metrics.to_csv(root / 'patient_metrics.csv', index=False)
    metric_statistics = {
        metric: _patient_bootstrap_mean_ci(metrics[metric])
        for metric in ('auroc', 'average_precision', 'recall_at_k', 'enrichment')
        if metric in metrics
    }
    (root / 'patient_bootstrap_statistics.json').write_text(
        json.dumps(metric_statistics, indent=2), encoding='utf-8'
    )
    summary = {
        'status': 'complete' if not metrics.empty else 'insufficient_eligible_patients',
        'patient_count': int(len(metrics)),
        'patient_macro_auroc': float(metrics['auroc'].mean()) if not metrics.empty else None,
        'patient_macro_average_precision': float(metrics['average_precision'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_k': float(metrics['recall_at_k'].mean()) if not metrics.empty else None,
        'patient_macro_enrichment': float(metrics['enrichment'].mean()) if not metrics.empty else None,
        'patient_bootstrap_statistics': metric_statistics,
        'clinical_labels_used_for_training': False,
        'attribution_space': 'standardized_native_electrode_time',
    }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_i4_epilepsy_localization(
    evidence: pd.DataFrame,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    workbook = Path(raw_root) / 'sourcedata' / 'clinical_data_summary.xlsx'
    labels = load_epilepsy_ieeg_soz_labels(workbook)
    positive = evidence.loc[evidence['label'].astype(int) == 1].copy()
    if positive.empty:
        summary = {
            'status': 'insufficient_positive_clips',
            'patient_count': 0,
            'clinical_labels_used_for_training': False,
            'attribution_space': 'standardized_native_electrode_time',
        }
        root = Path(output_dir) / 'interpretability' / 'I4_epilepsy_localization'
        root.mkdir(parents=True, exist_ok=True)
        (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    averaged = positive.groupby(
        ['patient_id', 'contact_key'], as_index=False
    )['evidence'].mean()
    contrast = evidence.groupby(
        ['patient_id', 'contact_key', 'label'], as_index=False
    )['evidence'].mean()
    contrast_pivot = contrast.pivot_table(
        index=['patient_id', 'contact_key'], columns='label', values='evidence', aggfunc='mean'
    )
    metric_rows = []
    contact_rows = []
    for patient_id, patient in averaged.groupby('patient_id', sort=True):
        if patient_id not in labels:
            continue
        patient_scores = patient.set_index('contact_key')['evidence'].sort_values(ascending=False)
        patient_contacts = patient_scores.index.tolist()
        patient_soz = labels[patient_id]
        patient_soz_set = set(patient_soz)
        if not patient_contacts:
            continue
        scores = patient_scores.to_numpy(dtype=np.float64)
        label_vector = np.asarray(
            [int(contact in patient_soz_set) for contact in patient_contacts],
            dtype=np.int64,
        )
        if label_vector.sum() == 0 or label_vector.sum() == len(label_vector):
            continue
        contrast_scores = []
        for contact in patient_contacts:
            key = (patient_id, contact)
            row = contrast_pivot.loc[key] if key in contrast_pivot.index else None
            if row is None or 1 not in row.index or 0 not in row.index:
                contrast_scores.append(np.nan)
            else:
                contrast_scores.append(float(row[1] - row[0]))
        ranking = np.argsort(-scores, kind='mergesort')
        for rank, index in enumerate(ranking, start=1):
            contact_rows.append({
                'patient_id': str(patient_id),
                'contact_key': str(patient_contacts[index]),
                'ictal_mean_evidence': float(scores[index]),
                'positive_minus_negative_evidence': (
                    float(contrast_scores[index]) if np.isfinite(contrast_scores[index]) else None
                ),
                'is_soz': int(patient_contacts[index] in patient_soz_set),
                'rank': int(rank),
                'contact_count': int(len(patient_contacts)),
            })
        try:
            metric_rows.append(asdict(compute_patient_localization_metrics(
                str(patient_id), patient_contacts, scores, patient_soz
            )))
        except ValueError:
            continue
    metrics = pd.DataFrame(metric_rows)
    contacts = pd.DataFrame(contact_rows)
    root = Path(output_dir) / 'interpretability' / 'I4_epilepsy_localization'
    root.mkdir(parents=True, exist_ok=True)
    contacts.to_csv(root / 'contact_evidence.csv', index=False)
    metrics.to_csv(root / 'patient_metrics.csv', index=False)
    metric_columns = [
        column for column in metrics.columns
        if column not in {'patient_id'}
        and np.issubdtype(metrics[column].dtype, np.number)
    ]
    metric_statistics = {
        metric: _patient_bootstrap_mean_ci(metrics[metric])
        for metric in metric_columns
        if metric in metrics
    }
    (root / 'patient_bootstrap_statistics.json').write_text(
        json.dumps(metric_statistics, indent=2), encoding='utf-8'
    )
    summary = {
        'status': 'complete' if not metrics.empty else 'insufficient_eligible_patients',
        'patient_count': int(len(metrics)),
        'patient_macro_auroc': float(metrics['auroc'].mean()) if not metrics.empty else None,
        'patient_macro_average_precision': float(metrics['average_precision'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_1': float(metrics['hit_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_3': float(metrics['hit_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_5': float(metrics['hit_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_10': float(metrics['hit_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_1': float(metrics['recall_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_3': float(metrics['recall_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_5': float(metrics['recall_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_10': float(metrics['recall_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_1': float(metrics['ndcg_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_3': float(metrics['ndcg_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_5': float(metrics['ndcg_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_10': float(metrics['ndcg_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_enrichment': float(metrics['enrichment'].mean()) if not metrics.empty else None,
        'patient_bootstrap_statistics': metric_statistics,
        'clinical_labels_used_for_training': False,
        'attribution_space': 'standardized_native_electrode_time',
        'localization_metric_family': 'patient_level_contact_ranking',
    }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_epilepsy_localization_task(
    evidence: pd.DataFrame,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    workbook = Path(raw_root) / 'sourcedata' / 'clinical_data_summary.xlsx'
    labels = load_epilepsy_ieeg_soz_labels(workbook)
    positive = evidence.loc[evidence['label'].astype(int) == 1].copy()
    root = Path(output_dir) / 'localization'
    root.mkdir(parents=True, exist_ok=True)
    if positive.empty:
        summary = {
            'status': 'insufficient_positive_clips',
            'patient_count': 0,
            'clinical_labels_used_for_training': False,
            'attribution_space': 'standardized_native_electrode_time',
            'task_family': 'eeg_ieeg_localization',
        }
        (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    averaged = positive.groupby(
        ['patient_id', 'contact_key'], as_index=False
    )['evidence'].mean()
    contrast = evidence.groupby(
        ['patient_id', 'contact_key', 'label'], as_index=False
    )['evidence'].mean()
    contrast_pivot = contrast.pivot_table(
        index=['patient_id', 'contact_key'], columns='label', values='evidence', aggfunc='mean'
    )
    metric_rows = []
    contact_rows = []
    for patient_id, patient in averaged.groupby('patient_id', sort=True):
        if patient_id not in labels:
            continue
        patient_scores = patient.set_index('contact_key')['evidence'].sort_values(ascending=False)
        patient_contacts = patient_scores.index.tolist()
        patient_soz = labels[patient_id]
        patient_soz_set = set(patient_soz)
        if not patient_contacts:
            continue
        scores = patient_scores.to_numpy(dtype=np.float64)
        label_vector = np.asarray(
            [int(contact in patient_soz_set) for contact in patient_contacts],
            dtype=np.int64,
        )
        if label_vector.sum() == 0 or label_vector.sum() == len(label_vector):
            continue
        contrast_scores = []
        for contact in patient_contacts:
            key = (patient_id, contact)
            row = contrast_pivot.loc[key] if key in contrast_pivot.index else None
            if row is None or 1 not in row.index or 0 not in row.index:
                contrast_scores.append(np.nan)
            else:
                contrast_scores.append(float(row[1] - row[0]))
        ranking = np.argsort(-scores, kind='mergesort')
        for rank, index in enumerate(ranking, start=1):
            contact_rows.append({
                'patient_id': str(patient_id),
                'contact_key': str(patient_contacts[index]),
                'ictal_mean_evidence': float(scores[index]),
                'positive_minus_negative_evidence': (
                    float(contrast_scores[index]) if np.isfinite(contrast_scores[index]) else None
                ),
                'is_soz': int(patient_contacts[index] in patient_soz_set),
                'rank': int(rank),
                'contact_count': int(len(patient_contacts)),
            })
        try:
            metric_rows.append(asdict(compute_patient_localization_metrics(
                str(patient_id), patient_contacts, scores, patient_soz
            )))
        except ValueError:
            continue
    metrics = pd.DataFrame(metric_rows)
    contacts = pd.DataFrame(contact_rows)
    contacts.to_csv(root / 'contact_evidence.csv', index=False)
    metrics.to_csv(root / 'patient_metrics.csv', index=False)
    metric_columns = [
        column for column in metrics.columns
        if column not in {'patient_id'}
        and np.issubdtype(metrics[column].dtype, np.number)
    ]
    metric_statistics = {
        metric: _patient_bootstrap_mean_ci(metrics[metric])
        for metric in metric_columns
        if metric in metrics
    }
    (root / 'patient_bootstrap_statistics.json').write_text(
        json.dumps(metric_statistics, indent=2), encoding='utf-8'
    )
    summary = {
        'status': 'complete' if not metrics.empty else 'insufficient_eligible_patients',
        'patient_count': int(len(metrics)),
        'patient_macro_auroc': float(metrics['auroc'].mean()) if not metrics.empty else None,
        'patient_macro_average_precision': float(metrics['average_precision'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_1': float(metrics['hit_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_3': float(metrics['hit_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_5': float(metrics['hit_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_hit_at_10': float(metrics['hit_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_1': float(metrics['recall_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_3': float(metrics['recall_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_5': float(metrics['recall_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_recall_at_10': float(metrics['recall_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_1': float(metrics['ndcg_at_1'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_3': float(metrics['ndcg_at_3'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_5': float(metrics['ndcg_at_5'].mean()) if not metrics.empty else None,
        'patient_macro_ndcg_at_10': float(metrics['ndcg_at_10'].mean()) if not metrics.empty else None,
        'patient_macro_enrichment': float(metrics['enrichment'].mean()) if not metrics.empty else None,
        'patient_bootstrap_statistics': metric_statistics,
        'clinical_labels_used_for_training': False,
        'attribution_space': 'standardized_native_electrode_time',
        'localization_metric_family': 'patient_level_contact_ranking',
        'task_family': 'eeg_ieeg_localization',
    }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def _nearest_future_onset(interval_json: str, clip_end: float) -> float | None:
    try:
        payload = json.loads(interval_json)
    except json.JSONDecodeError:
        return None
    onsets: list[float] = []
    if isinstance(payload, list):
        for value in payload:
            if isinstance(value, (list, tuple)) and value:
                onsets.append(float(value[0]))
            elif isinstance(value, dict):
                for key in ('onset', 'start', 'start_seconds'):
                    if key in value:
                        onsets.append(float(value[key]))
                        break
    future = [value for value in onsets if value >= clip_end]
    return min(future) if future else None


def _channel_group_path(raw_root: Path, relative_path: str) -> Path | None:
    patient = re.search(r'(sub-[^/]+)', relative_path)
    session = re.search(r'(ses-[^/]+)', relative_path)
    if patient is None or session is None:
        return None
    candidates = list((raw_root / 'derivatives' / 'channel_groups' / patient.group(1) / session.group(1)).glob('*.json'))
    return candidates[0] if len(candidates) == 1 else None


def _clinical_group(payload: dict[str, Any], contact_key: str) -> str:
    onset = {canonical_contact_name(value) for value in payload.get('onset', [])}
    propagation = {canonical_contact_name(value) for value in payload.get('prop', [])}
    if contact_key in onset:
        return 'SOZ_onset'
    if contact_key in propagation:
        return 'propagation'
    label = ''
    for key, value in payload.items():
        if canonical_contact_name(key) == contact_key and isinstance(value, str):
            label = value.upper()
            break
    side = 'ipsilateral' if label.startswith('IP ') else 'contralateral' if label.startswith('CO ') else 'unknown'
    is_thalamus = any(token in label.split() for token in ('AN', 'CM', 'PUL', 'MDM', 'MD'))
    tissue = 'thalamus' if is_thalamus else 'cortex'
    return f'{side}_{tissue}'


def run_i2_thalamocortical_trajectory(
    evidence: pd.DataFrame,
    raw_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    raw_root = Path(raw_root)
    rows = []
    cache: dict[str, dict[str, Any]] = {}
    for row in evidence.loc[evidence['label'] == 1].itertuples(index=False):
        onset = _nearest_future_onset(row.seizure_intervals_json, row.clip_end_seconds)
        if onset is None:
            continue
        minutes = (onset - row.clip_end_seconds) / 60.0
        if 5.0 <= minutes < 15.0:
            phase = 'proximal_5_15min'
        elif 15.0 <= minutes < 25.0:
            phase = 'middle_15_25min'
        elif 25.0 <= minutes <= 35.0:
            phase = 'distal_25_35min'
        else:
            continue
        path = _channel_group_path(raw_root, row.source_relative_path)
        if path is None:
            continue
        payload = cache.setdefault(str(path), json.loads(path.read_text(encoding='utf-8')))
        rows.append({
            'patient_id': row.patient_id,
            'clip_id': row.clip_id,
            'contact_key': row.contact_key,
            'phase': phase,
            'minutes_to_onset': float(minutes),
            'clinical_group': _clinical_group(payload, row.contact_key),
            'evidence': float(row.evidence),
        })
    trajectory = pd.DataFrame(rows)
    root = Path(output_dir) / 'interpretability' / 'I2_thalamocortical_trajectory'
    root.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(root / 'contact_trajectory.csv', index=False)
    if trajectory.empty:
        summary = {'status': 'insufficient_matched_annotations', 'patient_count': 0}
    else:
        patient_group = trajectory.groupby(
            ['patient_id', 'clinical_group', 'phase'], as_index=False
        )['evidence'].mean()
        patient_group.to_csv(root / 'patient_group_trajectory.csv', index=False)
        onset_rows = patient_group[patient_group['clinical_group'] == 'SOZ_onset']
        phase_means = onset_rows.groupby('phase')['evidence'].mean().to_dict()
        proximal = phase_means.get('proximal_5_15min')
        distal = phase_means.get('distal_25_35min')
        paired = onset_rows.pivot_table(
            index='patient_id', columns='phase', values='evidence', aggfunc='mean'
        )
        required_phases = {'proximal_5_15min', 'distal_25_35min'}
        if required_phases.issubset(paired.columns):
            paired = paired.dropna(subset=sorted(required_phases)).copy()
            paired['proximal_minus_distal'] = (
                paired['proximal_5_15min'] - paired['distal_25_35min']
            )
        else:
            paired = pd.DataFrame(columns=['proximal_minus_distal'])
        paired.reset_index().to_csv(root / 'patient_phase_contrasts.csv', index=False)
        paired_statistics = _patient_bootstrap_mean_ci(
            paired['proximal_minus_distal']
        )
        summary = {
            'status': 'complete',
            'patient_count': int(trajectory['patient_id'].nunique()),
            'clip_count': int(trajectory['clip_id'].nunique()),
            'contact_observation_count': int(len(trajectory)),
            'soz_onset_phase_mean_evidence': {key: float(value) for key, value in phase_means.items()},
            'proximal_minus_distal_soz_onset_evidence': (
                float(proximal - distal) if proximal is not None and distal is not None else None
            ),
            'paired_patient_proximal_minus_distal': paired_statistics,
            'clinical_labels_used_for_training': False,
            'attribution_space': 'standardized_native_electrode_time',
        }
    (root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_clinical_interpretability(
    spec: Any,
    model: Any,
    loader: Any,
    device: Any,
    forward_clip: Callable[[Any, Any, Any], Any],
    target_root: str | Path,
) -> dict[str, Any]:
    contract_path = Path(target_root) / 'dataset_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    raw_root = contract.get('source_root')
    if not raw_root:
        return {'status': 'skipped_missing_source_root_contract'}
    maximum = max(1, min(32, int(spec.interpretability_max_clips)))
    evidence, cohort = collect_native_contact_evidence(
        model, loader, device, forward_clip,
        dataset=spec.target_dataset,
        task=spec.task,
        maximum_per_patient_class=maximum,
        cohort_root=(
            Path(spec.result_root) / spec.mission_type / 'xai_cohorts'
            / spec.window_name
        ),
    )
    root = Path(spec.output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
    (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
    if spec.mission_type == 'eeg_ieeg_localization':
        result = {
            'L1': run_epilepsy_localization_task(evidence, raw_root, spec.output_dir),
            'cohort': cohort,
        }
        return result
    if spec.target_dataset == 'epilepsy_ieeg':
        result = {
            'I1': run_i1_epilepsy_soz(evidence, raw_root, spec.output_dir),
        }
        result['cohort'] = cohort
        return result
    if spec.target_dataset == 'thalamocortical_ieeg':
        return {'I2': run_i2_thalamocortical_trajectory(evidence, raw_root, spec.output_dir), 'cohort': cohort}
    return {'status': 'not_applicable_to_target', 'cohort': cohort}


def refresh_i3_budget_recovery(spec: Any) -> dict[str, Any]:
    direction_root = spec.output_dir.parent.parent
    rows = []
    for budget in (0, 25, 100):
        budget_name = f'{spec.model}_0_budget' if budget == 0 else f'{spec.model}_{budget}%_budget'
        run_root = direction_root / budget_name / f'seed_{spec.seed}'
        metrics_path = run_root / 'metrics.json'
        clinical_paths = list((run_root / 'interpretability').glob('I*/summary.json'))
        if not metrics_path.is_file() or not clinical_paths:
            continue
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        clinical = json.loads(clinical_paths[0].read_text(encoding='utf-8'))
        clip_metrics = metrics.get('clip_level', metrics)
        clinical_value = clinical.get('patient_macro_average_precision')
        if clinical_value is None:
            clinical_value = clinical.get('proximal_minus_distal_soz_onset_evidence')
        rows.append({
            'budget_percent': budget,
            'auroc': clip_metrics.get('auroc'),
            'auprc': clip_metrics.get('auprc'),
            'f1': clip_metrics.get('f1'),
            'clinical_evidence': clinical_value,
        })
    root = direction_root / 'interpretability_summary' / spec.model / f'seed_{spec.seed}'
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values('budget_percent') if rows else pd.DataFrame()
    if set(frame.get('budget_percent', [])) == {0, 25, 100}:
        indexed = frame.set_index('budget_percent')
        for metric in ('auroc', 'auprc', 'f1', 'clinical_evidence'):
            zero = indexed.at[0, metric]
            quarter = indexed.at[25, metric]
            full = indexed.at[100, metric]
            denominator = full - zero
            recovery = (
                0.0 if pd.isna(denominator) or abs(denominator) <= 1e-12
                else float((quarter - zero) / denominator)
            )
            frame.loc[frame['budget_percent'] == 25, f'{metric}_recovery_fraction'] = recovery
    frame.to_csv(root / 'I3_budget_recovery.csv', index=False)
    payload = {'status': 'complete' if set(frame.get('budget_percent', [])) == {0, 25, 100} else 'partial', 'available_budgets': frame.get('budget_percent', pd.Series(dtype=int)).tolist()}
    (root / 'I3_budget_recovery.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload




import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def aggregate_clip_attribution_to_patient(
    frame: pd.DataFrame,
    attribution: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {'patient_id', 'clip_id'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Attribution metadata lacks columns: {sorted(missing)}')
    values = np.asarray(attribution, dtype=np.float64)
    if values.shape[0] != len(frame):
        raise ValueError('Attribution rows do not match clip metadata')
    rows = []
    parts = []
    for patient_id, indices in frame.groupby('patient_id', sort=True).indices.items():
        selected = np.asarray(indices, dtype=np.int64)
        parts.append(values[selected].mean(axis=0))
        rows.append({
            'patient_id': str(patient_id),
            'clip_count': int(len(selected)),
        })
    return pd.DataFrame(rows), np.stack(parts).astype(np.float32)


def cosine_similarity_to_reference(
    current: np.ndarray,
    reference: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError('Current and reference evidence shapes must match')
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return (numerator / np.maximum(denominator, epsilon)).astype(np.float32)


def patient_spearman_to_reference(
    current: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    left = np.asarray(current, dtype=np.float64).reshape(len(current), -1)
    right = np.asarray(reference, dtype=np.float64).reshape(len(reference), -1)
    if left.shape != right.shape:
        raise ValueError('Current and reference evidence shapes must match')
    output = []
    for current_row, reference_row in zip(left, right):
        value = float(spearmanr(current_row, reference_row).statistic)
        output.append(value if np.isfinite(value) else 0.0)
    return np.asarray(output, dtype=np.float32)


def recovery_fraction(
    zero_shot: float,
    budget_value: float,
    full_target: float,
    epsilon: float = 1e-12,
) -> float:
    denominator = full_target - zero_shot
    if abs(denominator) <= epsilon:
        return 0.0
    return float((budget_value - zero_shot) / denominator)




import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


CROSS_MODAL_ADAPTER_POLICY = 'native_electrode_set_attention'
DEFAULT_LATENT_CHANNELS = 18
DEFAULT_DESCRIPTOR_DIM = 16


@dataclass(frozen=True)
class NativeElectrodeAdapterContract:
    policy: str = CROSS_MODAL_ADAPTER_POLICY
    latent_channels: int = DEFAULT_LATENT_CHANNELS
    descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM
    hidden_dim: int = 32
    sampling_frequency: float = 256.0
    dropout: float = 0.0
    preserves_native_electrode_axis: bool = True
    truncates_native_electrodes: bool = False
    uses_target_labels: bool = False
    uses_target_test_statistics: bool = False
    source_training: str = 'joint_with_source_baseline'
    budget_training: str = 'full_model_finetuning_with_source_rehearsal'
    target_inference: str = 'frozen_checkpoint_single_final_evaluation'
    descriptor_scope: str = 'all_model_views_aggregated_once_per_clip'
    attention_shared_across_views: bool = True
    attribution_space: str = 'standardized_native_electrode_time'
    input_normalization: str = 'per_clip_per_native_channel_zscore'

    def __post_init__(self) -> None:
        if self.policy != CROSS_MODAL_ADAPTER_POLICY:
            raise ValueError(f'Unsupported cross-modal adapter policy: {self.policy}')
        if self.latent_channels <= 0:
            raise ValueError('latent_channels must be positive')
        if self.descriptor_dim != DEFAULT_DESCRIPTOR_DIM:
            raise ValueError(
                f'descriptor_dim must be {DEFAULT_DESCRIPTOR_DIM} for this policy'
            )
        if self.hidden_dim <= 0:
            raise ValueError('hidden_dim must be positive')
        if not math.isfinite(self.sampling_frequency) or self.sampling_frequency <= 90.0:
            raise ValueError('sampling_frequency must exceed 90 Hz for the configured bands')
        if self.dropout != 0.0:
            raise ValueError('The benchmark contract requires adapter dropout to be zero')

    @property
    def latent_channel_keys(self) -> tuple[str, ...]:
        return tuple(
            f'LATENT_ELECTRODE_SET_{index:02d}'
            for index in range(1, self.latent_channels + 1)
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(',', ':')
        ).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['latent_channel_keys'] = list(self.latent_channel_keys)
        payload['fingerprint'] = self.fingerprint
        return payload


def build_native_electrode_adapter(
    contract: NativeElectrodeAdapterContract | None = None,
):
    import torch

    selected = contract or NativeElectrodeAdapterContract()

    class NativeElectrodeSetAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.contract = selected
            self.descriptor_encoder = torch.nn.Sequential(
                torch.nn.LayerNorm(selected.descriptor_dim),
                torch.nn.Linear(selected.descriptor_dim, selected.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(selected.hidden_dim, selected.hidden_dim),
            )
            self.latent_queries = torch.nn.Parameter(
                torch.empty(selected.latent_channels, selected.hidden_dim)
            )
            torch.nn.init.orthogonal_(self.latent_queries)

        def _descriptors(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            centered = signal - signal.mean(dim=-1, keepdim=True)
            difference = signal[..., 1:] - signal[..., :-1]
            spectrum = torch.fft.rfft(centered, dim=-1).abs().square()
            frequency = torch.linspace(
                0.0, selected.sampling_frequency / 2.0, spectrum.shape[-1], device=signal.device,
                dtype=signal.dtype,
            )
            band_edges = (
                (0.5, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0),
                (30.0, 45.0), (45.0, 55.0), (65.0, 95.0),
            )
            total_selected = (
                (frequency >= 0.5)
                & (frequency < min(95.0, selected.sampling_frequency / 2.0))
                & ~((frequency >= 55.0) & (frequency < 65.0))
            )
            total_power = spectrum[..., total_selected].sum(dim=-1).clamp_min(1e-12)
            band_power = []
            for low, high in band_edges:
                band_selected = (frequency >= low) & (frequency < high)
                band_power.append(
                    spectrum[..., band_selected].sum(dim=-1) / total_power
                )
            normalized_signal = centered / centered.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            correlation = torch.einsum(
                'bct,bdt->bcd', normalized_signal, normalized_signal
            ).abs()
            channel_count = correlation.shape[-1]
            off_diagonal = 1.0 - torch.eye(
                channel_count, device=signal.device, dtype=signal.dtype
            )[None]
            valid_pairs = mask[:, :, None] * mask[:, None, :]
            correlation = correlation * off_diagonal * valid_pairs
            denominator = (mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
            mean_connectivity = correlation.sum(dim=-1) / denominator
            max_connectivity = correlation.amax(dim=-1)
            return torch.stack(
                (
                    signal.mean(dim=-1),
                    signal.std(dim=-1, unbiased=False),
                    signal.square().mean(dim=-1).clamp_min(1e-12).sqrt(),
                    signal.abs().mean(dim=-1),
                    difference.abs().mean(dim=-1),
                    centered.abs().amax(dim=-1),
                    *band_power,
                    difference.square().mean(dim=-1).clamp_min(1e-12).sqrt(),
                    mean_connectivity,
                    max_connectivity,
                ),
                dim=-1,
            )

        def forward(
            self,
            signal: torch.Tensor,
            channel_mask: torch.Tensor,
            return_attention: bool = False,
        ):
            if signal.ndim != 3:
                raise ValueError(
                    f'Native adapter expects batch by channel by time, got {tuple(signal.shape)}'
                )
            if channel_mask.shape != signal.shape[:2]:
                raise ValueError(
                    'Native adapter channel mask must match batch and channel dimensions'
                )
            mask = channel_mask.to(device=signal.device, dtype=torch.bool)
            if torch.any(mask.sum(dim=1) == 0):
                raise ValueError('Every clip must expose at least one native electrode')
            descriptors = self._descriptors(signal, mask)
            encoded = self.descriptor_encoder(descriptors)
            logits = torch.einsum(
                'bch,kh->bkc', encoded, self.latent_queries
            ) / math.sqrt(float(self.contract.hidden_dim))
            logits = logits.masked_fill(~mask[:, None, :], torch.finfo(logits.dtype).min)
            attention = torch.softmax(logits, dim=-1)
            latent = torch.einsum('bkc,bct->bkt', attention, signal)
            latent_mask = torch.ones(
                (signal.shape[0], self.contract.latent_channels),
                dtype=torch.bool,
                device=signal.device,
            )
            if return_attention:
                return latent, latent_mask, attention
            return latent, latent_mask

        def adapt_formatted(
            self,
            eeg: torch.Tensor,
            channel_mask: torch.Tensor,
            layout: str,
            return_attention: bool = False,
        ):
            if eeg.ndim < 4:
                raise ValueError(f'Formatted EEG tensor is invalid: {tuple(eeg.shape)}')
            batch, views = eeg.shape[:2]
            if layout == 'continuous':
                channel_count, points = eeg.shape[2], eeg.shape[3]
                view_signal = eeg
                restore = lambda value: value
            elif layout == 'eegnet':
                if eeg.ndim != 5 or eeg.shape[-1] != 1:
                    raise ValueError(f'EEGNet tensor is invalid: {tuple(eeg.shape)}')
                channel_count, points = eeg.shape[2], eeg.shape[3]
                view_signal = eeg[..., 0]
                restore = lambda value: value[..., None]
            elif layout == 'patch':
                if eeg.ndim != 5:
                    raise ValueError(f'Patch tensor is invalid: {tuple(eeg.shape)}')
                channel_count, patches, patch_points = eeg.shape[2:]
                points = patches * patch_points
                view_signal = eeg.reshape(batch, views, channel_count, points)
                restore = lambda value: value.reshape(
                    batch, views, value.shape[2], patches, patch_points
                )
            elif layout == 'time_channel_patch':
                if eeg.ndim != 5:
                    raise ValueError(
                        f'Time-channel-patch tensor is invalid: {tuple(eeg.shape)}'
                    )
                patches, channel_count, patch_points = eeg.shape[2:]
                points = patches * patch_points
                view_signal = eeg.transpose(2, 3).reshape(
                    batch, views, channel_count, points
                )
                restore = lambda value: value.reshape(
                    batch, views, value.shape[2], patches, patch_points
                ).transpose(2, 3)
            else:
                raise ValueError(f'Unsupported model layout: {layout}')
            if channel_mask.shape != (batch, channel_count):
                raise ValueError(
                    'Formatted adapter mask must match batch and native channel dimensions'
                )
            whole_clip = view_signal.transpose(1, 2).reshape(
                batch, channel_count, views * points
            )
            _, latent_mask, attention = self.forward(
                whole_clip, channel_mask, return_attention=True
            )
            latent_views = torch.einsum('bkc,bvct->bvkt', attention, view_signal)
            formatted = restore(latent_views)
            clip_mask = latent_mask
            if not return_attention:
                return formatted, clip_mask
            view_attention = attention[:, None].expand(
                batch, views, self.contract.latent_channels, channel_count
            )
            return formatted, clip_mask, view_attention

    return NativeElectrodeSetAdapter()


def attach_native_electrode_adapter(
    model: Any,
    policy: str,
    layout: str,
    contract: NativeElectrodeAdapterContract | None = None,
) -> dict[str, Any] | None:
    if policy != CROSS_MODAL_ADAPTER_POLICY:
        return None
    if hasattr(model, 'native_electrode_adapter'):
        raise ValueError('Model already has a native electrode adapter')
    adapter = build_native_electrode_adapter(contract)
    model.add_module('native_electrode_adapter', adapter)
    model.native_electrode_layout = str(layout)
    return adapter.contract.to_dict()


def attach_native_electrode_adapter_to(
    owner: Any,
    model: Any,
    policy: str,
    layout: str,
    contract: NativeElectrodeAdapterContract | None = None,
) -> dict[str, Any] | None:
    if policy != CROSS_MODAL_ADAPTER_POLICY:
        return None
    if hasattr(owner, 'native_electrode_adapter'):
        raise ValueError('Adapter owner already has a native electrode adapter')
    adapter = build_native_electrode_adapter(contract)
    owner.add_module('native_electrode_adapter', adapter)
    owner.native_electrode_layout = str(layout)
    model.native_electrode_adapter = adapter
    model.native_electrode_layout = str(layout)
    return adapter.contract.to_dict()


def apply_native_electrode_adapter(
    model: Any,
    eeg: Any,
    channel_mask: Any,
    return_attention: bool = False,
):
    adapter = getattr(model, 'native_electrode_adapter', None)
    if adapter is None:
        if return_attention:
            return eeg, channel_mask, None
        return eeg, channel_mask
    return adapter.adapt_formatted(
        eeg,
        channel_mask,
        layout=str(model.native_electrode_layout),
        return_attention=return_attention,
    )




import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eeg_benchmark.engine import CROSS_MODAL_LATENT_KEYS, parse_channel_names
from eeg_benchmark.tasks.cross_dataset import DATASET_DISPLAY_NAMES


APPROVED_CROSS_MODAL_TARGETS = {
    'detection': 'epilepsy_ieeg',
    'prediction': 'thalamocortical_ieeg',
    'localization': 'epilepsy_ieeg',
}
APPROVED_IEEG_TYPES = frozenset({'ECOG', 'SEEG'})


def _split_summary(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for split, split_frame in frame.groupby('split', sort=True):
        labels = split_frame['label'].astype(int).value_counts()
        output[str(split)] = {
            'patients': int(split_frame['patient_id'].astype(str).nunique()),
            'clips': int(len(split_frame)),
            'label_0': int(labels.get(0, 0)),
            'label_1': int(labels.get(1, 0)),
        }
    return output


def _channel_count_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    counts = np.asarray(
        [len(parse_channel_names(value)) for value in frame['channel_names']],
        dtype=np.int64,
    )
    return {
        'minimum': int(counts.min()),
        'median': float(np.median(counts)),
        'maximum': int(counts.max()),
        'unique_counts': int(np.unique(counts).size),
    }


def _montage_summary(frame: pd.DataFrame) -> dict[str, int]:
    if 'montage' not in frame:
        return {}
    counts = frame['montage'].fillna('unknown').astype(str).value_counts()
    return {str(key): int(value) for key, value in counts.sort_index().items()}


def _requested_signal_contract(task_root: Path) -> dict[str, Any]:
    path = task_root / 'dataset_contract.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing dataset contract: {path}')
    payload = json.loads(path.read_text(encoding='utf-8'))
    required = {
        'window_seconds', 'target_sfreq', 'requested_bandpass_hz',
        'requested_notch_freqs_hz', 'normalization',
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f'Dataset contract lacks cross-modal signal fields {sorted(missing)}: {path}'
        )
    contract = {
        'protocol_track': str(payload.get('protocol_track', 'unknown')),
        'window_seconds': float(payload['window_seconds']),
        'target_sfreq': int(payload['target_sfreq']),
        'requested_bandpass_hz': [
            float(value) for value in payload['requested_bandpass_hz']
        ],
        'requested_notch_freqs_hz': [
            float(value) for value in payload['requested_notch_freqs_hz']
        ],
        'normalization': str(payload['normalization']),
    }
    for field in (
        'line_noise_policy', 'notch_method', 'notch_filter_length',
        'bandpass_method', 'bandpass_phase', 'fir_window', 'fir_design',
        'resampling_method',
    ):
        contract[field] = str(payload.get(field, 'unknown'))
    contract['notch_mt_bandwidth'] = float(payload.get('notch_mt_bandwidth', 0.0))
    return contract


def _validate_signal_contract_pair(
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    required_values = {
        'protocol_track': 'eeg_ieeg_cross_modal',
        'target_sfreq': 256,
        'requested_bandpass_hz': [0.5, 95.0],
        'normalization': 'per_clip_per_channel_zscore',
        'line_noise_policy': 'dataset_native_spectrum_fit',
        'notch_method': 'spectrum_fit',
        'bandpass_method': 'fir',
        'bandpass_phase': 'zero',
        'resampling_method': 'fft',
    }
    for role, contract in (('source', source), ('target', target)):
        mismatches = {
            field: {'expected': expected, 'actual': contract.get(field)}
            for field, expected in required_values.items()
            if contract.get(field) != expected
        }
        if mismatches:
            raise ValueError(
                f'Cross-modal {role} cache violates the final preprocessing '
                f'contract: {mismatches}'
            )
    source_notches = source.get('requested_notch_freqs_hz', [])
    target_notches = target.get('requested_notch_freqs_hz', [])
    for role, values in (('source', source_notches), ('target', target_notches)):
        if len(values) != 1 or float(values[0]) not in {50.0, 60.0}:
            raise ValueError(
                f'Cross-modal {role} must use one dataset-native 50 or 60 Hz line frequency'
            )
    comparable_source = dict(source)
    comparable_target = dict(target)
    comparable_source.pop('requested_notch_freqs_hz', None)
    comparable_target.pop('requested_notch_freqs_hz', None)
    if comparable_source != comparable_target:
        raise ValueError(
            'Cross-modal source and target requested signal contracts differ: '
            f'{source} versus {target}'
        )


def _sample_archives(
    task_root: Path,
    frame: pd.DataFrame,
    maximum_archives: int,
) -> dict[str, Any]:
    ordered = frame.sort_values(['split', 'patient_id', 'clip_id'], kind='mergesort')
    selected_parts = []
    groups = list(ordered.groupby(['split', 'label'], sort=True))
    per_group = max(1, maximum_archives // max(len(groups), 1))
    for _, group in groups:
        selected_parts.append(group.head(per_group))
    selected = pd.concat(selected_parts, ignore_index=True).head(maximum_archives)
    channel_types: set[str] = set()
    finite_position_channels = 0
    total_position_channels = 0
    coordinate_systems: set[str] = set()
    normalization_values: set[str] = set()
    checked = []
    for row in selected.itertuples(index=False):
        clip_path = task_root / row.relative_path
        with np.load(clip_path, allow_pickle=False) as archive:
            names = [str(value) for value in archive['channel_names']]
            types = (
                [str(value).upper() for value in archive['channel_types']]
                if 'channel_types' in archive.files
                else ['UNKNOWN'] * len(names)
            )
            positions = (
                np.asarray(archive['channel_positions'], dtype=np.float32)
                if 'channel_positions' in archive.files
                else np.full((len(names), 3), np.nan, dtype=np.float32)
            )
            metadata = (
                json.loads(str(archive['metadata_json'].item()))
                if 'metadata_json' in archive.files
                else {}
            )
        if len(types) != len(names) or positions.shape != (len(names), 3):
            raise ValueError(f'Channel metadata shape mismatch: {clip_path}')
        channel_types.update(types)
        total_position_channels += len(names)
        finite_position_channels += int(np.isfinite(positions).all(axis=1).sum())
        coordinate = metadata.get('coordinate_metadata', {})
        coordinate_system = coordinate.get('coordinate_system')
        if coordinate_system:
            coordinate_systems.add(str(coordinate_system))
        normalization_values.add(str(metadata.get('normalization', 'unknown')))
        checked.append(str(row.clip_id))
    return {
        'checked_archive_count': len(checked),
        'checked_clip_ids': checked,
        'channel_types': sorted(channel_types),
        'coordinate_systems': sorted(coordinate_systems),
        'finite_coordinate_fraction': (
            float(finite_position_channels / total_position_channels)
            if total_position_channels
            else 0.0
        ),
        'normalization_values': sorted(normalization_values),
    }


def audit_cross_modal_cache(
    source_root: str | Path,
    target_root: str | Path,
    source_dataset: str,
    target_dataset: str,
    task: str,
    maximum_archives: int = 24,
    source_rehearsal_fraction: float = 0.25,
) -> dict[str, Any]:
    source_root = Path(source_root)
    target_root = Path(target_root)
    expected_target = APPROVED_CROSS_MODAL_TARGETS.get(task)
    if expected_target is None or target_dataset != expected_target:
        raise ValueError(
            f'Cross-modal {task} target must be {expected_target}, got {target_dataset}'
        )
    source = pd.read_csv(
        source_root / 'manifest.csv', dtype={'patient_id': str, 'clip_id': str}
    )
    target = pd.read_csv(
        target_root / 'manifest.csv', dtype={'patient_id': str, 'clip_id': str}
    )
    for name, frame in (('source', source), ('target', target)):
        required = {'split', 'patient_id', 'clip_id', 'label', 'relative_path', 'channel_names'}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f'{name} manifest lacks columns: {sorted(missing)}')
        split_membership = frame.groupby('patient_id')['split'].nunique()
        if int((split_membership > 1).sum()) != 0:
            raise ValueError(f'{name} patient leakage across train, dev, and test')
        for split in ('train', 'dev', 'test'):
            labels = set(frame.loc[frame['split'] == split, 'label'].astype(int))
            if labels != {0, 1}:
                raise ValueError(f'{name} {split} does not contain both classes')
    target_archive_audit = _sample_archives(
        target_root, target, maximum_archives=maximum_archives
    )
    source_signal_contract = _requested_signal_contract(source_root)
    target_signal_contract = _requested_signal_contract(target_root)
    _validate_signal_contract_pair(source_signal_contract, target_signal_contract)
    unsupported_types = sorted(
        set(target_archive_audit['channel_types']) - APPROVED_IEEG_TYPES
    )
    if unsupported_types:
        raise ValueError(f'Unsupported target channel types: {unsupported_types}')
    payload: dict[str, Any] = {
        'status': 'pass',
        'mission': 'inductive_eeg_to_ieeg_transfer',
        'task': task,
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'source_display_name': DATASET_DISPLAY_NAMES[source_dataset],
        'target_display_name': DATASET_DISPLAY_NAMES[target_dataset],
        'adapter': {
            'policy': 'native_electrode_set_attention',
            'fit_parameters': 'source_joint_training_then_budget_lpft',
            'output_channels': len(CROSS_MODAL_LATENT_KEYS),
            'output_channel_keys': list(CROSS_MODAL_LATENT_KEYS),
            'uses_target_labels': False,
            'uses_target_test_statistics': False,
            'uses_electrode_coordinates': False,
            'native_channels_retained_in_cache': True,
            'descriptor_scope': 'all_model_views_aggregated_once_per_clip',
            'attention_shared_across_views': True,
            'attribution_space': 'standardized_native_electrode_time',
        },
        'source': {
            'root': str(source_root),
            'splits': _split_summary(source),
            'channel_count': _channel_count_summary(source),
            'montages': _montage_summary(source),
            'preprocessing_statistics_json': str(
                source_root / 'preprocessing_statistics.json'
            ),
            'requested_signal_contract': source_signal_contract,
        },
        'target': {
            'root': str(target_root),
            'splits': _split_summary(target),
            'channel_count': _channel_count_summary(target),
            'montages': _montage_summary(target),
            'archive_metadata_sample': target_archive_audit,
            'preprocessing_statistics_json': str(
                target_root / 'preprocessing_statistics.json'
            ),
            'requested_signal_contract': target_signal_contract,
        },
        'selection_contract': {
            'zero_shot_model_selection': 'source_dev_auroc',
            'zero_shot_threshold': 'source_dev_max_f1',
            'budget_ft_model_selection': 'target_dev_auroc',
            'budget_ft_threshold': 'target_dev_max_f1',
            'budget_scope': 'target_train_and_dev_complete_patients',
            'target_dev_policy': 'same_percentage_nested_complete_patients',
            'target_test_role': 'single_final_evaluation_and_posthoc_interpretability_only',
            'budget_training': {
                'total_epochs': 50,
                'strategy': 'linear_probe_then_model_appropriate_full_finetuning',
                'learning_rate': 'model_specific_head_and_backbone_target_lr',
                'patience': 10,
                'target_sampling': (
                    'selected_patients_all_ictal_dynamic_hard_far_one_to_two'
                    if task == 'detection'
                    else 'selected_patients_all_preictal_dynamic_patient_balanced_interictal_one_to_one'
                ),
                'source_rehearsal_sampling': (
                    'patient_class_balanced_one_to_two'
                    if task == 'detection'
                    else 'patient_class_balanced_one_to_one'
                ),
                'source_rehearsal_fraction': float(source_rehearsal_fraction),
            },
        },
        'known_limitations': [
            'The latent adapter is permutation invariant; clinical attribution is computed before the adapter on native contacts.',
            'Coordinate-based anatomy is auxiliary because coordinate availability is incomplete.',
            'Scalp EEG and iEEG retain their native montage before the common deterministic bridge.',
        ],
    }
    payload['fingerprint'] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return payload


def write_cross_modal_audit(
    output_path: str | Path,
    source_root: str | Path,
    target_root: str | Path,
    source_dataset: str,
    target_dataset: str,
    task: str,
    source_rehearsal_fraction: float = 0.25,
) -> dict[str, Any]:
    payload = audit_cross_modal_cache(
        source_root,
        target_root,
        source_dataset,
        target_dataset,
        task,
        source_rehearsal_fraction=source_rehearsal_fraction,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return payload


def cross_modal_audit_main() -> None:
    parser = argparse.ArgumentParser(
        description='Audit the paper-ready inductive EEG to iEEG cache contract'
    )
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path, required=True)
    parser.add_argument('--source-dataset', choices=['tusz', 'siena', 'chbmit'], required=True)
    parser.add_argument(
        '--target-dataset',
        choices=['epilepsy_ieeg', 'thalamocortical_ieeg'],
        required=True,
    )
    parser.add_argument('--task', choices=['detection', 'prediction', 'localization'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = write_cross_modal_audit(
        args.output,
        args.source_root,
        args.target_root,
        args.source_dataset,
        args.target_dataset,
        args.task,
    )
    print('|Field|Value|')
    print('|:--|:--|')
    print(f'|Status|{payload["status"]}|')
    print(f'|Task|{payload["task"]}|')
    print(f'|Source|{payload["source_display_name"]}|')
    print(f'|Target|{payload["target_display_name"]}|')
    print(f'|Adapter|{payload["adapter"]["policy"]}|')
    print(f'|Output|{args.output}|')




import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PALETTE = {
    'vermilion': '#D55E00',
    'orange': '#E69F00',
    'warm': '#FDDBC7',
    'light_blue': '#56B4E9',
    'blue': '#0072B2',
    'navy': '#003366',
    'gray': '#B3B3B3',
    'charcoal': '#1A1A1A',
}

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
})


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix('.svg'), bbox_inches='tight')
    figure.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    figure.savefig(stem.with_suffix('.tiff'), dpi=600, bbox_inches='tight')
    plt.close(figure)


def deterministic_subset(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    patient_ids = np.asarray(patient_ids, dtype=str)
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    strata = sorted(set(zip(patient_ids.tolist(), labels.tolist())))
    quota = max(1, maximum // max(len(strata), 1))
    for patient_id, label in strata:
        candidates = np.flatnonzero((patient_ids == patient_id) & (labels == label))
        rng.shuffle(candidates)
        selected.extend(candidates[:quota].tolist())
    if len(selected) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), np.asarray(selected), assume_unique=False)
        rng.shuffle(remaining)
        selected.extend(remaining[: maximum - len(selected)].tolist())
    return np.asarray(sorted(selected[:maximum]), dtype=np.int64)


def save_embedding_bundle(
    output_dir: str | Path,
    features: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    maximum_clips: int,
) -> dict[str, object]:
    root = Path(output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    required = {'clip_id', 'patient_id', 'label', 'score'}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f'Embedding metadata lacks columns: {sorted(missing)}')
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != len(metadata):
        raise ValueError('Embedding matrix and metadata have incompatible shapes')
    selected = deterministic_subset(
        metadata['label'].to_numpy(),
        metadata['patient_id'].astype(str).to_numpy(),
        maximum_clips,
        seed,
    )
    selected_features = features[selected]
    selected_metadata = metadata.iloc[selected].reset_index(drop=True)
    np.savez_compressed(
        root / 'target_test_embeddings.npz',
        features=selected_features,
        clip_id=selected_metadata['clip_id'].astype(str).to_numpy(),
        patient_id=selected_metadata['patient_id'].astype(str).to_numpy(),
        label=selected_metadata['label'].astype(int).to_numpy(),
        score=selected_metadata['score'].astype(float).to_numpy(),
    )
    selected_metadata.to_csv(root / 'target_test_embedding_metadata.csv', index=False)
    if len(selected_metadata) < 4 or selected_metadata['label'].nunique() < 2:
        raise ValueError('Embedding visualization requires at least four samples and both classes')
    scaled = StandardScaler().fit_transform(selected_features)
    perplexity = min(30.0, max(2.0, (len(scaled) - 1.0) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        random_state=seed,
    ).fit_transform(scaled)
    _plot_embedding(tsne, selected_metadata, root / 'target_test_tsne', 't-SNE')
    umap_status = 'complete'
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(scaled) - 1),
            min_dist=0.1,
            metric='euclidean',
            random_state=seed,
            transform_seed=seed,
        )
        coordinates = reducer.fit_transform(scaled)
        _plot_embedding(coordinates, selected_metadata, root / 'target_test_umap', 'UMAP')
    except Exception as exc:
        umap_status = f'skipped: {type(exc).__name__}: {exc}'
    return {
        'status': 'complete',
        'sample_count': int(len(selected_metadata)),
        'feature_count': int(selected_features.shape[1]),
        'selection': 'deterministic_patient_and_label_stratified',
        'tsne_perplexity': float(perplexity),
        'umap_status': umap_status,
    }


def _plot_embedding(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    stem: Path,
    method: str,
) -> None:
    figure, axis = plt.subplots(figsize=(3.5, 3.0))
    colors = np.where(
        metadata['label'].astype(int).to_numpy() == 1,
        PALETTE['vermilion'],
        PALETTE['blue'],
    )
    axis.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=10, alpha=0.72, linewidths=0)
    axis.set_xlabel(f'{method}1')
    axis.set_ylabel(f'{method}2')
    axis.set_title('Target test representation')
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', color=PALETTE['blue'], label='Negative'),
        plt.Line2D([], [], marker='o', linestyle='', color=PALETTE['vermilion'], label='Positive'),
    ]
    axis.legend(handles=handles, loc='best')
    figure.tight_layout()
    _save_figure(figure, stem)


def plot_prediction_explanations(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    root = Path(output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.5, 2.8))
    negative = predictions.loc[predictions['label'].astype(int) == 0, 'score'].astype(float)
    positive = predictions.loc[predictions['label'].astype(int) == 1, 'score'].astype(float)
    bins = np.linspace(0.0, 1.0, 31)
    axis.hist(negative, bins=bins, density=True, alpha=0.62, color=PALETTE['blue'], label='Negative')
    axis.hist(positive, bins=bins, density=True, alpha=0.62, color=PALETTE['vermilion'], label='Positive')
    axis.set_xlabel('Predicted positive probability')
    axis.set_ylabel('Density')
    axis.set_title('Target test score distribution')
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, root / 'target_test_score_distribution')


class LastLinearInputCapture:
    def __init__(self, model: Any) -> None:
        import torch

        candidates = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        if not candidates:
            raise ValueError('The model has no linear layer for penultimate feature capture')
        binary = [module for module in candidates if module.out_features == 1]
        self.module = binary[-1] if binary else candidates[-1]
        self.values: list[Any] = []
        self.handle = self.module.register_forward_pre_hook(self._hook)

    def _hook(self, module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs:
            self.values.append(inputs[0].detach())

    def close(self) -> None:
        self.handle.remove()


def _aggregate_captured_features(values: list[Any], batch_size: int) -> np.ndarray:
    import torch

    if not values:
        raise ValueError('The classifier hook did not capture penultimate features')
    if len(values) == batch_size:
        rows = [value.reshape(-1, value.shape[-1]).mean(dim=0) for value in values]
        return torch.stack(rows).cpu().numpy().astype(np.float32)
    if len(values) == 1 and values[0].shape[0] == batch_size:
        value = values[0].reshape(batch_size, -1, values[0].shape[-1]).mean(dim=1)
        return value.cpu().numpy().astype(np.float32)
    merged = torch.cat([value.reshape(-1, value.shape[-1]) for value in values], dim=0)
    if merged.shape[0] % batch_size:
        raise ValueError('Cannot aggregate captured features to clip level')
    return merged.reshape(batch_size, -1, merged.shape[-1]).mean(dim=1).cpu().numpy().astype(np.float32)


def collect_standard_torch_embeddings(
    model: Any,
    loader: Any,
    device: Any,
    forward_clip: Callable[[Any, Any, Any], Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    import torch

    capture = LastLinearInputCapture(model)
    feature_parts = []
    rows = []
    try:
        model.to(device).eval()
        with torch.no_grad():
            for batch in loader:
                capture.values.clear()
                eeg = batch['eeg'].to(device, non_blocking=True)
                mask = batch['channel_mask'].to(device, non_blocking=True)
                scores = torch.sigmoid(forward_clip(model, eeg, mask).reshape(-1))
                feature_parts.append(_aggregate_captured_features(capture.values, len(scores)))
                for index, score in enumerate(scores.detach().cpu().numpy()):
                    rows.append({
                        'clip_id': str(batch['clip_id'][index]),
                        'patient_id': str(batch['patient_id'][index]),
                        'label': int(batch['label'][index]),
                        'score': float(score),
                    })
    finally:
        capture.close()
    return np.concatenate(feature_parts, axis=0), pd.DataFrame(rows)


def _channel_profile(signal: np.ndarray) -> np.ndarray:
    if signal.shape[-1] > 2048:
        step = int(math.ceil(signal.shape[-1] / 2048))
        signal = signal[:, ::step]
    difference = np.diff(signal, axis=-1)
    return np.stack([
        np.sqrt(np.mean(np.square(signal), axis=-1)),
        np.mean(np.abs(signal), axis=-1),
        np.std(signal, axis=-1),
        np.mean(np.abs(difference), axis=-1),
    ], axis=1)


def compute_channel_profiles(
    task_root: str | Path,
    predictions: pd.DataFrame,
    maximum_clips: int,
    seed: int,
) -> pd.DataFrame:
    root = Path(task_root)
    manifest = pd.read_csv(root / 'manifest.csv', dtype={'clip_id': str, 'patient_id': str})
    manifest_columns = ['clip_id', 'relative_path']
    for optional in ('coordinate_system', 'coordinate_units'):
        if optional in manifest.columns:
            manifest_columns.append(optional)
    merged = predictions[['clip_id', 'patient_id', 'label', 'score']].merge(
        manifest[manifest_columns], on='clip_id', how='inner', validate='one_to_one'
    )
    selected = deterministic_subset(
        merged['label'].to_numpy(), merged['patient_id'].to_numpy(),
        min(maximum_clips, 64), seed,
    )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for row in merged.iloc[selected].itertuples(index=False):
        with np.load(root / row.relative_path, allow_pickle=False) as archive:
            signal = np.asarray(archive['eeg'], dtype=np.float32)
            names = [str(value) for value in archive['channel_names']]
            positions = (
                np.asarray(archive['channel_positions'], dtype=np.float32)
                if 'channel_positions' in archive.files
                else np.full((len(names), 3), np.nan, dtype=np.float32)
            )
        profiles = _channel_profile(signal)
        for index, name in enumerate(names):
            key = (str(row.patient_id), name)
            item = records.setdefault(key, {
                'profile': [], 'score': [], 'position': [],
                'coordinate_system': str(getattr(row, 'coordinate_system', '')),
                'coordinate_units': str(getattr(row, 'coordinate_units', '')),
            })
            item['profile'].append(profiles[index])
            item['score'].append(float(row.score))
            item['position'].append(positions[index])
    output = []
    for (patient_id, channel_name), values in records.items():
        profile = np.asarray(values['profile'], dtype=np.float64)
        score = np.asarray(values['score'], dtype=np.float64)
        if len(score) >= 5 and np.std(score) > 0.0:
            association = abs(float(spearmanr(profile[:, 3], score).statistic))
            if not math.isfinite(association):
                association = 0.0
        else:
            association = float(np.mean(np.abs(profile[:, 3]) * np.maximum(score, 1e-6)))
        position = np.asarray(values['position'], dtype=np.float64)
        finite = np.isfinite(position).all(axis=1)
        xyz = np.median(position[finite], axis=0) if finite.any() else [np.nan, np.nan, np.nan]
        output.append({
            'patient_id': patient_id,
            'channel_name': channel_name,
            'channel_key': f'{patient_id}:{channel_name}',
            'importance': association,
            'profile_rms': float(profile[:, 0].mean()),
            'profile_mean_abs': float(profile[:, 1].mean()),
            'profile_std': float(profile[:, 2].mean()),
            'profile_line_length': float(profile[:, 3].mean()),
            'x': float(xyz[0]), 'y': float(xyz[1]), 'z': float(xyz[2]),
            'clip_count': int(len(score)),
            'coordinate_system': values['coordinate_system'],
            'coordinate_units': values['coordinate_units'],
        })
    frame = pd.DataFrame(output)
    if not frame.empty:
        maximum = float(frame['importance'].max())
        if maximum > 0.0:
            frame['importance'] = frame['importance'] / maximum
    return frame


def plot_cross_dataset_channel_linkage(
    source_profiles: pd.DataFrame,
    target_profiles: pd.DataFrame,
    output_dir: str | Path,
    maximum_channels: int = 16,
) -> None:
    if source_profiles.empty or target_profiles.empty:
        raise ValueError('Channel linkage requires non-empty source and target profiles')
    source = source_profiles.nlargest(maximum_channels, 'importance').copy()
    target = target_profiles.nlargest(maximum_channels, 'importance').copy()
    columns = ['profile_rms', 'profile_mean_abs', 'profile_std', 'profile_line_length']
    combined = np.concatenate(
        [source[columns].to_numpy(), target[columns].to_numpy()], axis=0
    )
    scaled = StandardScaler().fit_transform(combined)
    source_values = scaled[:len(source)]
    target_values = scaled[len(source):]
    source_norm = source_values / np.maximum(np.linalg.norm(source_values, axis=1, keepdims=True), 1e-8)
    target_norm = target_values / np.maximum(np.linalg.norm(target_values, axis=1, keepdims=True), 1e-8)
    similarity = (source_norm @ target_norm.T + 1.0) / 2.0
    importance = np.sqrt(
        source['importance'].to_numpy()[:, None] * target['importance'].to_numpy()[None, :]
    )
    linkage = similarity * importance
    root = Path(output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        linkage,
        index=source['channel_key'],
        columns=target['channel_key'],
    ).to_csv(root / 'source_target_channel_linkage.csv')
    figure, axis = plt.subplots(figsize=(6.8, 5.2))
    linkage_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'benchmark_linkage',
        [
            PALETTE['navy'], PALETTE['blue'], PALETTE['light_blue'],
            PALETTE['warm'], PALETTE['orange'], PALETTE['vermilion'],
        ],
    )
    image = axis.imshow(
        linkage, cmap=linkage_cmap, vmin=0.0,
        vmax=max(1.0, float(linkage.max())),
    )
    axis.set_xticks(np.arange(len(target)), labels=target['channel_key'], rotation=90)
    axis.set_yticks(np.arange(len(source)), labels=source['channel_key'])
    axis.set_xlabel('Target native channels')
    axis.set_ylabel('Source native channels')
    axis.set_title('Model-score-conditioned channel profile linkage')
    figure.colorbar(image, ax=axis, label='Linkage score')
    figure.tight_layout()
    _save_figure(figure, root / 'source_target_channel_linkage')


def plot_coronal_electrode_importance(
    target_profiles: pd.DataFrame,
    output_dir: str | Path,
    mri_template_path: str | Path | None = None,
) -> str:
    positioned = target_profiles.dropna(subset=['x', 'y', 'z']).copy()
    if positioned.empty:
        return 'skipped because target electrode coordinates are unavailable'
    selected = positioned.nlargest(64, 'importance')
    figure, axis = plt.subplots(figsize=(4.0, 3.2), facecolor=PALETTE['charcoal'])
    axis.set_facecolor(PALETTE['charcoal'])
    coordinate_systems = set(positioned.get('coordinate_system', pd.Series(dtype=str)).dropna().astype(str))
    used_mri = False
    plot_x = selected['x'].to_numpy()
    plot_z = selected['z'].to_numpy()
    if mri_template_path is not None and Path(mri_template_path).exists():
        if not any('MNI' in value.upper() for value in coordinate_systems):
            plt.close(figure)
            return 'skipped MRI overlay because electrode and template coordinate systems differ'
        try:
            import nibabel as nib

            image = nib.load(str(mri_template_path))
            volume = np.asarray(image.get_fdata(), dtype=np.float32)
            world = selected[['x', 'y', 'z']].to_numpy(dtype=np.float64)
            voxel = nib.affines.apply_affine(np.linalg.inv(image.affine), world)
            coronal = int(np.clip(np.median(voxel[:, 1]), 0, volume.shape[1] - 1))
            axis.imshow(
                np.rot90(volume[:, coronal, :]), cmap='gray', origin='lower',
                extent=[0, volume.shape[0], 0, volume.shape[2]], alpha=0.85,
            )
            plot_x = voxel[:, 0]
            plot_z = voxel[:, 2]
            used_mri = True
        except Exception:
            used_mri = False
    points = axis.scatter(
        plot_x, plot_z, c=selected['importance'],
        cmap='OrRd', s=20 + 65 * selected['importance'], alpha=0.85,
        edgecolors='white', linewidths=0.25,
    )
    if not used_mri:
        axis.axvline(0.0, color=PALETTE['gray'], linewidth=0.5, alpha=0.6)
    axis.set_xlabel('Left to right coordinate', color='white')
    axis.set_ylabel('Inferior to superior coordinate', color='white')
    axis.set_title('Coronal electrode importance projection', color='white')
    axis.tick_params(colors='white')
    for spine in axis.spines.values():
        spine.set_color('white')
    colorbar = figure.colorbar(points, ax=axis, label='Score association')
    colorbar.ax.yaxis.label.set_color('white')
    colorbar.ax.tick_params(colors='white')
    figure.tight_layout()
    _save_figure(figure, Path(output_dir) / 'interpretability' / 'coronal_electrode_importance')
    return 'complete with MRI background' if used_mri else 'complete coordinate projection without MRI background'


def write_interpretability_status(output_dir: str | Path, payload: dict[str, object]) -> None:
    root = Path(output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'status.json').write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8'
    )


def finalize_standard_torch_interpretability(
    spec: Any,
    model: Any,
    target_loader: Any,
    device: Any,
    forward_clip: Callable[[Any, Any, Any], Any],
    source_root: str | Path,
    target_root: str | Path,
    target_predictions: pd.DataFrame,
) -> dict[str, object]:
    status: dict[str, object] = {
        'enabled': bool(spec.generate_interpretability),
        'core_conclusion': 'The validation-selected model representation and native-channel score associations are traceable on target data',
        'archetype': 'quantitative_grid',
        'selection_rule': 'validation_auroc_never_target_test',
        'palette': PALETTE,
    }
    if not spec.generate_interpretability:
        write_interpretability_status(spec.output_dir, status)
        return status
    try:
        features, metadata = collect_standard_torch_embeddings(
            model, target_loader, device, forward_clip
        )
        status['embeddings'] = save_embedding_bundle(
            spec.output_dir, features, metadata, spec.seed,
            spec.interpretability_max_clips,
        )
    except Exception as exc:
        status['embeddings'] = f'skipped: {type(exc).__name__}: {exc}'
    try:
        source_prediction_path = (
            spec.output_dir / 'selection_predictions.csv'
            if spec.budget_percent == 0.0
            else spec.zero_shot_output_dir / 'selection_predictions.csv'
        )
        source_predictions = pd.read_csv(
            source_prediction_path, dtype={'clip_id': str, 'patient_id': str}
        )
        source_profiles = compute_channel_profiles(
            source_root, source_predictions, spec.interpretability_max_clips, spec.seed
        )
        target_profiles = compute_channel_profiles(
            target_root, target_predictions, spec.interpretability_max_clips, spec.seed
        )
        root = Path(spec.output_dir) / 'interpretability'
        root.mkdir(parents=True, exist_ok=True)
        source_profiles.to_csv(root / 'source_channel_profiles.csv', index=False)
        target_profiles.to_csv(root / 'target_channel_profiles.csv', index=False)
        plot_cross_dataset_channel_linkage(source_profiles, target_profiles, spec.output_dir)
        status['channel_linkage'] = 'complete'
        status['coronal_electrodes'] = plot_coronal_electrode_importance(
            target_profiles, spec.output_dir, spec.mri_template_path
        )
    except Exception as exc:
        status['channel_linkage'] = f'skipped: {type(exc).__name__}: {exc}'
    if spec.mission_type in {'eeg_ieeg_transfer', 'eeg_ieeg_localization'}:
        try:
            status['clinical_tasks'] = run_clinical_interpretability(
                spec, model, target_loader, device, forward_clip, target_root
            )
        except Exception as exc:
            status['clinical_tasks'] = f'skipped: {type(exc).__name__}: {exc}'
    write_interpretability_status(spec.output_dir, status)
    return status


def finalize_precomputed_interpretability(
    spec: Any,
    features: np.ndarray | None,
    embedding_metadata: pd.DataFrame | None,
    source_root: str | Path,
    target_root: str | Path,
    target_predictions: pd.DataFrame,
) -> dict[str, object]:
    status: dict[str, object] = {
        'enabled': bool(spec.generate_interpretability),
        'core_conclusion': 'The validation-selected model representation and native-channel score associations are traceable on target data',
        'archetype': 'quantitative_grid',
        'selection_rule': 'validation_auroc_never_target_test',
        'palette': PALETTE,
    }
    if not spec.generate_interpretability:
        write_interpretability_status(spec.output_dir, status)
        return status
    if features is not None and embedding_metadata is not None:
        try:
            status['embeddings'] = save_embedding_bundle(
                spec.output_dir, features, embedding_metadata, spec.seed,
                spec.interpretability_max_clips,
            )
        except Exception as exc:
            status['embeddings'] = f'skipped: {type(exc).__name__}: {exc}'
    else:
        status['embeddings'] = 'skipped because the native trainer did not expose features'
    try:
        source_prediction_path = (
            spec.output_dir / 'selection_predictions.csv'
            if spec.budget_percent == 0.0
            else spec.zero_shot_output_dir / 'selection_predictions.csv'
        )
        source_predictions = pd.read_csv(
            source_prediction_path, dtype={'clip_id': str, 'patient_id': str}
        )
        source_profiles = compute_channel_profiles(
            source_root, source_predictions, spec.interpretability_max_clips, spec.seed
        )
        target_profiles = compute_channel_profiles(
            target_root, target_predictions, spec.interpretability_max_clips, spec.seed
        )
        root = Path(spec.output_dir) / 'interpretability'
        root.mkdir(parents=True, exist_ok=True)
        source_profiles.to_csv(root / 'source_channel_profiles.csv', index=False)
        target_profiles.to_csv(root / 'target_channel_profiles.csv', index=False)
        plot_cross_dataset_channel_linkage(source_profiles, target_profiles, spec.output_dir)
        status['channel_linkage'] = 'complete'
        status['coronal_electrodes'] = plot_coronal_electrode_importance(
            target_profiles, spec.output_dir, spec.mri_template_path
        )
    except Exception as exc:
        status['channel_linkage'] = f'skipped: {type(exc).__name__}: {exc}'
    write_interpretability_status(spec.output_dir, status)
    return status


def refresh_budget_interpretability(spec: Any) -> dict[str, object]:
    direction_root = Path(spec.output_dir).parents[1]
    candidates = []
    for run_root in sorted(direction_root.glob(f'{spec.model}_*_budget/seed_{spec.seed}')):
        summary_path = run_root / 'run_summary.json'
        bundle_path = run_root / 'interpretability' / 'target_test_embeddings.npz'
        if not summary_path.exists() or not bundle_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if summary.get('status') != 'complete':
            continue
        budget_text = run_root.parent.name[len(spec.model) + 1:].replace('_budget', '')
        budget = 0.0 if budget_text == '0' else float(budget_text.rstrip('%'))
        candidates.append((budget, run_root, summary, bundle_path))
    if not candidates:
        return {'status': 'skipped', 'reason': 'no completed embedding bundles'}
    candidates.sort(key=lambda item: item[0])
    root = direction_root / 'interpretability_summary' / spec.model / f'seed_{spec.seed}'
    root.mkdir(parents=True, exist_ok=True)
    validation = {
        str(budget): float(summary.get('validation_auroc', float('nan')))
        for budget, _, summary, _ in candidates
    }
    valid = [item for item in candidates if math.isfinite(validation[str(item[0])])]
    best = max(valid, key=lambda item: validation[str(item[0])]) if valid else None
    (root / 'best_validation_budget.json').write_text(
        json.dumps({
            'selection_rule': 'maximum_validation_auroc_never_target_test',
            'best_budget_percent': None if best is None else best[0],
            'validation_auroc_by_budget': validation,
        }, ensure_ascii=False, indent=2, allow_nan=True),
        encoding='utf-8',
    )
    for method in ('tsne', 'umap'):
        figure, axes = plt.subplots(
            1, len(candidates), figsize=(3.0 * len(candidates), 2.8), squeeze=False
        )
        for axis, (budget, _, summary, bundle_path) in zip(axes[0], candidates):
            with np.load(bundle_path, allow_pickle=False) as archive:
                features = np.asarray(archive['features'], dtype=np.float32)
                labels = np.asarray(archive['label'], dtype=np.int64)
                patients = np.asarray(archive['patient_id'], dtype=str)
            selected = deterministic_subset(
                labels, patients, min(500, len(labels)), spec.seed
            )
            scaled = StandardScaler().fit_transform(features[selected])
            if method == 'tsne':
                perplexity = min(30.0, max(2.0, (len(scaled) - 1.0) / 3.0))
                coordinates = TSNE(
                    n_components=2, perplexity=perplexity, init='pca',
                    learning_rate='auto', random_state=spec.seed,
                ).fit_transform(scaled)
            else:
                try:
                    import umap

                    coordinates = umap.UMAP(
                        n_components=2, n_neighbors=min(15, len(scaled) - 1),
                        min_dist=0.1, random_state=spec.seed,
                        transform_seed=spec.seed,
                    ).fit_transform(scaled)
                except Exception:
                    plt.close(figure)
                    break
            colors = np.where(
                labels[selected] == 1, PALETTE['vermilion'], PALETTE['blue']
            )
            axis.scatter(
                coordinates[:, 0], coordinates[:, 1], c=colors,
                s=7, alpha=0.68, linewidths=0,
            )
            validation_value = validation[str(budget)]
            validation_label = 'NA' if not math.isfinite(validation_value) else f'{validation_value:.3f}'
            axis.set_title(f'{budget:g}% budget\nval AUROC={validation_label}')
            axis.set_xlabel(method.upper() + '1')
            axis.set_ylabel(method.upper() + '2')
        else:
            figure.suptitle('Target representation across label budgets')
            figure.tight_layout()
            _save_figure(figure, root / f'{method}_budget_comparison')
            continue
    return {
        'status': 'complete',
        'budget_count': len(candidates),
        'best_budget_percent': None if best is None else best[0],
        'selection_rule': 'validation_auroc',
        'output_root': str(root),
    }


def safe_refresh_budget_interpretability(spec: Any) -> dict[str, object]:
    
    try:
        payload = refresh_budget_interpretability(spec)
    except Exception as exc:
        payload = {'status': 'skipped', 'reason': f'{type(exc).__name__}: {exc}'}
    if spec.mission_type in {'eeg_ieeg_transfer', 'eeg_ieeg_localization'}:
        try:
            payload['I3_clinical_budget_recovery'] = refresh_i3_budget_recovery(spec)
        except Exception as exc:
            payload['I3_clinical_budget_recovery'] = {
                'status': 'skipped', 'reason': f'{type(exc).__name__}: {exc}'
            }
    return payload




import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding='utf-8'))


def collect_runs(result_root: Path) -> pd.DataFrame:
    mission_root = result_root / 'eeg_ieeg_transfer'
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(mission_root.glob('*/*/*/*/seed_*/metrics.json')):
        run_root = metrics_path.parent
        protocol_path = run_root / 'protocol.json'
        summary_path = run_root / 'run_summary.json'
        audit_path = run_root / 'cross_modal_audit.json'
        required = (protocol_path, summary_path, audit_path)
        if not all(path.is_file() and path.stat().st_size > 0 for path in required):
            continue
        protocol = _read_json(protocol_path)
        summary = _read_json(summary_path)
        audit = _read_json(audit_path)
        if summary.get('status') != 'complete' or audit.get('status') != 'pass':
            continue
        metrics = _read_json(metrics_path)
        clip = metrics.get('clip_level', {})
        event = metrics.get('event_level', {})
        uncertainty = metrics.get('uncertainty', {})
        rows.append({
            'model': protocol['model'],
            'task': protocol['task'],
            'source_dataset': protocol['source_dataset'],
            'target_dataset': protocol['target_dataset'],
            'window_seconds': float(protocol['window_seconds']),
            'budget_percent': float(protocol['budget_percent']),
            'seed': int(protocol['seed']),
            'use_pretrained': bool(protocol['use_pretrained']),
            'auroc': float(clip.get('auroc', np.nan)),
            'auprc': float(clip.get('auprc', np.nan)),
            'f1': float(clip.get('f1', np.nan)),
            'event_sensitivity': float(event.get('event_sensitivity', np.nan)),
            'event_precision': float(event.get('event_precision', np.nan)),
            'event_f1': float(event.get('event_f1', np.nan)),
            'fa_per_hour': float(event.get('fa_per_hour', np.nan)),
            'fa_per_24h': float(event.get('fa_per_24h', np.nan)),
            'bootstrap_patient_count': int(uncertainty.get('patient_count', 0)),
            'run_root': str(run_root),
        })
    return pd.DataFrame(rows)


def summarize_runs(runs: pd.DataFrame, expected_seeds: set[int]) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    group_columns = [
        'model', 'task', 'source_dataset', 'target_dataset',
        'window_seconds', 'budget_percent', 'use_pretrained',
    ]
    metric_columns = [
        'auroc', 'auprc', 'f1', 'event_sensitivity', 'event_precision',
        'event_f1', 'fa_per_hour', 'fa_per_24h',
    ]
    rows: list[dict[str, object]] = []
    for keys, group in runs.groupby(group_columns, sort=True, dropna=False):
        row = dict(zip(group_columns, keys))
        observed = set(group['seed'].astype(int))
        row['completed_seed_count'] = len(observed)
        row['completed_seeds'] = ','.join(str(value) for value in sorted(observed))
        row['missing_seeds'] = ','.join(
            str(value) for value in sorted(expected_seeds - observed)
        )
        row['status'] = 'complete' if expected_seeds.issubset(observed) else 'incomplete'
        for metric in metric_columns:
            values = (
                group[metric].to_numpy(dtype=float)
                if metric in group.columns
                else np.full(len(group), np.nan, dtype=float)
            )
            finite = values[np.isfinite(values)]
            row[f'{metric}_mean'] = float(finite.mean()) if finite.size else np.nan
            row[f'{metric}_std'] = (
                float(finite.std(ddof=1)) if finite.size > 1 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_main() -> None:
    parser = argparse.ArgumentParser(
        description='Aggregate completed EEG to iEEG runs without test-based selection'
    )
    parser.add_argument(
        '--result-root', type=Path,
        default=Path(
            os.environ.get(
                'RESULT_ROOT',
                os.environ.get(
                    'CROSS_MODAL_RESULT_ROOT',
                    str(Path.cwd() / 'data' / 'results' / 'cross_modal'),
                ),
            )
        ),
    )
    parser.add_argument('--expected-seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    runs = collect_runs(args.result_root)
    summary = summarize_runs(runs, set(args.expected_seeds))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / 'per_run.csv', index=False)
    summary.to_csv(args.output_dir / 'across_seed_summary.csv', index=False)
    print(summary.to_markdown(index=False) if not summary.empty else 'No completed runs found')


def main() -> None:
    from eeg_benchmark.tasks.cross_dataset import main as launch_main

    launch_main()


if __name__ == '__main__':
    main()
