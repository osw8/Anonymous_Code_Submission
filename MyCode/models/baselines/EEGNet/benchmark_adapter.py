from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import faulthandler
import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
EEGNET_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, EEGNET_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from eeg_benchmark.engine import align_preprocessed_clip, build_channel_union
from eeg_benchmark.tasks.cross_dataset import (
    SOURCE_REHEARSAL_FRACTION,
    patient_budget_rehearsal_indices,
)
from eeg_benchmark.engine import evaluate_predictions, select_f1_threshold
from eeg_benchmark.engine import print_model_information, save_json, setup_run_logger
from eeg_benchmark.engine import resample_clip, split_native_views
from eeg_benchmark.tasks.cross_dataset import add_mission_arguments, build_spec, prepare_mission, resolve_in_domain_source_checkpoint, validate_mode_args, validate_zero_shot_reference
from eeg_benchmark.tasks.cross_dataset import (
    DETECTION_UNDERSAMPLE_DATASETS,
    balance_prediction_training_frame,
    mission_sampling_summary,
    undersample_detection_training_frame,
)
from eeg_benchmark.tasks.cross_modal import finalize_precomputed_interpretability, safe_refresh_budget_interpretability
from eeg_benchmark.tasks.cross_modal import (
    CROSS_MODAL_ADAPTER_POLICY,
    NativeElectrodeAdapterContract,
)
from eeg_benchmark.tasks.cross_modal import (
    run_i1_epilepsy_soz,
    run_i4_epilepsy_localization,
    run_i2_thalamocortical_trajectory,
    run_epilepsy_localization_task,
)
from eeg_benchmark.tasks.cross_modal import build_fixed_xai_cohort
from eeg_benchmark.tasks.cross_modal import canonical_contact_name


class ClipSequence:
    def __init__(
        self,
        root: Path,
        frame: pd.DataFrame,
        contract,
        batch_size: int,
        shuffle: bool,
        seed: int,
        window_seconds: float,
        dataset_key: str | None = None,
        task: str | None = None,
        sampling_audit_root: Path | None = None,
        patient_balanced: bool = False,
        patient_budget_training: bool = False,
        source_rehearsal: tuple[Path, pd.DataFrame] | None = None,
        source_rehearsal_fraction: float = SOURCE_REHEARSAL_FRACTION,
    ) -> None:
        self.root = root
        target_frame = frame.reset_index(drop=True).copy()
        target_frame['_task_root'] = str(root)
        self.rehearsal_source_frame = None
        self.source_rehearsal_fraction = float(source_rehearsal_fraction)
        if source_rehearsal is not None:
            source_root, source_frame = source_rehearsal
            source_frame = source_frame.reset_index(drop=True).copy()
            source_frame['_task_root'] = str(source_root)
            self.rehearsal_source_frame = source_frame
            self.full_frame = pd.concat([target_frame, source_frame], ignore_index=True)
            self.target_frame_count = len(target_frame)
        else:
            self.full_frame = target_frame
            self.target_frame_count = len(target_frame)
        self.frame = self.full_frame
        self.contract = contract
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.epoch = -1
        self.dataset_key = dataset_key
        self.task = task
        self.dynamic_detection_sampling = bool(
            (task == 'detection' or patient_budget_training)
            and dataset_key in DETECTION_UNDERSAMPLE_DATASETS
        )
        self.dynamic_prediction_sampling = bool(
            task == 'prediction'
            and source_rehearsal is None
        )
        self.prediction_negative_to_positive_ratio = (
            1.0
            if task == 'prediction'
            else 2.0
        )
        self.sampling_audit_root = sampling_audit_root
        self.patient_balanced = bool(patient_balanced)
        self.indices = np.arange(len(self.frame))
        self.window_seconds = window_seconds
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index: int):
        selected = self.indices[batch_index * self.batch_size : (batch_index + 1) * self.batch_size]
        expected_views = int(round(self.window_seconds))
        input_channels = (
            int(self.contract.native_channel_capacity)
            if self.contract.policy == CROSS_MODAL_ADAPTER_POLICY
            else len(self.contract.channel_keys)
        )
        clips = np.empty(
            (len(selected), expected_views, input_channels, 128, 1),
            dtype=np.float32,
        )
        labels = np.empty(len(selected), dtype=np.float32)
        for sample_index, index in enumerate(selected):
            row = self.frame.iloc[int(index)]
            task_root = Path(str(row.get('_task_root', self.root)))
            with np.load(task_root / row.relative_path, allow_pickle=False) as archive:
                signal = np.asarray(archive["eeg"], dtype=np.float32)
                names = [str(value) for value in archive["channel_names"]]
                sfreq = int(archive["sfreq"])
                label = int(archive["label"])
                channel_mean_uv = np.asarray(archive['channel_mean_uv'], dtype=np.float32) if 'channel_mean_uv' in archive else None
                channel_std_uv = np.asarray(archive['channel_std_uv'], dtype=np.float32) if 'channel_std_uv' in archive else None
                metadata = json.loads(str(archive['metadata_json'].item())) if 'metadata_json' in archive else {}
            aligned, _ = align_preprocessed_clip(
                signal,
                names,
                self.contract,
                channel_mean_uv,
                channel_std_uv,
                metadata.get('normalization'),
                str(row.dataset),
            )
            aligned = resample_clip(aligned, sfreq, 128)
            if self.contract.policy == CROSS_MODAL_ADAPTER_POLICY:
                capacity = int(self.contract.native_channel_capacity)
                if aligned.shape[0] > capacity:
                    raise ValueError(
                        f'Native channel count {aligned.shape[0]} exceeds contract capacity {capacity}'
                    )
                padded = np.zeros((capacity, aligned.shape[1]), dtype=np.float32)
                padded[:aligned.shape[0]] = aligned
                aligned = padded
            views, _ = split_native_views(aligned, 128, 1.0)
            if views.shape != (expected_views, input_channels, 128):
                raise ValueError(
                    f'EEGNet expected {(expected_views, input_channels, 128)} '
                    f'but received {views.shape}'
                )
            clips[sample_index] = views[..., None]
            labels[sample_index] = label
        return clips, labels

    def on_epoch_end(self) -> None:
        if self.rehearsal_source_frame is not None:
            self.epoch += 1
            target = self.full_frame.iloc[:self.target_frame_count].reset_index(drop=True)
            source = self.full_frame.iloc[self.target_frame_count:].reset_index(drop=True)
            self.indices, report = patient_budget_rehearsal_indices(
                target, source, self.seed, self.epoch,
                rehearsal_fraction=self.source_rehearsal_fraction,
                target_dataset=str(self.dataset_key),
                source_dataset=str(source.iloc[0]['dataset']).lower(),
                task=self.task,
                prediction_negative_to_positive_ratio=(
                    self.prediction_negative_to_positive_ratio
                ),
            )
            self.frame = self.full_frame
            if (
                self.sampling_audit_root is not None
                and os.environ.get('SAMPLING_AUDIT_MODE', 'compact') != 'none'
            ):
                self.sampling_audit_root.mkdir(parents=True, exist_ok=True)
                (self.sampling_audit_root / f'rehearsal_epoch_{self.epoch:04d}.json').write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
                )
            return
        if self.dynamic_prediction_sampling:
            self.epoch += 1
            self.frame, _ = balance_prediction_training_frame(
                self.full_frame,
                dataset=str(self.dataset_key),
                split='train',
                seed=self.seed,
                epoch=self.epoch,
                audit_root=self.sampling_audit_root,
            )
            self.indices = np.arange(len(self.frame))
            return
        if self.patient_balanced:
            self.epoch += 1
            rng = np.random.default_rng(self.seed + self.epoch)
            patient_ids = sorted(self.full_frame['patient_id'].astype(str).unique())
            if not patient_ids:
                raise ValueError('Patient-balanced sampling requires patients')
            per_class = int(np.ceil(len(self.full_frame) / len(patient_ids) / 2))
            selected = []
            for patient_id in patient_ids:
                patient = self.full_frame['patient_id'].astype(str).to_numpy() == patient_id
                labels = self.full_frame['label'].astype(int).to_numpy()
                for label in (0, 1):
                    candidates = np.flatnonzero(patient & (labels == label))
                    if candidates.size == 0:
                        raise ValueError(
                            f'Patient-balanced class sampling requires both classes: {patient_id}'
                        )
                    selected.extend(rng.choice(
                        candidates, size=per_class, replace=candidates.size < per_class
                    ).tolist())
            self.indices = np.asarray(selected, dtype=np.int64)
            rng.shuffle(self.indices)
            return
        if self.dynamic_detection_sampling:
            self.epoch += 1
            self.frame, _ = undersample_detection_training_frame(
                self.full_frame,
                dataset=str(self.dataset_key),
                split='train',
                seed=self.seed,
                epoch=self.epoch,
                audit_root=self.sampling_audit_root,
            )
            self.indices = np.arange(len(self.frame))
            epoch_rng = np.random.default_rng(self.seed + self.epoch)
            epoch_rng.shuffle(self.indices)
            return
        if self.shuffle:
            self.rng.shuffle(self.indices)


def iter_prefetched_batches(sequence, workers: int):
    worker_count = max(int(workers), 1)
    if worker_count == 1:
        for index in range(len(sequence)):
            yield sequence[index]
        return

    pending_limit = min(worker_count, len(sequence))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix='eegnet-input',
    ) as executor:
        pending = deque()
        next_index = 0

        while next_index < pending_limit:
            pending.append(executor.submit(sequence.__getitem__, next_index))
            next_index += 1

        while pending:
            future = pending.popleft()
            yield future.result()
            if next_index < len(sequence):
                pending.append(executor.submit(sequence.__getitem__, next_index))
                next_index += 1


def load_frame(root: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(root / "manifest.csv", dtype={"patient_id": str, "clip_id": str})
    result = frame[frame["split"] == split].reset_index(drop=True)
    if result.empty:
        raise ValueError(f"No {split} clips in {root}")
    return result


def build_clip_model(contract, learning_rate: float, window_seconds: float):
    import tensorflow as tf
    from EEGModels import EEGNet

    latent_channel_count = len(contract.channel_keys)
    input_channel_count = (
        int(contract.native_channel_capacity)
        if contract.policy == CROSS_MODAL_ADAPTER_POLICY
        else latent_channel_count
    )

    class NativeElectrodeSetAdapterTF(tf.keras.layers.Layer):
        def __init__(self, latent_channels: int, hidden_dim: int = 32, **kwargs):
            super().__init__(**kwargs)
            self.latent_channels = int(latent_channels)
            self.hidden_dim = int(hidden_dim)
            self.normalization = tf.keras.layers.LayerNormalization(axis=-1)
            self.dense_one = tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu)
            self.dense_two = tf.keras.layers.Dense(hidden_dim)

        def build(self, input_shape):
            self.latent_queries = self.add_weight(
                name='latent_queries',
                shape=(self.latent_channels, self.hidden_dim),
                initializer=tf.keras.initializers.Orthogonal(),
                trainable=True,
            )
            super().build(input_shape)

        def call(self, inputs):
            signal = inputs[..., 0]
            batch = tf.shape(signal)[0]
            views = tf.shape(signal)[1]
            channels = tf.shape(signal)[2]
            points = tf.shape(signal)[3]
            native_mask = tf.reduce_any(tf.abs(signal) > 1e-7, axis=(1, 3))
            whole_clip = tf.reshape(
                tf.transpose(signal, perm=(0, 2, 1, 3)),
                (batch, channels, views * points),
            )
            centered = whole_clip - tf.reduce_mean(
                whole_clip, axis=-1, keepdims=True
            )
            difference = whole_clip[..., 1:] - whole_clip[..., :-1]
            spectrum = tf.square(tf.abs(tf.signal.rfft(centered)))
            spectrum_points = tf.shape(spectrum)[-1]
            frequency = tf.linspace(
                tf.cast(0.0, signal.dtype), tf.cast(64.0, signal.dtype), spectrum_points
            )
            total_selected = tf.logical_and(frequency >= 0.5, frequency < 64.0)
            total_selected = tf.logical_and(
                total_selected,
                tf.logical_not(tf.logical_and(frequency >= 55.0, frequency < 65.0)),
            )
            total_power = tf.maximum(
                tf.reduce_sum(
                    spectrum * tf.cast(total_selected[None, None, :], spectrum.dtype),
                    axis=-1,
                ),
                1e-12,
            )
            band_power = []
            for low, high in (
                (0.5, 4.0), (4.0, 8.0), (8.0, 13.0),
                (13.0, 30.0), (30.0, 45.0), (45.0, 55.0), (65.0, 95.0),
            ):
                selected = tf.logical_and(frequency >= low, frequency < high)
                selected_power = tf.reduce_sum(
                    spectrum * tf.cast(selected[None, None, :], spectrum.dtype),
                    axis=-1,
                )
                band_power.append(selected_power / total_power)
            normalized_signal = centered / tf.maximum(
                tf.norm(centered, axis=-1, keepdims=True), 1e-6
            )
            correlation = tf.abs(tf.einsum(
                'bct,bdt->bcd', normalized_signal, normalized_signal
            ))
            off_diagonal = 1.0 - tf.eye(channels, dtype=signal.dtype)[None]
            valid_pairs = (
                tf.cast(native_mask[:, :, None], signal.dtype)
                * tf.cast(native_mask[:, None, :], signal.dtype)
            )
            correlation = correlation * off_diagonal * valid_pairs
            denominator = tf.cast(
                tf.maximum(tf.reduce_sum(tf.cast(native_mask, tf.int32), axis=1, keepdims=True) - 1, 1),
                signal.dtype,
            )
            mean_connectivity = tf.reduce_sum(correlation, axis=-1) / denominator
            max_connectivity = tf.reduce_max(correlation, axis=-1)
            descriptors = tf.stack(
                (
                    tf.reduce_mean(whole_clip, axis=-1),
                    tf.math.reduce_std(whole_clip, axis=-1),
                    tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(whole_clip), axis=-1), 1e-12)),
                    tf.reduce_mean(tf.abs(whole_clip), axis=-1),
                    tf.reduce_mean(tf.abs(difference), axis=-1),
                    tf.reduce_max(tf.abs(centered), axis=-1),
                    *band_power,
                    tf.sqrt(tf.maximum(tf.reduce_mean(tf.square(difference), axis=-1), 1e-12)),
                    mean_connectivity,
                    max_connectivity,
                ),
                axis=-1,
            )
            encoded = self.dense_two(self.dense_one(self.normalization(descriptors)))
            logits = tf.einsum('bch,kh->bkc', encoded, self.latent_queries)
            logits = logits / tf.sqrt(tf.cast(self.hidden_dim, logits.dtype))
            logits = tf.where(native_mask[:, None, :], logits, tf.cast(-1e9, logits.dtype))
            attention = tf.nn.softmax(logits, axis=-1)
            latent = tf.einsum('bkc,bvct->bvkt', attention, signal)
            return latent[..., None]

    base = EEGNet(
        nb_classes=2,
        Chans=latent_channel_count,
        Samples=128,
        dropoutRate=0.0,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        dropoutType="Dropout",
    )
    view_count = int(round(window_seconds))
    clip_input = tf.keras.Input(
        shape=(view_count, input_channel_count, 128, 1), name="clip"
    )
    adapted = clip_input
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        adapted = NativeElectrodeSetAdapterTF(
            latent_channel_count, name='native_electrode_adapter'
        )(clip_input)
    flattened = tf.reshape(adapted, (-1, latent_channel_count, 128, 1))
    view_logit_model = tf.keras.Model(
        base.input,
        base.get_layer('dense').output,
        name='eegnet_view_logits',
    )
    view_logits = view_logit_model(flattened)
    view_logits = tf.reshape(view_logits, (-1, view_count, 2))
    clip_logits = tf.reduce_mean(view_logits, axis=1)
    output = tf.nn.softmax(clip_logits, axis=-1)
    class DiscriminativeClipModel(tf.keras.Model):
        def __init__(self, inputs, outputs, eegnet_base, **kwargs):
            super().__init__(inputs=inputs, outputs=outputs, **kwargs)
            self.eegnet_base = eegnet_base
            self.backbone_gradient_scale = 1.0
            self.head_variable_ids: set[int] = set()

        def configure_discriminative_finetuning(
            self, head_learning_rate: float, backbone_learning_rate: float,
        ) -> None:
            if not 0.0 < backbone_learning_rate <= head_learning_rate:
                raise ValueError(
                    'EEGNet backbone learning rate must be positive and not exceed head rate'
                )
            self.backbone_gradient_scale = (
                backbone_learning_rate / head_learning_rate
            )
            self.head_variable_ids = {
                id(variable)
                for variable in self.eegnet_base.get_layer('dense').trainable_variables
            }
            if not self.head_variable_ids:
                raise ValueError('EEGNet classifier head has no trainable variables')

        def train_step(self, data):
            x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                    regularization_losses=self.losses,
                )
            variables = self.trainable_variables
            gradients = tape.gradient(loss, variables)
            scaled = []
            for gradient, variable in zip(gradients, variables):
                if gradient is None or id(variable) in self.head_variable_ids:
                    scaled.append(gradient)
                else:
                    scaled.append(gradient * self.backbone_gradient_scale)
            self.optimizer.apply_gradients(
                (gradient, variable)
                for gradient, variable in zip(scaled, variables)
                if gradient is not None
            )
            self.compiled_metrics.update_state(
                y, y_pred, sample_weight=sample_weight
            )
            return {
                metric.name: metric.result() for metric in self.metrics
            }

    model = DiscriminativeClipModel(
        clip_input,
        output,
        base,
        name=f'EEGNet{view_count}s',
    )
    model.eegnet_view_logit_model = view_logit_model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


def prepare_target_spatial_kernel(model, contract) -> None:
    import tensorflow as tf

    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        return

    union_index = {name: index for index, name in enumerate(contract.channel_keys)}
    source_indices = [union_index[name] for name in contract.source_channel_keys]
    source_set = set(contract.source_channel_keys)
    target_only = [union_index[name] for name in contract.target_channel_keys if name not in source_set]
    if not source_indices or not target_only:
        return
    spatial_layer = next(
        layer for layer in model.eegnet_base.layers
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D)
    )
    weights = spatial_layer.get_weights()
    kernel = weights[0]
    source_mean = kernel[source_indices].mean(axis=0, keepdims=True)
    kernel[target_only] = source_mean
    weights[0] = kernel
    spatial_layer.set_weights(weights)


def configure_linear_probe(model) -> list[str]:
    for layer in model.layers:
        layer.trainable = False
    model.eegnet_view_logit_model.trainable = True
    model.eegnet_base.trainable = True
    for layer in model.eegnet_base.layers:
        layer.trainable = layer.name == 'dense'

    classifier = model.eegnet_base.get_layer('dense')
    expected_ids = {id(variable) for variable in classifier.trainable_variables}
    actual_ids = {id(variable) for variable in model.trainable_variables}
    if not expected_ids or actual_ids != expected_ids:
        raise RuntimeError(
            'EEGNet linear probe must expose only classifier-head variables'
        )
    return [variable.name for variable in model.trainable_variables]


def collect_eegnet_native_evidence(model, target_root, target_frame, contract, spec):
    import tensorflow as tf

    cohort = build_fixed_xai_cohort(
        target_frame,
        dataset=spec.target_dataset,
        task=spec.task,
        seed=spec.budget_seed,
        maximum_clips_per_patient_per_class=max(
            1, min(32, int(spec.interpretability_max_clips))
        ),
        require_both_classes_per_patient=not (
            spec.target_dataset == 'thalamocortical_ieeg' and spec.task == 'prediction'
        ),
    )
    shared_root = (
        Path(spec.result_root) / spec.mission_type / 'xai_cohorts' / spec.window_name
        / spec.target_dataset / spec.task
    )
    existing = shared_root / 'xai_cohort.json'
    if existing.is_file():
        payload = json.loads(existing.read_text(encoding='utf-8'))
        if payload.get('sha256') != cohort.sha256:
            shared_root.mkdir(parents=True, exist_ok=True)
            (shared_root / 'xai_cohort_conflict.json').write_text(
                json.dumps(
                    {
                        'status': 'rebuilt_shared_xai_cohort',
                        'reason': 'existing_shared_cohort_differs_from_current_contract',
                        'existing_sha256': payload.get('sha256'),
                        'current_sha256': cohort.sha256,
                        'existing_sampling_seed': payload.get('sampling_seed'),
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
    selected_frame = target_frame[
        target_frame['clip_id'].astype(str).isin(selected_ids)
    ].reset_index(drop=True)
    gradient_batch_size = max(1, min(int(spec.batch_size), 16))
    sequence = ClipSequence(
        Path(target_root), selected_frame, contract, gradient_batch_size, False,
        spec.seed, spec.window_seconds,
    )
    with tf.device('/CPU:0'):
        evidence_model = build_clip_model(
            contract, learning_rate=1e-3, window_seconds=spec.window_seconds
        )
        evidence_model.set_weights(model.get_weights())
        base = evidence_model.eegnet_base
        dense_layer = next(
            layer for layer in reversed(base.layers)
            if isinstance(layer, tf.keras.layers.Dense)
        )
        view_logit_model = tf.keras.Model(base.input, dense_layer.output)
        adapter_layer = evidence_model.get_layer('native_electrode_adapter')
    rows = []
    for batch_index in range(len(sequence)):
        values, labels = sequence[batch_index]
        with tf.device('/CPU:0'):
            values = tf.convert_to_tensor(values)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(values)
                adapted = adapter_layer(values, training=False)
                flat = tf.reshape(
                    adapted, (-1, len(contract.channel_keys), 128, 1)
                )
                view_logits = view_logit_model(flat, training=False)
                logits = tf.reduce_mean(
                    tf.reshape(view_logits, (-1, int(round(spec.window_seconds)), 2)),
                    axis=1,
                )[:, 1]
                positive_logit_sum = tf.reduce_sum(logits)
            gradient = tape.gradient(positive_logit_sum, values)
            if gradient is None:
                raise RuntimeError('EEGNet native-contact input gradient is unavailable')
            evidence = tf.reduce_sum(tf.abs(values * gradient), axis=(1, 3, 4))
            evidence = evidence / tf.maximum(
                tf.reduce_sum(evidence, axis=1, keepdims=True), 1e-12
            )
        scores = np.asarray(evidence)
        start = batch_index * gradient_batch_size
        batch_frame = selected_frame.iloc[start:start + len(values)]
        for sample_index, row in enumerate(batch_frame.itertuples(index=False)):
            with np.load(Path(target_root) / row.relative_path, allow_pickle=False) as archive:
                names = [str(value) for value in archive['channel_names']]
                positions = np.asarray(archive['channel_positions'], dtype=np.float32)
            for channel_index, name in enumerate(names):
                position = positions[channel_index]
                rows.append({
                    'dataset': spec.target_dataset,
                    'task': spec.task,
                    'patient_id': str(row.patient_id),
                    'clip_id': str(row.clip_id),
                    'label': int(labels[sample_index]),
                    'source_relative_path': str(row.source_relative_path),
                    'clip_start_seconds': float(row.clip_start_seconds),
                    'clip_end_seconds': float(row.clip_end_seconds),
                    'seizure_intervals_json': str(row.seizure_intervals_json),
                    'contact_name': name,
                    'contact_key': canonical_contact_name(name),
                    'evidence': float(scores[sample_index, channel_index]),
                    'x': float(position[0]), 'y': float(position[1]), 'z': float(position[2]),
                })
    return pd.DataFrame(rows), {
        'cohort_sha256': cohort.sha256,
        'clip_count': int(len(selected_frame)),
        'patient_count': int(cohort.frame['patient_id'].nunique()),
        'gradient_device': 'CPU:0',
        'gradient_batch_size': gradient_batch_size,
    }


def recover_run_summary_for_interpretability(spec):
    summary_path = spec.output_dir / 'run_summary.json'
    if summary_path.is_file():
        return json.loads(summary_path.read_text(encoding='utf-8'))
    metrics_path = spec.output_dir / 'metrics.json'
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f'Interpretability-only metrics do not exist: {metrics_path}'
        )
    summary = {
        'status': 'complete',
        'selection_metric': 'validation_auroc',
        'threshold_source': (
            'source_dev_max_f1'
            if spec.budget_percent == 0.0
            else 'fixed_target_dev_max_f1'
        ),
        'metrics': json.loads(metrics_path.read_text(encoding='utf-8')),
        'recovered_after_post_evaluation_interpretability_failure': True,
    }
    selection_path = spec.output_dir / 'selection_predictions.csv'
    if selection_path.is_file():
        selection = pd.read_csv(selection_path)
        if {'label', 'score'}.issubset(selection.columns):
            labels = selection['label'].astype(int).to_numpy()
            if np.unique(labels).size == 2:
                summary['validation_auroc'] = float(
                    roc_auc_score(labels, selection['score'].astype(float).to_numpy())
                )
    return summary


def finalize_eegnet_clinical_interpretability(model, target_root, target_frame, contract, spec):
    dataset_contract = json.loads(
        (Path(target_root) / 'dataset_contract.json').read_text(encoding='utf-8')
    )
    raw_root = dataset_contract.get('source_root')
    if not raw_root:
        return {'status': 'skipped_missing_source_root_contract'}
    evidence, cohort = collect_eegnet_native_evidence(
        model, target_root, target_frame, contract, spec
    )
    root = Path(spec.output_dir) / 'interpretability'
    root.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
    (root / 'fixed_cohort.json').write_text(json.dumps(cohort, indent=2), encoding='utf-8')
    if spec.target_dataset == 'epilepsy_ieeg':
        result = {'I1': run_i1_epilepsy_soz(evidence, raw_root, spec.output_dir)}
        if spec.mission_type == 'eeg_ieeg_localization':
            result['I4'] = run_i4_epilepsy_localization(evidence, raw_root, spec.output_dir)
        result['cohort'] = cohort
        return result
    return {'I2': run_i2_thalamocortical_trajectory(evidence, raw_root, spec.output_dir), 'cohort': cohort}


class GreenProgress:
    def callback(self, epochs: int, steps: int):
        import tensorflow as tf

        outer = self

        class Callback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                outer.epoch_bar = tqdm(total=epochs, desc="Training", unit="epoch", colour="green")

            def on_epoch_begin(self, epoch, logs=None):
                outer.batch_bar = tqdm(
                    total=steps,
                    desc=f'Train epoch {epoch + 1}/{epochs}',
                    unit='batch',
                    colour='green',
                    leave=False,
                    dynamic_ncols=True,
                )

            def on_train_batch_end(self, batch, logs=None):
                outer.batch_bar.update(1)
                if logs and 'loss' in logs:
                    outer.batch_bar.set_postfix(loss=f'{logs["loss"]:.5f}')

            def on_epoch_end(self, epoch, logs=None):
                outer.batch_bar.close()
                outer.epoch_bar.update(1)
                outer.epoch_bar.set_postfix({key: f"{value:.5f}" for key, value in (logs or {}).items()})

            def on_train_end(self, logs=None):
                outer.epoch_bar.close()

        return Callback()


class ValidationAUROC:
    def callback(
        self,
        sequence,
        patience: int,
        min_delta: float,
        checkpoint_path: Path,
        logger,
        workers: int,
        initial_best: float = -np.inf,
        enable_early_stop: bool = True,
    ):
        import tensorflow as tf

        outer = self

        class Callback(tf.keras.callbacks.Callback):
            def on_train_begin(self, logs=None):
                outer.best_checkpoint = float(initial_best)
                outer.early_stop_reference = -np.inf
                outer.stale = 0
                logger.info(
                    'Validation input prefetch enabled with %d ordered CPU workers',
                    max(int(workers), 1),
                )

            def on_epoch_end(self, epoch, logs=None):
                labels = []
                scores = []
                progress = tqdm(
                    total=len(sequence),
                    desc='Validation',
                    unit='batch',
                    colour='green',
                    leave=False,
                    dynamic_ncols=True,
                )
                try:
                    for x, y in iter_prefetched_batches(sequence, workers):
                        labels.append(y.astype(np.int64))
                        scores.append(self.model.predict_on_batch(x)[:, 1])
                        progress.update(1)
                finally:
                    progress.close()
                y_true = np.concatenate(labels)
                y_score = np.concatenate(scores)
                if np.unique(y_true).size < 2:
                    raise ValueError('EEGNet dev split must contain both classes')
                value = float(roc_auc_score(y_true, y_score))
                if logs is not None:
                    logs['val_auroc'] = value
                logger.info('epoch=%d val_auroc=%.8f', epoch + 1, value)
                if value > outer.best_checkpoint:
                    outer.best_checkpoint = value
                    self.model.save_weights(checkpoint_path)
                if value > outer.early_stop_reference + min_delta:
                    outer.early_stop_reference = value
                    outer.stale = 0
                else:
                    outer.stale += 1
                if enable_early_stop and outer.stale >= patience:
                    logger.info(
                        'early_stop_epoch=%d patience=%d min_delta=%.8f',
                        epoch + 1, patience, min_delta,
                    )
                    self.model.stop_training = True

        return Callback()


def predict(
    model,
    sequence: ClipSequence,
    frame: pd.DataFrame,
    workers: int,
) -> pd.DataFrame:
    score_parts = []
    label_parts = []
    progress = tqdm(
        total=len(sequence),
        desc='Evaluating',
        unit='batch',
        colour='green',
        dynamic_ncols=True,
    )
    try:
        for x, y in iter_prefetched_batches(sequence, workers):
            score_parts.append(model.predict_on_batch(x)[:, 1])
            label_parts.append(y.astype(np.int64))
            progress.update(1)
    finally:
        progress.close()
    result = frame.copy()
    result["label"] = np.concatenate(label_parts)
    result["score"] = np.concatenate(score_parts)
    return result


def collect_embeddings(
    model,
    sequence: ClipSequence,
    frame: pd.DataFrame,
    contract,
) -> tuple[np.ndarray, pd.DataFrame]:
    import tensorflow as tf

    base = model.eegnet_base
    extractor = tf.keras.Model(base.input, base.get_layer('flatten').output)
    adapter = (
        model.get_layer('native_electrode_adapter')
        if contract.policy == CROSS_MODAL_ADAPTER_POLICY
        else None
    )
    feature_parts = []
    score_parts = []
    label_parts = []
    for index in tqdm(range(len(sequence)), desc='Embedding', unit='batch', colour='green'):
        x, y = sequence[index]
        batch, views = x.shape[:2]
        values = tf.convert_to_tensor(x)
        if adapter is not None:
            values = adapter(values, training=False)
        flattened = tf.reshape(
            values,
            (batch * views, len(contract.channel_keys), 128, 1),
        )
        features = np.asarray(
            extractor(flattened, training=False)
        ).reshape(batch, views, -1).mean(axis=1)
        feature_parts.append(features.astype(np.float32))
        score_parts.append(model.predict_on_batch(x)[:, 1])
        label_parts.append(y.astype(np.int64))
    metadata = frame[['clip_id', 'patient_id']].copy()
    metadata['label'] = np.concatenate(label_parts)
    metadata['score'] = np.concatenate(score_parts)
    return np.concatenate(feature_parts, axis=0), metadata


def generate_eegnet_interpretability(
    model,
    test_sequence,
    target_test,
    contract,
    spec,
    source_root,
    target_root,
    predictions,
):
    features = None
    embedding_metadata = None
    if spec.generate_interpretability:
        features, embedding_metadata = collect_embeddings(
            model, test_sequence, target_test, contract
        )
    interpretability = finalize_precomputed_interpretability(
        spec, features, embedding_metadata, source_root, target_root, predictions
    )
    if spec.generate_interpretability and contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        interpretability['clinical_tasks'] = finalize_eegnet_clinical_interpretability(
            model, target_root, target_test, contract, spec
        )
    return interpretability


def run(args: argparse.Namespace) -> None:
    validate_mode_args(args)
    if args.use_pretrained:
        raise ValueError("EEGNet has no official pretrained weight")
    spec = build_spec(args, 'EEGNet')
    logger = setup_run_logger(spec.output_dir, 'benchmark.task1.eegnet')
    crash_path = spec.output_dir / 'crash.log'
    crash_stream = crash_path.open('a', encoding='utf-8', buffering=1)
    faulthandler.enable(file=crash_stream, all_threads=True)
    logger.info(
        'bootstrap_start python=%s executable=%s physical_gpu=%d logical_gpu=%d',
        sys.version.replace('\n', ' '),
        sys.executable,
        spec.physical_gpu,
        args.gpu,
    )
    numeric_mode = os.environ.get('BENCHMARK_NUMERIC_MODE', 'fp32').strip().lower()
    if numeric_mode not in {'fp32', 'tf32'}:
        raise ValueError('BENCHMARK_NUMERIC_MODE must be fp32 or tf32')
    fast_algorithms = bool(int(os.environ.get('BENCHMARK_FAST_ALGORITHMS', '0')))
    strict_deterministic_algorithms = bool(
        args.deterministic and not fast_algorithms and numeric_mode == 'fp32'
    )
    if strict_deterministic_algorithms:
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("TF_DETERMINISTIC_OPS", None)
        os.environ.pop("TF_CUDNN_DETERMINISTIC", None)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import tensorflow as tf
    except Exception:
        logger.exception('tensorflow_import_failed')
        raise
    try:
        tf.config.experimental.enable_tensor_float_32_execution(
            numeric_mode == 'tf32'
        )
    except Exception:
        logger.warning('tensorflow_tf32_configuration_unavailable')

    physical_gpus = tf.config.list_physical_devices('GPU')
    logger.info(
        'tensorflow_import_ok version=%s detected_gpus=%s',
        tf.__version__,
        [device.name for device in physical_gpus],
    )
    if args.gpu >= len(physical_gpus):
        raise ValueError(f'Requested GPU {args.gpu} but TensorFlow found {len(physical_gpus)} GPUs')
    selected_gpu = physical_gpus[args.gpu]
    tf.config.set_visible_devices(selected_gpu, 'GPU')
    tf.config.experimental.set_memory_growth(selected_gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    if strict_deterministic_algorithms:
        tf.config.experimental.enable_op_determinism()
    logger.info(
        'tensorflow_gpu_configuration_complete selected=%s numeric_mode=%s fast_algorithms=%s strict_deterministic_algorithms=%s',
        selected_gpu.name,
        numeric_mode,
        fast_algorithms,
        strict_deterministic_algorithms,
    )
    source_root, target_root, budget_selection = prepare_mission(spec)
    contract = build_channel_union(source_root / "manifest.csv", target_root / "manifest.csv", spec.source_dataset, spec.target_dataset)
    contract.save(spec.output_dir / "channel_union.json")
    effective_input_workers = args.num_workers
    effective_prefetch_batches = max(2 * effective_input_workers, 1)
    input_channels = (
        int(contract.native_channel_capacity)
        if contract.policy == CROSS_MODAL_ADAPTER_POLICY
        else len(contract.channel_keys)
    )
    raw_batch_gib = (
        args.batch_size
        * int(round(spec.window_seconds))
        * input_channels
        * 128
        * np.dtype(np.float32).itemsize
        / 1024**3
    )
    logger.info(
        'input_memory_contract channels=%d raw_batch_gib=%.4f '
        'requested_workers=%d effective_workers=%d',
        input_channels,
        raw_batch_gib,
        args.num_workers,
        effective_input_workers,
    )
    source_train = load_frame(source_root, "train")
    source_dev = load_frame(source_root, "dev")
    target_test = load_frame(target_root, "test")
    target_train = None
    target_dev = None
    if spec.budget_percent > 0.0:
        target_train = load_frame(target_root, "train")
        target_train = target_train[
            target_train['clip_id'].astype(str).isin(
                set(budget_selection.clip_ids('train'))
            )
        ].reset_index(drop=True)
        if target_train.empty:
            raise ValueError("Target budget selected no clips")
        target_dev = load_frame(target_root, "dev")
        target_dev = target_dev[
            target_dev['clip_id'].astype(str).isin(
                set(budget_selection.clip_ids('dev'))
            )
        ].reset_index(drop=True)
        if target_dev.empty:
            raise ValueError('Target budget selected no development clips')
    class KerasClipSequence(tf.keras.utils.Sequence):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def __len__(self):
            return len(self.wrapped)

        def __getitem__(self, index):
            return self.wrapped[index]

        def on_epoch_end(self):
            self.wrapped.on_epoch_end()

    train_sequence = KerasClipSequence(ClipSequence(
        source_root, source_train, contract, args.batch_size, True,
        spec.undersample_seed,
        spec.window_seconds,
        dataset_key=spec.source_dataset,
        task=spec.task,
        sampling_audit_root=spec.output_dir / 'sampling' / 'source_train',
    ))
    dev_sequence = KerasClipSequence(ClipSequence(source_root, source_dev, contract, args.batch_size, False, args.seed, spec.window_seconds))
    test_sequence = KerasClipSequence(ClipSequence(target_root, target_test, contract, args.batch_size, False, args.seed, spec.window_seconds))
    model = build_clip_model(contract, args.lr, spec.window_seconds)
    if contract.policy == CROSS_MODAL_ADAPTER_POLICY:
        save_json(
            spec.output_dir / 'cross_modal_adapter.json',
            NativeElectrodeAdapterContract(sampling_frequency=128.0).to_dict(),
        )
    print_model_information(
        {
            "Model": "EEGNet",
            'Mission': spec.mission_type,
            'In-domain source policy': (
                'reuse canonical completed source checkpoint'
                if spec.is_in_domain else 'train source model'
            ),
            "Parameters": model.count_params(),
            "Mode": spec.mode,
            "Budget percent": spec.budget_percent,
            "Budget unit": spec.to_dict()['budget_unit'],
            "Budget seed": spec.budget_seed,
            "Window seconds": spec.window_seconds,
            "Task": spec.task,
            "Source": spec.source_dataset,
            "Target": spec.target_dataset,
            "Use pretrained": False,
            "Channel policy": contract.policy,
            "Contract channels": len(contract.channel_keys),
            "Source available channels": len(contract.source_channel_keys),
            "Target available channels": len(contract.target_channel_keys),
            "Epochs": spec.epochs,
            "Patience": spec.patience,
            'Budget epochs': args.budget_epochs,
            'Budget patience': args.budget_patience,
            'Budget fine-tuning': 'linear_probe_then_full_model_discriminative_lr',
            'Budget head-only epochs': args.budget_head_only_epochs,
            'Early stopping min delta': spec.min_delta,
            "Seed": spec.seed,
            "GPU": spec.physical_gpu,
            "Logical GPU after visibility mask": spec.gpu,
            "Batch size": spec.batch_size,
            'Input CPU workers': spec.num_workers,
            'Effective input CPU workers': effective_input_workers,
            'Raw input batch GiB': raw_batch_gib,
            'Validation prefetch batches': effective_prefetch_batches,
            'TensorFlow GPU memory growth': True,
            'Statistics CPU workers': spec.stats_num_workers,
            'Bootstrap resamples': spec.bootstrap_resamples,
            'Training sampling': mission_sampling_summary(spec),
            'Source rehearsal fraction': spec.source_rehearsal_fraction,
            'Undersample seed': spec.undersample_seed,
            'Interpretability': spec.generate_interpretability,
            "Output": spec.output_dir,
        }
    )
    save_json(spec.output_dir / "args.json", {
        **spec.to_dict(),
        "learning_rate": args.lr,
        "target_learning_rate": args.target_lr,
        'target_backbone_learning_rate': args.target_backbone_lr,
        'budget_epochs': args.budget_epochs,
        'budget_patience': args.budget_patience,
        'budget_head_only_epochs': args.budget_head_only_epochs,
        'budget_finetune_strategy': 'linear_probe_then_full_model_discriminative_lr',
        'effective_input_workers': effective_input_workers,
        'effective_prefetch_batches': effective_prefetch_batches,
        'raw_input_batch_gib': raw_batch_gib,
    })
    if spec.task == 'localization':
        reference_checkpoint = spec.source_reference_checkpoint_path
        if not reference_checkpoint.is_file():
            raise FileNotFoundError(
                f'Localization requires a completed detection reference checkpoint: {reference_checkpoint}'
            )
        logger.info(
            'Loading localization reference checkpoint from %s',
            reference_checkpoint,
        )
        model.load_weights(reference_checkpoint)
        prepare_target_spatial_kernel(model, contract)
        save_json(spec.output_dir / 'reference_checkpoint.json', {
            'path': str(reference_checkpoint),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'policy': 'reuse_completed_detection_checkpoint_for_soz_localization',
        })
        evidence, cohort = collect_eegnet_native_evidence(
            model,
            target_root,
            target_test,
            contract,
            spec,
        )
        root = spec.output_dir / 'localization'
        root.mkdir(parents=True, exist_ok=True)
        evidence.to_csv(spec.output_dir / 'predictions.csv', index=False)
        evidence.to_csv(root / 'native_contact_evidence.csv', index=False)
        (root / 'fixed_cohort.json').write_text(
            json.dumps(cohort, indent=2),
            encoding='utf-8',
        )
        target_contract = json.loads(
            (Path(target_root) / 'dataset_contract.json').read_text(
                encoding='utf-8'
            )
        )
        raw_root = target_contract.get('source_root')
        if not raw_root:
            raise ValueError(
                'Localization requires source_root in target dataset contract'
            )
        summary = run_epilepsy_localization_task(
            evidence,
            raw_root,
            spec.output_dir,
        )
        save_json(spec.output_dir / 'metrics.json', summary)
        save_json(spec.output_dir / 'run_summary.json', {
            'status': 'complete',
            'task': spec.task,
            'localization': summary,
            'reference_checkpoint': str(reference_checkpoint),
            'reference_output_dir': str(spec.source_reference_output_dir),
            'cohort_summary': cohort,
        })
        logger.info(
            'Completed EEGNet localization run at %s',
            spec.output_dir,
        )
        return
    if args.interpretability_only:
        if not spec.generate_interpretability:
            raise ValueError('Interpretability-only mode requires interpretability enabled')
        checkpoint_path = (
            spec.output_dir / 'source' / 'best.weights.h5'
            if spec.budget_percent == 0.0
            else spec.output_dir / 'target' / 'best.weights.h5'
        )
        predictions_path = spec.output_dir / 'predictions.csv'
        summary_path = spec.output_dir / 'run_summary.json'
        for required_path in (checkpoint_path, predictions_path):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f'Interpretability-only input does not exist: {required_path}'
                )
        model.load_weights(checkpoint_path)
        if spec.budget_percent == 0.0:
            prepare_target_spatial_kernel(model, contract)
        predictions = pd.read_csv(
            predictions_path, dtype={'clip_id': str, 'patient_id': str}
        )
        interpretability = generate_eegnet_interpretability(
            model,
            test_sequence,
            target_test,
            contract,
            spec,
            source_root,
            target_root,
            predictions,
        )
        run_summary = recover_run_summary_for_interpretability(spec)
        run_summary['interpretability'] = interpretability
        run_summary['interpretability_refresh'] = {
            'status': 'complete',
            'checkpoint': str(checkpoint_path),
        }
        save_json(summary_path, run_summary)
        run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
        save_json(summary_path, run_summary)
        logger.info('Completed EEGNet interpretability refresh at %s', spec.output_dir)
        return
    if spec.budget_percent == 0.0:
        stage_root = spec.output_dir / "source"
        if spec.is_in_domain:
            checkpoint_path, source_reference = resolve_in_domain_source_checkpoint(
                spec,
                'best.weights.h5',
                {'learning_rate': args.lr},
                contract,
            )
            save_json(spec.output_dir / 'source_checkpoint_reference.json', source_reference)
            logger.info('In-domain source checkpoint loaded without source retraining: %s', checkpoint_path)
        else:
            stage_root.mkdir(parents=True, exist_ok=True)
            checkpoint_path = stage_root / "best.weights.h5"
            callbacks = [
                ValidationAUROC().callback(
                    dev_sequence,
                    args.patience,
                    args.min_delta,
                    checkpoint_path,
                    logger,
                    effective_input_workers,
                ),
                GreenProgress().callback(args.epochs, len(train_sequence)),
                tf.keras.callbacks.CSVLogger(stage_root / "epochs.csv"),
            ]
            model.fit(
                train_sequence,
                epochs=args.epochs,
                callbacks=callbacks,
                verbose=0,
                shuffle=False,
                workers=effective_input_workers,
                use_multiprocessing=False,
                max_queue_size=effective_prefetch_batches,
            )
            model.save_weights(stage_root / "last.weights.h5")
        model.load_weights(checkpoint_path)
        selection_sequence = dev_sequence
        selection_frame = source_dev
        threshold_source = "source_dev_max_f1"
    else:
        source_checkpoint = spec.zero_shot_output_dir / "source" / "best.weights.h5"
        validate_zero_shot_reference(spec, source_checkpoint, {
            "learning_rate": args.lr,
        }, contract)
        model.load_weights(source_checkpoint)
        prepare_target_spatial_kernel(model, contract)
        target_train_sequence = KerasClipSequence(ClipSequence(
            target_root, target_train, contract, args.batch_size, True,
            spec.undersample_seed,
            spec.window_seconds,
            dataset_key=spec.target_dataset,
            task=spec.task,
            sampling_audit_root=spec.output_dir / 'sampling' / 'target_train',
            patient_balanced=False,
            patient_budget_training=True,
            source_rehearsal=(source_root, source_train),
            source_rehearsal_fraction=args.source_rehearsal_fraction,
        ))
        target_dev_sequence = KerasClipSequence(ClipSequence(target_root, target_dev, contract, args.batch_size, False, args.seed, spec.window_seconds))
        stage_root = spec.output_dir / "target"
        stage_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = stage_root / "best.weights.h5"
        if args.budget_head_only_epochs > 0:
            probe_root = spec.output_dir / 'target_linear_probe'
            probe_root.mkdir(parents=True, exist_ok=True)
            probe_checkpoint = probe_root / 'best.weights.h5'
            probe_variables = configure_linear_probe(model)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.target_lr),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            )
            logger.info(
                'target_stage phase=linear_probe epochs=%d head_lr=%g',
                args.budget_head_only_epochs,
                args.target_lr,
            )
            logger.info(
                'target_stage phase=linear_probe trainable_variables=%s',
                probe_variables,
            )
            probe_callbacks = [
                ValidationAUROC().callback(
                    target_dev_sequence,
                    args.budget_head_only_epochs + 1,
                    args.min_delta,
                    probe_checkpoint,
                    logger,
                    effective_input_workers,
                    enable_early_stop=False,
                ),
                GreenProgress().callback(
                    args.budget_head_only_epochs,
                    len(target_train_sequence),
                ),
                tf.keras.callbacks.CSVLogger(probe_root / 'epochs.csv'),
            ]
            model.fit(
                target_train_sequence,
                epochs=args.budget_head_only_epochs,
                callbacks=probe_callbacks,
                verbose=0,
                shuffle=False,
                workers=effective_input_workers,
                use_multiprocessing=False,
                max_queue_size=effective_prefetch_batches,
            )
            model.save_weights(probe_root / 'last.weights.h5')
            model.load_weights(probe_checkpoint)
        for layer in model.layers:
            layer.trainable = True
        model.eegnet_base.trainable = True
        for layer in model.eegnet_base.layers:
            layer.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.target_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        )
        model.configure_discriminative_finetuning(
            args.target_lr, args.target_backbone_lr,
        )
        logger.info(
            'target_stage phase=full_model strategy=discriminative_lr '
            'head_lr=%g backbone_lr=%g',
            args.target_lr, args.target_backbone_lr,
        )
        target_callbacks = [
            ValidationAUROC().callback(
                target_dev_sequence,
                args.budget_patience,
                args.min_delta,
                checkpoint_path,
                logger,
                effective_input_workers,
                enable_early_stop=True,
            ),
            GreenProgress().callback(args.budget_epochs, len(target_train_sequence)),
            tf.keras.callbacks.CSVLogger(stage_root / "epochs.csv"),
        ]
        model.fit(
            target_train_sequence,
            epochs=args.budget_epochs,
            callbacks=target_callbacks,
            verbose=0,
            shuffle=False,
            workers=effective_input_workers,
            use_multiprocessing=False,
            max_queue_size=effective_prefetch_batches,
        )
        model.save_weights(stage_root / "last.weights.h5")
        model.load_weights(checkpoint_path)
        save_json(spec.output_dir / "source_checkpoint_reference.json", {"path": str(source_checkpoint)})
        selection_sequence = target_dev_sequence
        selection_frame = target_dev
        threshold_source = "target_dev_max_f1"
    dev_predictions = predict(
        model,
        selection_sequence,
        selection_frame,
        effective_input_workers,
    )
    dev_predictions.to_csv(spec.output_dir / 'selection_predictions.csv', index=False)
    validation_auroc = float(roc_auc_score(
        dev_predictions['label'], dev_predictions['score']
    ))
    threshold = select_f1_threshold(dev_predictions["label"], dev_predictions["score"])
    if spec.budget_percent == 0.0:
        prepare_target_spatial_kernel(model, contract)
    predictions = predict(
        model,
        test_sequence,
        target_test,
        effective_input_workers,
    )
    predictions["predicted_label"] = (predictions["score"] >= threshold).astype(np.int64)
    metrics = evaluate_predictions(
        predictions,
        spec.task,
        threshold,
        spec.output_dir,
        bootstrap_seed=spec.seed,
        bootstrap_resamples=spec.bootstrap_resamples,
        bootstrap_workers=spec.stats_num_workers,
    )
    interpretability = generate_eegnet_interpretability(
        model,
        test_sequence,
        target_test,
        contract,
        spec,
        source_root,
        target_root,
        predictions,
    )
    run_summary = {
        "status": "complete", "selection_metric": "validation_auroc",
        "threshold_source": threshold_source, 'validation_auroc': validation_auroc,
        "metrics": metrics, 'interpretability': interpretability,
    }
    save_json(spec.output_dir / "run_summary.json", run_summary)
    run_summary['budget_interpretability'] = safe_refresh_budget_interpretability(spec)
    save_json(spec.output_dir / "run_summary.json", run_summary)
    logger.info("Completed EEGNet Task1 run at %s", spec.output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EEGNet native Keras trainer for Task1")
    parser.add_argument("--model", choices=["EEGNet"], default="EEGNet")
    add_mission_arguments(parser)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-lr", type=float, default=1e-4)
    parser.add_argument('--interpretability-only', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as error:
        trace = traceback.format_exc()
        try:
            spec = build_spec(args, 'EEGNet')
            spec.output_dir.mkdir(parents=True, exist_ok=True)
            save_json(spec.output_dir / 'fatal_error.json', {
                'error_type': type(error).__name__,
                'message': str(error),
                'traceback': trace,
                'python': sys.version,
                'executable': sys.executable,
            })
            logger = setup_run_logger(
                spec.output_dir, 'benchmark.task1.eegnet.fatal'
            )
            logger.error('fatal_error\n%s', trace)
        except Exception:
            print(trace, file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
