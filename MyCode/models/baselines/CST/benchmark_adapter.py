from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from models.baselines.CST.ResizeNet import TransformerResizeNet
from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


class EEGNetFeatureExtractor(nn.Module):
    def __init__(
        self,
        channels: int,
        points: int = 128,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel: int = 64,
        separable_kernel: int = 16,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.points = int(points)
        depthwise_filters = temporal_filters * depth_multiplier
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, temporal_kernel),
                padding='same',
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                depthwise_filters,
                kernel_size=(self.channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                depthwise_filters,
                depthwise_filters,
                kernel_size=(1, separable_kernel),
                padding='same',
                groups=depthwise_filters,
                bias=False,
            ),
            nn.Conv2d(depthwise_filters, separable_filters, kernel_size=1, bias=False),
            nn.BatchNorm2d(separable_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
        )
        with torch.no_grad():
            example = torch.zeros(1, 1, self.channels, self.points)
            feature_dim = int(self._feature_map(example).numel())
        self.feature_dim = feature_dim

    def _feature_map(self, signal: torch.Tensor) -> torch.Tensor:
        return self.separable(self.spatial(self.temporal(signal)))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self._feature_map(signal).flatten(start_dim=1)


class CSTTransferModel(nn.Module):
    def __init__(
        self,
        contract,
        resize_heads: int = 2,
        resize_layers: int = 2,
        resize_feedforward: int = 128,
    ) -> None:
        super().__init__()
        self.cross_modal = contract.policy == 'native_electrode_set_attention'
        self.output_channels = 16
        capacity = max(
            int(getattr(contract, 'native_channel_capacity', 0)),
            len(contract.channel_keys),
            self.output_channels,
        )
        if capacity % resize_heads:
            capacity += resize_heads - capacity % resize_heads
        self.input_capacity = capacity
        self.resize_heads = int(resize_heads)
        self.resize_layers = int(resize_layers)
        self.resize_feedforward = int(resize_feedforward)
        self.target_dataset = self._dataset_key(contract.target_dataset)
        self.resizenet = (
            TransformerResizeNet(
                input_dim=self.input_capacity,
                output_dim=self.output_channels,
                num_heads=self.resize_heads,
                num_layers=self.resize_layers,
                dim_feedforward=self.resize_feedforward,
                dropout=0.0,
            )
            if self.cross_modal
            else None
        )
        self.feature_extractor = EEGNetFeatureExtractor(self.output_channels)
        self.classifier = nn.Linear(self.feature_extractor.feature_dim, 2)

    @staticmethod
    def _dataset_key(value: str) -> str:
        return re.sub(r'[^a-z0-9]', '', str(value).lower())

    def _pad_channels(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, channels, points = eeg.shape
        if channels > self.input_capacity:
            raise ValueError(
                f'CST received {channels} channels but its audited capacity is '
                f'{self.input_capacity}'
            )
        signal = eeg * channel_mask[:, None, :, None].to(eeg.dtype)
        if channels == self.input_capacity:
            return signal, channel_mask
        padded = signal.new_zeros(batch, views, self.input_capacity, points)
        padded[:, :, :channels] = signal
        mask = torch.zeros(
            batch,
            self.input_capacity,
            dtype=torch.bool,
            device=channel_mask.device,
        )
        mask[:, :channels] = channel_mask
        return padded, mask

    def forward_details(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if eeg.ndim != 5 or eeg.shape[-1] != 1:
            raise ValueError(
                f'CST expects batch by view by channel by time by one, got '
                f'{tuple(eeg.shape)}'
            )
        signal = eeg[..., 0]
        batch, views, channels, points = signal.shape
        if points != 128:
            raise ValueError(f'CST expects 128 samples per view, got {points}')
        if channel_mask.shape != (batch, channels):
            raise ValueError('CST channel mask does not match its signal')
        padded, _ = self._pad_channels(signal, channel_mask)
        selected = padded[:, :, :self.output_channels]
        selected_views = selected.reshape(
            batch * views, 1, self.output_channels, points
        )
        selected_features = self.feature_extractor(selected_views)
        selected_logits = self.classifier(selected_features).reshape(batch, views, 2)
        if self.resizenet is None:
            resized_features = selected_features
            resized_logits = selected_logits
        else:
            resized_views = self.resizenet(
                padded.reshape(batch * views, 1, self.input_capacity, points)
            )
            resized_features = self.feature_extractor(resized_views)
            resized_logits = self.classifier(resized_features).reshape(batch, views, 2)
        return {
            'resized_logits': resized_logits.mean(dim=1),
            'selected_logits': selected_logits.mean(dim=1),
            'resized_features': resized_features.reshape(batch, views, -1).mean(dim=1),
            'selected_features': selected_features.reshape(batch, views, -1).mean(dim=1),
        }

    def forward(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_details(eeg, channel_mask)['resized_logits']
        return logits[:, 1] - logits[:, 0]

    def cross_modal_adapter_contract(self) -> dict[str, object]:
        return {
            'policy': 'native_electrode_set_attention',
            'model_native_adapter': 'CST_TransformerResizeNet',
            'input_channel_capacity': self.input_capacity,
            'output_channels': self.output_channels,
            'preserves_native_electrode_axis_at_input': True,
            'truncates_native_electrodes': False,
            'uses_target_labels': False,
            'uses_target_test_statistics': False,
            'source_training': 'source_only_with_selection_path_distillation',
            'budget_training': 'labeled_target_msa_with_source_rehearsal',
            'target_inference': 'frozen_checkpoint_single_final_evaluation',
            'ea_policy': 'shared_per_clip_channel_zscore_no_target_test_fit',
        }

    def model_information(self) -> dict[str, object]:
        return {
            'Architecture': 'ResizeNet plus EEGNet plus multi-space alignment',
            'ResizeNet enabled': self.cross_modal,
            'ResizeNet input capacity': self.input_capacity,
            'ResizeNet output channels': self.output_channels,
            'ResizeNet layers': self.resize_layers,
            'ResizeNet heads': self.resize_heads,
            'ResizeNet feedforward': self.resize_feedforward,
            'Feature extractor': 'EEGNet F1=8 D=2 F2=16',
            'View length': '1 second at 128 Hz',
            'View aggregation': 'mean logit across deterministic views',
            'Target test statistics used': False,
        }


def minimum_class_confusion(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError('MCC requires batch by two-class logits')
    if logits.shape[0] < 2:
        return logits.sum() * 0.0
    probabilities = torch.softmax(logits / temperature, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
    weights = 1.0 + torch.exp(-entropy)
    weights = weights * (logits.shape[0] / weights.sum().clamp_min(1e-8))
    confusion = (probabilities * weights[:, None]).transpose(0, 1) @ probabilities
    confusion = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (confusion.sum() - torch.trace(confusion)) / confusion.shape[0]


def knowledge_distillation(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = F.log_softmax(student_logits / temperature, dim=1)
    teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(student, teacher, reduction='batchmean') * temperature**2


def make_training_objective(args: argparse.Namespace):
    def objective(model, batch, eeg, channel_mask, labels, stage):
        details = model.forward_details(eeg, channel_mask)
        resized_logits = details['resized_logits']
        binary_logits = resized_logits[:, 1] - resized_logits[:, 0]
        classification = F.cross_entropy(resized_logits, labels.long())
        zero = classification.detach() * 0.0
        if not model.cross_modal:
            kd_loss = zero
            mcc_loss = zero
        else:
            kd_loss = knowledge_distillation(
                resized_logits,
                details['selected_logits'],
                args.cst_kd_temperature,
            )
            target_rows = torch.tensor(
                [
                    model._dataset_key(dataset) == model.target_dataset
                    for dataset in batch['dataset']
                ],
                dtype=torch.bool,
                device=resized_logits.device,
            )
            mcc_loss = (
                minimum_class_confusion(
                    resized_logits[target_rows], args.cst_mcc_temperature
                )
                if target_rows.any()
                else zero
            )
        loss = (
            classification
            + args.cst_kd_weight * kd_loss
            + args.cst_mcc_weight * mcc_loss
        )
        return binary_logits, loss, {
            'classification_loss': classification.detach(),
            'knowledge_distillation_loss': kd_loss.detach(),
            'minimum_class_confusion_loss': mcc_loss.detach(),
        }

    return objective


def model_factory(args: argparse.Namespace):
    def factory(contract) -> CSTTransferModel:
        return CSTTransferModel(
            contract,
            resize_heads=args.cst_resize_heads,
            resize_layers=args.cst_resize_layers,
            resize_feedforward=args.cst_resize_feedforward,
        )

    return factory


def load_pretrained(model, path: Path, contract) -> dict[str, object]:
    raise ValueError('CST does not define a pretrained checkpoint')


def transfer_configurator(model: CSTTransferModel, mode: str) -> None:
    if mode not in {'SourceOnly', 'FT'}:
        raise ValueError(f'Unsupported CST mode: {mode}')


def forward_clip(
    model: CSTTransferModel,
    eeg: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    return model(eeg, channel_mask)


def budget_module_resolver(model: CSTTransferModel):
    return model.classifier, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='CST ResizeNet and multi-space alignment transfer baseline'
    )
    parser.add_argument('--model', choices=['CST'], default='CST')
    add_common_transfer_arguments(parser)
    parser.add_argument('--cst-resize-heads', type=int, default=2)
    parser.add_argument('--cst-resize-layers', type=int, default=2)
    parser.add_argument('--cst-resize-feedforward', type=int, default=128)
    parser.add_argument('--cst-kd-weight', type=float, default=1.0)
    parser.add_argument('--cst-mcc-weight', type=float, default=1.0)
    parser.add_argument('--cst-kd-temperature', type=float, default=4.0)
    parser.add_argument('--cst-mcc-temperature', type=float, default=2.5)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.use_pretrained:
        raise ValueError('CST has no official pretrained checkpoint')
    if args.cst_resize_heads != 2 or args.cst_resize_layers != 2:
        raise ValueError('The paper-defined CST ResizeNet requires two heads and two layers')
    if args.cst_resize_feedforward <= 0:
        raise ValueError('CST ResizeNet feedforward width must be positive')
    for name in ('cst_kd_weight', 'cst_mcc_weight'):
        if getattr(args, name) < 0.0:
            raise ValueError(f'{name} must be non-negative')
    for name in ('cst_kd_temperature', 'cst_mcc_temperature'):
        if getattr(args, name) <= 0.0:
            raise ValueError(f'{name} must be positive')


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    run_torch_transfer(
        args,
        model_name='CST',
        input_spec_name='cst',
        model_factory=model_factory(args),
        pretrained_loader=load_pretrained,
        transfer_configurator=transfer_configurator,
        forward_clip=forward_clip,
        budget_module_resolver=budget_module_resolver,
        handles_native_electrodes=True,
        model_arguments={
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
        },
        training_objective=make_training_objective(args),
    )


if __name__ == '__main__':
    main()
