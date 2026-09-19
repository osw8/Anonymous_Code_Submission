from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


class BenchmarkTSMNet(nn.Module):
    def __init__(self, channel_count: int, temporal_filters: int = 24, subspace_dim: int = 24) -> None:
        super().__init__()
        self.channel_count = int(channel_count)
        self.temporal = nn.Sequential(
            nn.Conv1d(self.channel_count, temporal_filters, kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(temporal_filters),
            nn.ELU(),
            nn.Conv1d(temporal_filters, subspace_dim, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(subspace_dim),
            nn.ELU(),
        )
        covariance_dim = subspace_dim * (subspace_dim + 1) // 2
        self.classifier = nn.Sequential(
            nn.Linear(covariance_dim + subspace_dim, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def model_information(self) -> dict[str, object]:
        return {
            'Baseline family': 'TSMNet',
            'Input adapter': 'standardized_clip_to_temporal_spatial_covariance',
            'Covariance head': 'log_diagonal_plus_upper_triangular_spd_features',
        }

    @staticmethod
    def _signal(eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        signal = eeg * channel_mask[:, None, :, None].to(eeg.dtype)
        batch, views, channels, points = signal.shape
        return signal.permute(0, 2, 1, 3).reshape(batch, channels, views * points)

    def forward(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        features = self.temporal(self._signal(eeg, channel_mask))
        centered = features - features.mean(dim=-1, keepdim=True)
        covariance = centered @ centered.transpose(1, 2)
        covariance = covariance / max(centered.shape[-1] - 1, 1)
        covariance = covariance + torch.eye(covariance.shape[-1], device=covariance.device, dtype=covariance.dtype)[None] * 1e-4
        triangular = torch.triu_indices(covariance.shape[-1], covariance.shape[-1], device=covariance.device)
        spd_features = covariance[:, triangular[0], triangular[1]]
        diag_features = torch.log(torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-6))
        return self.classifier(torch.cat((spd_features, diag_features), dim=1)).reshape(-1)


def build_model(contract):
    return BenchmarkTSMNet(len(contract.channel_keys))


def load_pretrained(model, path, contract):
    return {'status': 'not_applicable', 'path': str(path)}


def configure_transfer(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = True


def forward_clip(model, eeg, channel_mask):
    return model(eeg, channel_mask)


def budget_modules(model):
    return model.classifier, model.temporal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='TSMNet benchmark adapter')
    parser.add_argument('--model', choices=['TSMNet'], default='TSMNet')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    run_torch_transfer(
        build_parser().parse_args(),
        model_name='TSMNet',
        input_spec_name='steegformer',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        budget_module_resolver=budget_modules,
        model_arguments={
            'tsmnet_temporal_filters': 24,
            'tsmnet_subspace_dim': 24,
            'tsmnet_input_adapter': 'steegformer_continuous_views_to_temporal_spatial_covariance',
            'tsmnet_covariance_features': 'upper_triangular_plus_log_diagonal',
        },
    )
