from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
BFEML_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, BFEML_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


try:
    from models import EML
    ORIGINAL_IMPORT_ERROR = None
except Exception as error:  
    EML = None
    ORIGINAL_IMPORT_ERROR = repr(error)


class CompactEML(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.view0 = nn.Sequential(nn.Flatten(), nn.Linear(23 * 256, 128), nn.ELU())
        self.view1 = nn.Sequential(nn.Flatten(), nn.Linear(23 * 27, 64), nn.ELU())
        self.view2 = nn.Sequential(nn.Flatten(), nn.Linear(23 * 14 * 256, 128), nn.ELU())
        self.classifier = nn.Linear(320, 2)

    def forward(self, inputs: dict[int, torch.Tensor]):
        features = torch.cat((self.view0(inputs[0]), self.view1(inputs[1]), self.view2(inputs[2])), dim=1)
        return self.classifier(features)


class BenchmarkBFEML(nn.Module):
    def __init__(self, channel_count: int) -> None:
        super().__init__()
        self.channel_count = int(channel_count)
        self.channel_projection = nn.Conv1d(self.channel_count, 23, kernel_size=1, bias=False)
        with torch.no_grad():
            self.channel_projection.weight.zero_()
            diagonal = min(self.channel_count, 23)
            index = torch.arange(diagonal)
            self.channel_projection.weight[index, index, 0] = 1.0
        self.backbone = (
            EML(
                sample_shapes=[(23, 256), (23, 27), (23, 14, 256)],
                num_classes=2,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            if EML is not None
            else CompactEML()
        )
        self.head = self.backbone

    def model_information(self) -> dict[str, object]:
        return {
            'Baseline family': 'BF-EML',
            'Input adapter': 'standardized_clip_to_three_original_bfeml_views',
            'Original implementation loaded': EML is not None,
            'Original import error': ORIGINAL_IMPORT_ERROR,
        }

    def _project_signal(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        signal = eeg * channel_mask[:, None, :, None].to(eeg.dtype)
        batch, views, channels, points = signal.shape
        signal = signal.permute(0, 2, 1, 3).reshape(batch, channels, views * points)
        return self.channel_projection(signal)

    def _views(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> dict[int, torch.Tensor]:
        signal = self._project_signal(eeg, channel_mask)
        view0 = F.interpolate(signal, size=256, mode='linear', align_corners=False)
        view1 = F.interpolate(signal.abs(), size=27, mode='linear', align_corners=False)
        view2 = F.interpolate(signal, size=14 * 256, mode='linear', align_corners=False).reshape(signal.shape[0], 23, 14, 256)
        return {
            0: view0.reshape(signal.shape[0], -1),
            1: view1.reshape(signal.shape[0], -1),
            2: view2.reshape(signal.shape[0], -1),
        }

    def forward(self, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        inputs = self._views(eeg, channel_mask)
        if EML is not None:
            view_e, fusion_e, _, _ = self.backbone(inputs, target=None)
            del view_e
            logits = fusion_e
        else:
            logits = self.backbone(inputs)
        if logits.shape[-1] != 2:
            raise ValueError(f'BF-EML expected binary evidence, got {tuple(logits.shape)}')
        return logits[:, 1] - logits[:, 0]


def build_model(contract):
    return BenchmarkBFEML(len(contract.channel_keys))


def load_pretrained(model, path, contract):
    return {'status': 'not_applicable', 'path': str(path)}


def configure_transfer(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad = True


def forward_clip(model, eeg, channel_mask):
    return model(eeg, channel_mask)


def budget_modules(model):
    return model.head, model.channel_projection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='BF-EML benchmark adapter')
    parser.add_argument('--model', choices=['BF-EML'], default='BF-EML')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    run_torch_transfer(
        build_parser().parse_args(),
        model_name='BF-EML',
        input_spec_name='steegformer',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        budget_module_resolver=budget_modules,
        model_arguments={
            'bfeml_input_adapter': 'steegformer_continuous_views_to_original_bfeml_views',
            'bfeml_view0_shape': [23, 256],
            'bfeml_view1_shape': [23, 27],
            'bfeml_view2_shape': [23, 14, 256],
            'bfeml_original_loss_reused': False,
            'bfeml_runtime_loss': 'binary_cross_entropy_with_logits',
        },
    )
