from __future__ import annotations

# Adapts the model to the shared benchmark contract.
import argparse
import contextlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
EEGPT_ROOT = Path(__file__).resolve().parent
for value in (BENCHMARK_ROOT, EEGPT_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from downstream_tueg.Modules.models.EEGPT_mcae_finetune_change import EEGPTClassifier
from eeg_benchmark.engine import ChannelUnionContract
from eeg_benchmark.engine import disable_stochastic_layers, load_shape_compatible_state
from eeg_benchmark.engine import add_common_transfer_arguments, run_torch_transfer


EEGPT_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'T7', 'C3',
    'CZ', 'C4', 'T8', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'O2',
]


def build_model(contract: ChannelUnionContract) -> EEGPTClassifier:
    forward_chunk_size = int(os.environ.get('EEGPT_FORWARD_CHUNK_SIZE', '64'))
    use_activation_checkpointing = bool(
        int(os.environ.get('EEGPT_ACTIVATION_CHECKPOINTING', '0'))
    )
    model = EEGPTClassifier(
        num_classes=1,
        in_channels=len(contract.channel_keys),
        img_size=[len(EEGPT_CHANNELS), 2000],
        use_channels_names=EEGPT_CHANNELS,
        use_chan_conv=True,
        use_mean_pooling=True,
    )
    disable_stochastic_layers(model)
    model.forward_chunk_size = int(forward_chunk_size)
    model.use_activation_checkpointing = use_activation_checkpointing

    def model_information() -> dict[str, object]:
        return {
            'EEGPT forward chunk size': (
                'full_flat_batch'
                if model.forward_chunk_size <= 0
                else model.forward_chunk_size
            ),
            'EEGPT activation checkpointing': model.use_activation_checkpointing,
            'EEGPT memory policy': 'step_level_weight_constraint_safe_chunking',
        }

    model.model_information = model_information
    return model


def load_pretrained(model, path: Path, contract: ChannelUnionContract) -> dict[str, object]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    report = load_shape_compatible_state(model, state)
    report['input_adapter_policy'] = 'random_union_channel_adapter_trained_on_source'
    return report


def configure_transfer(model: EEGPTClassifier, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def budget_modules(model: EEGPTClassifier) -> tuple[torch.nn.Module, torch.nn.Module]:
    return model.head, model.target_encoder.blocks[-1]


def prepare_target_input(model: EEGPTClassifier, contract: ChannelUnionContract) -> None:
    union_index = {name: index for index, name in enumerate(contract.channel_keys)}
    source_indices = [union_index[name] for name in contract.source_channel_keys]
    target_only = [
        union_index[name]
        for name in contract.target_channel_keys
        if name not in set(contract.source_channel_keys)
    ]
    if not source_indices or not target_only:
        return
    first_conv = model.chan_conv[0]
    with torch.no_grad():
        source_mean = first_conv.weight[:, source_indices].mean(dim=1, keepdim=True)
        first_conv.weight[:, target_only] = source_mean


def apply_weight_constraints_once(model: EEGPTClassifier) -> list[torch.nn.Module]:
    constrained_modules = []
    with torch.no_grad():
        for module in model.modules():
            if not bool(getattr(module, 'doWeightNorm', False)):
                continue
            weight = getattr(module, 'weight', None)
            max_norm = getattr(module, 'max_norm', None)
            if weight is None or max_norm is None:
                continue
            weight.renorm_(p=2, dim=0, maxnorm=max_norm)
            constrained_modules.append(module)
    return constrained_modules


@contextlib.contextmanager
def suspend_forward_weight_constraints(modules: list[torch.nn.Module]):
    original_flags = [bool(getattr(module, 'doWeightNorm', False)) for module in modules]
    try:
        for module in modules:
            module.doWeightNorm = False
        yield
    finally:
        for module, flag in zip(modules, original_flags):
            module.doWeightNorm = flag


def forward_clip(model: EEGPTClassifier, eeg: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    batch_size, views, channels, points = eeg.shape
    flat = eeg.reshape(batch_size * views, channels, points)
    configured_chunk_size = int(getattr(model, 'forward_chunk_size', 8))
    chunk_size = flat.shape[0] if configured_chunk_size <= 0 else max(1, configured_chunk_size)
    use_checkpointing = bool(
        getattr(model, 'use_activation_checkpointing', True)
        and model.training
        and torch.is_grad_enabled()
    )
    outputs = []
    constrained_modules = (
        apply_weight_constraints_once(model)
        if model.training and torch.is_grad_enabled()
        else []
    )
    with suspend_forward_weight_constraints(constrained_modules):
        for chunk in flat.split(chunk_size, dim=0):
            if use_checkpointing:
                outputs.append(
                    checkpoint(
                        model,
                        chunk,
                        use_reentrant=False,
                    )
                )
            else:
                outputs.append(model(chunk))
    logits = torch.cat(outputs, dim=0).reshape(batch_size, views)
    return logits.mean(dim=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='EEGPT Task1 60s clip transfer')
    parser.add_argument('--model', choices=['EEGPT'], default='EEGPT')
    return add_common_transfer_arguments(parser)


if __name__ == '__main__':
    args = build_parser().parse_args()
    run_torch_transfer(
        args,
        model_name='EEGPT',
        input_spec_name='eegpt_tueg',
        model_factory=build_model,
        pretrained_loader=load_pretrained,
        transfer_configurator=configure_transfer,
        forward_clip=forward_clip,
        target_input_adapter=prepare_target_input,
        budget_module_resolver=budget_modules,
        runtime_arguments={
            'eegpt_forward_chunk_size': int(
                os.environ.get('EEGPT_FORWARD_CHUNK_SIZE', '64')
            ),
            'eegpt_activation_checkpointing': bool(
                int(os.environ.get('EEGPT_ACTIVATION_CHECKPOINTING', '0'))
            ),
            'eegpt_memory_policy': 'step_level_weight_constraint_safe_chunking',
        },
    )
