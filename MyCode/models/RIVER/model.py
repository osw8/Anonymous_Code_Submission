from __future__ import annotations

# Defines the RIVER architecture.
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


FREQUENCY_BANDS_HZ = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 45.0),
    (45.0, 65.0),
    (65.0, 95.0),
)


@dataclass(frozen=True)
class EpiTransOpConfig:
    sampling_frequency: int = 256
    patch_points: int = 256
    descriptor_dim: int = 16
    descriptor_hidden_dim: int = 64
    descriptor_window_seconds: float = 4.0
    descriptor_slow_seconds: float = 16.0
    descriptor_channel_chunk_size: int = 16384
    latent_electrodes: int = 12
    embed_dim: int = 128
    depth: int = 6
    heads: int = 8
    ff_dim: int = 384
    fourier_modes: int = 32
    fourier_rank: int = 8
    local_kernel_size: int = 5
    slot_temperature: float = 0.50
    sinkhorn_iterations: int = 4
    transport_epsilon: float = 0.20
    electrode_mass_relaxation: float = 0.80
    electrode_mass_temperature: float = 0.50
    pooling_temperature: float = 0.50
    evidence_smoothing_kernel: int = 3
    dropout: float = 0.0
    adapter_mode: str = 'temporal_uot'
    temporal_mixer: str = 'spectral_local'
    task: str = 'detection'
    mass_aware_transport: bool = True
    residual_bottleneck_dim: int = 32

    def __post_init__(self) -> None:
        integer_fields = {
            'sampling_frequency': self.sampling_frequency,
            'patch_points': self.patch_points,
            'descriptor_dim': self.descriptor_dim,
            'descriptor_hidden_dim': self.descriptor_hidden_dim,
            'descriptor_channel_chunk_size': self.descriptor_channel_chunk_size,
            'latent_electrodes': self.latent_electrodes,
            'embed_dim': self.embed_dim,
            'depth': self.depth,
            'heads': self.heads,
            'ff_dim': self.ff_dim,
            'fourier_modes': self.fourier_modes,
            'fourier_rank': self.fourier_rank,
            'local_kernel_size': self.local_kernel_size,
            'sinkhorn_iterations': self.sinkhorn_iterations,
            'evidence_smoothing_kernel': self.evidence_smoothing_kernel,
            'residual_bottleneck_dim': self.residual_bottleneck_dim,
        }
        invalid = [name for name, value in integer_fields.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f'Positive integer model fields required: {invalid}')
        if self.embed_dim % self.heads:
            raise ValueError('embed_dim must be divisible by heads')
        if self.descriptor_dim != 16:
            raise ValueError('The dynamic descriptor contract has exactly 16 features')
        if self.local_kernel_size % 2 == 0:
            raise ValueError('local_kernel_size must be odd')
        if self.evidence_smoothing_kernel % 2 == 0:
            raise ValueError('evidence_smoothing_kernel must be odd')
        if self.descriptor_window_seconds <= 0.0 or self.descriptor_slow_seconds <= 0.0:
            raise ValueError('Descriptor scales must be positive')
        if not 0.0 < self.slot_temperature <= 2.0:
            raise ValueError('slot_temperature must be in (0, 2]')
        if not 0.0 < self.transport_epsilon <= 2.0:
            raise ValueError('transport_epsilon must be in (0, 2]')
        if not 0.0 < self.electrode_mass_relaxation <= 1.0:
            raise ValueError('electrode_mass_relaxation must be in (0, 1]')
        if self.electrode_mass_temperature <= 0.0:
            raise ValueError('electrode_mass_temperature must be positive')
        if self.pooling_temperature <= 0.0:
            raise ValueError('pooling_temperature must be positive')
        if self.dropout != 0.0:
            raise ValueError('EpiTransOp requires dropout=0 for deterministic experiments')
        if self.adapter_mode not in {'temporal_uot', 'static_uot', 'query_attention'}:
            raise ValueError(
                'adapter_mode must be temporal_uot, static_uot, or query_attention'
            )
        if self.temporal_mixer not in {'spectral_local', 'axial_attention'}:
            raise ValueError('temporal_mixer must be spectral_local or axial_attention')
        if self.task not in {'detection', 'prediction'}:
            raise ValueError('task must be detection or prediction')
        if self.residual_bottleneck_dim <= 0 or self.residual_bottleneck_dim > self.embed_dim:
            raise ValueError('residual_bottleneck_dim must be in (0, embed_dim]')


class SharedChannelPatchEncoder(nn.Module):
    def __init__(self, patch_points: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_points = int(patch_points)
        temporal_bins = int(math.ceil(self.patch_points / 16))
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 12, kernel_size=33, stride=16, padding=16, bias=False),
            nn.GroupNorm(4, 12),
            nn.GELU(),
            nn.Flatten(start_dim=1),
            nn.Linear(12 * temporal_bins, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 4:
            raise ValueError(
                f'Patch encoder expects [batch, channel, patch, point], got {tuple(signal.shape)}'
            )
        batch, channels, patches, points = signal.shape
        if points != self.patch_points:
            raise ValueError(
                f'Expected {self.patch_points} points per patch, received {points}'
            )
        flat = signal.reshape(batch * channels * patches, 1, points)
        encoded = self.encoder(flat)
        return self.output_norm(encoded).reshape(batch, channels, patches, -1)


def _temporal_average(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    patches = int(values.shape[2])
    kernel = min(max(int(kernel_size), 1), patches)
    if kernel <= 1:
        return values
    if kernel % 2 == 0:
        kernel -= 1
    flattened = values.permute(0, 1, 3, 2).reshape(-1, values.shape[-1], patches)
    smoothed = F.avg_pool1d(
        flattened,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
        count_include_pad=False,
    )
    return smoothed.reshape(
        values.shape[0], values.shape[1], values.shape[-1], patches
    ).permute(0, 1, 3, 2)


class DynamicElectrodeStateEncoder(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.descriptor_hidden_dim
        self.descriptor_encoder = nn.Sequential(
            nn.LayerNorm(3 * config.descriptor_dim),
            nn.Linear(3 * config.descriptor_dim, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.learned_state_encoder = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.fusion_gate = nn.Linear(2 * hidden, hidden)
        self.fused_norm = nn.LayerNorm(hidden)

    @staticmethod
    def _safe_standardize(signal: torch.Tensor) -> torch.Tensor:
        centered = signal - signal.mean(dim=-1, keepdim=True)
        scale = centered.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        return centered / scale

    def _patch_power(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        points = int(signal.shape[-1])
        window = torch.hann_window(
            points,
            periodic=True,
            device=signal.device,
            dtype=signal.dtype,
        )
        scale = window.square().sum().clamp_min(1e-8)
        flattened = signal.reshape(-1, points)
        chunks = []
        chunk_size = int(self.config.descriptor_channel_chunk_size)
        for start in range(0, flattened.shape[0], chunk_size):
            selected = flattened[start:start + chunk_size]
            spectrum = torch.fft.rfft(selected * window, dim=-1)
            chunks.append(spectrum.abs().square() / scale)
        power = torch.cat(chunks, dim=0).reshape(*signal.shape[:-1], -1)
        frequency = torch.fft.rfftfreq(
            points,
            d=1.0 / float(self.config.sampling_frequency),
            device=signal.device,
        ).to(dtype=signal.dtype)
        return power, frequency

    def signal_descriptors(
        self,
        signal: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if signal.ndim != 4:
            raise ValueError('Electrode descriptors require patched channel signals')
        batch, channels, _, _ = signal.shape
        if channel_mask.shape != (batch, channels):
            raise ValueError('Channel mask does not match descriptor input')
        mask = channel_mask.to(device=signal.device, dtype=torch.bool)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError('Every clip must contain at least one valid electrode')

        with torch.autocast(device_type=signal.device.type, enabled=False):
            standardized = self._safe_standardize(signal.detach().float())
            first_difference = standardized[..., 1:] - standardized[..., :-1]
            second_difference = first_difference[..., 1:] - first_difference[..., :-1]
            variance = standardized.square().mean(dim=-1).clamp_min(1e-8)
            first_variance = first_difference.var(dim=-1, unbiased=False).clamp_min(1e-8)
            second_variance = second_difference.var(dim=-1, unbiased=False).clamp_min(1e-8)
            mobility = (first_variance / variance).sqrt().clamp_max(10.0)
            derivative_mobility = (second_variance / first_variance).sqrt()
            complexity = (derivative_mobility / mobility.clamp_min(1e-4)).clamp_max(10.0)
            skewness = standardized.pow(3).mean(dim=-1).clamp(-10.0, 10.0)
            kurtosis = (standardized.pow(4).mean(dim=-1) - 3.0).clamp(-10.0, 50.0)
            lag_one = (
                standardized[..., 1:] * standardized[..., :-1]
            ).mean(dim=-1).clamp(-1.0, 1.0)

            power, frequency = self._patch_power(standardized)
            selected = (frequency >= 0.5) & (frequency < 95.0)
            selected_power = power[..., selected].clamp_min(1e-12)
            total_power = selected_power.sum(dim=-1).clamp_min(1e-12)
            probability = selected_power / total_power.unsqueeze(-1)
            spectral_entropy = -(
                probability * probability.clamp_min(1e-12).log()
            ).sum(dim=-1) / math.log(max(int(selected.sum()), 2))
            band_power = []
            for low_hz, high_hz in FREQUENCY_BANDS_HZ:
                band = (frequency >= low_hz) & (frequency < high_hz)
                value = (
                    power[..., band].sum(dim=-1) / total_power
                    if torch.any(band)
                    else torch.zeros_like(total_power)
                )
                band_power.append(value)

            valid = mask.to(dtype=standardized.dtype)
            valid_count = valid.sum(dim=1, keepdim=True)
            signal_sum = (standardized * valid[:, :, None, None]).sum(
                dim=1, keepdim=True
            )
            leave_one_out = (signal_sum - standardized) / (
                valid_count[:, :, None, None] - 1.0
            ).clamp_min(1.0)
            single_reference = signal_sum / valid_count[:, :, None, None].clamp_min(1.0)
            reference = torch.where(
                (valid_count > 1.0)[:, :, None, None],
                leave_one_out,
                single_reference,
            )
            reference_correlation = (
                standardized * self._safe_standardize(reference)
            ).mean(dim=-1).clamp(-1.0, 1.0)

            descriptors = torch.stack(
                (
                    first_difference.abs().mean(dim=-1),
                    (
                        standardized[..., 1:] * standardized[..., :-1] < 0
                    ).to(standardized.dtype).mean(dim=-1),
                    mobility,
                    complexity,
                    skewness,
                    kurtosis,
                    lag_one,
                    spectral_entropy,
                    *band_power,
                    reference_correlation,
                ),
                dim=-1,
            )
            descriptors = descriptors.masked_fill(~mask[:, :, None, None], 0.0)
        if descriptors.shape[-1] != self.config.descriptor_dim:
            raise RuntimeError('Electrode descriptor dimension changed unexpectedly')
        if not torch.isfinite(descriptors).all():
            raise ValueError('Electrode descriptors contain non-finite values')
        return descriptors.to(dtype=signal.dtype)

    def forward(
        self,
        channel_tokens: torch.Tensor,
        descriptors: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        mesoscopic_patches = max(1, int(round(self.config.descriptor_window_seconds)))
        slow_patches = max(1, int(round(self.config.descriptor_slow_seconds)))
        multiscale = torch.cat(
            (
                descriptors,
                _temporal_average(descriptors, mesoscopic_patches),
                _temporal_average(descriptors, slow_patches),
            ),
            dim=-1,
        )
        physical_state = self.descriptor_encoder(multiscale)
        learned_state = self.learned_state_encoder(channel_tokens)
        gate = torch.sigmoid(
            self.fusion_gate(torch.cat((physical_state, learned_state), dim=-1))
        )
        fused = self.fused_norm(
            gate * physical_state + (1.0 - gate) * learned_state
        )
        return fused.masked_fill(~channel_mask[:, :, None, None], 0.0)


class ObservabilityGuidedTemporalTransport(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.descriptor_hidden_dim
        self.state_encoder = DynamicElectrodeStateEncoder(config)
        self.relation_norm = nn.LayerNorm(hidden)
        self.relation_query = nn.Linear(hidden, hidden, bias=False)
        self.relation_key = nn.Linear(hidden, hidden, bias=False)
        self.relation_value = nn.Linear(hidden, hidden, bias=False)
        self.relation_output = nn.Linear(hidden, hidden, bias=False)
        self.transport_key = nn.Linear(hidden, hidden, bias=False)
        self.transport_query = nn.Linear(hidden, hidden, bias=False)
        self.transport_value = nn.Linear(config.embed_dim, config.embed_dim, bias=False)
        self.observability = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.continuity_gate = nn.Sequential(
            nn.Linear(1, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.latent_queries = nn.Parameter(
            torch.empty(config.latent_electrodes, hidden)
        )
        nn.init.orthogonal_(self.latent_queries)

    def signal_descriptors(
        self,
        signal: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.state_encoder.signal_descriptors(signal, channel_mask)

    def _relational_context(
        self,
        state: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = state.shape[-1]
        normalized = self.relation_norm(state)
        query = self.relation_query(normalized)
        key = self.relation_key(normalized)
        value = self.relation_value(normalized)
        logits = torch.einsum('bpch,bpdh->bpcd', query, key) / math.sqrt(float(hidden))
        key_mask = channel_mask[:, None, None, :]
        relation = torch.softmax(logits.masked_fill(~key_mask, -1e4), dim=-1)
        relation = relation.masked_fill(~key_mask, 0.0)
        context = torch.einsum('bpcd,bpdh->bpch', relation, value)
        state = state + self.relation_output(context)
        state = state.masked_fill(~channel_mask[:, None, :, None], 0.0)
        return state, relation

    def _temporal_cost(self, state: torch.Tensor) -> torch.Tensor:
        keys = F.normalize(self.transport_key(state), dim=-1)
        queries = F.normalize(self.transport_query(self.latent_queries), dim=-1)
        similarity = torch.einsum('kh,bpch->bpkc', queries, keys)
        cost = (1.0 - similarity).clamp_min(0.0)
        if self.config.adapter_mode == 'static_uot':
            return cost.mean(dim=1, keepdim=True).expand_as(cost)
        if cost.shape[1] <= 1:
            return cost
        neighbor = F.avg_pool2d(
            cost.permute(0, 2, 1, 3),
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
            count_include_pad=False,
        ).permute(0, 2, 1, 3)
        change = torch.zeros(
            state.shape[0], state.shape[1], 1,
            device=state.device,
            dtype=state.dtype,
        )
        change[:, 1:] = (state[:, 1:] - state[:, :-1]).square().mean(
            dim=(-1, -2)
        ).sqrt().unsqueeze(-1)
        raw_gate = torch.sigmoid(self.continuity_gate(change))[:, :, None]
        return raw_gate * cost + (1.0 - raw_gate) * neighbor

    def _unbalanced_transport(
        self,
        cost: torch.Tensor,
        prior_mass: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.adapter_mode == 'query_attention':
            assignment = torch.softmax(
                (-cost / self.config.slot_temperature).masked_fill(
                    ~channel_mask[:, None, None, :], -1e4
                ),
                dim=-1,
            )
            assignment = assignment.masked_fill(
                ~channel_mask[:, None, None, :], 0.0
            )
            raw_plan = assignment / float(self.config.latent_electrodes)
            return assignment, raw_plan

        epsilon = float(self.config.transport_epsilon)
        relaxation = float(self.config.electrode_mass_relaxation)
        latent_count = int(cost.shape[-2])
        log_kernel = torch.nan_to_num(
            -cost.float() / epsilon,
            nan=-1e4,
            posinf=1e4,
            neginf=-1e4,
        ).clamp(-1e4, 1e4)
        log_a = cost.new_full(
            (*cost.shape[:-2], latent_count),
            -math.log(float(latent_count)),
            dtype=torch.float32,
        )
        log_b = prior_mass.float().clamp_min(1e-8).log()
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(self.config.sinkhorn_iterations):
            log_u = relaxation * (
                log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(-2), dim=-1)
            )
            log_u = torch.nan_to_num(log_u, nan=0.0, posinf=0.0, neginf=0.0)
            log_v = relaxation * (
                log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(-1), dim=-2)
            )
            log_v = torch.nan_to_num(log_v, nan=0.0, posinf=0.0, neginf=0.0)
        raw_plan = torch.exp(
            log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2)
        ).masked_fill(~channel_mask[:, None, None, :], 0.0)
        raw_plan = torch.nan_to_num(raw_plan, nan=0.0, posinf=0.0, neginf=0.0)
        row_mass = raw_plan.sum(dim=-1, keepdim=True)
        valid_count = channel_mask.sum(dim=-1).clamp_min(1).to(dtype=torch.float32)
        fallback = (
            channel_mask[:, None, None, :].to(dtype=torch.float32)
            / valid_count[:, None, None, None]
        )
        assignment = torch.where(
            row_mass > 1e-8,
            raw_plan / row_mass.clamp_min(1e-8),
            fallback.expand_as(raw_plan),
        )
        return assignment.to(dtype=cost.dtype), raw_plan.to(dtype=cost.dtype)

    def forward(
        self,
        channel_tokens: torch.Tensor,
        descriptors: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self.state_encoder(channel_tokens, descriptors, channel_mask)
        state_by_patch = state.permute(0, 2, 1, 3)
        state_by_patch, relation = self._relational_context(
            state_by_patch, channel_mask
        )
        cost = self._temporal_cost(state_by_patch)
        prior_logits = self.observability(state_by_patch).squeeze(-1)
        prior_mass = torch.softmax(
            torch.nan_to_num(
                prior_logits.float() / self.config.electrode_mass_temperature,
                nan=-1e4,
                posinf=1e4,
                neginf=-1e4,
            ).clamp(-1e4, 1e4).masked_fill(~channel_mask[:, None, :], -1e4),
            dim=-1,
        ).masked_fill(~channel_mask[:, None, :], 0.0)
        assignment, raw_plan = self._unbalanced_transport(
            cost, prior_mass, channel_mask
        )
        transported = torch.einsum(
            'bpkc,bcpd->bkpd', assignment, self.transport_value(channel_tokens)
        )
        column_mass = raw_plan.sum(dim=2)
        column_mass = column_mass / column_mass.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        column_mass = torch.nan_to_num(column_mass, nan=0.0, posinf=0.0, neginf=0.0)
        slot_state = torch.einsum('bpkc,bpch->bpkh', assignment, state_by_patch)
        source_mass = raw_plan.sum(dim=-1)
        details = {
            'transport': assignment.mean(dim=1),
            'temporal_transport': assignment,
            'electrode_prior_mass': prior_mass.mean(dim=1),
            'temporal_electrode_prior_mass': prior_mass,
            'transport_column_mass': column_mass.mean(dim=1),
            'temporal_transport_column_mass': column_mass,
            'transport_cost': cost.mean(dim=1),
            'temporal_transport_cost': cost,
            'functional_relation': relation.mean(dim=1),
            'temporal_functional_relation': relation,
            'adapter_slot_state': slot_state.mean(dim=1),
            'descriptors': descriptors,
            'raw_transport_mass': raw_plan.sum(dim=(-1, -2)),
            'temporal_source_mass': source_mass,
            'source_mass': source_mass.mean(dim=1),
        }
        return transported, details

    def contract(self) -> dict[str, Any]:
        implementations = {
            'temporal_uot': 'observability_guided_temporal_unbalanced_transport',
            'static_uot': 'observability_guided_static_unbalanced_transport',
            'query_attention': 'independent_query_attention',
        }
        return {
            'policy': 'native_electrode_set_attention',
            'implementation': implementations[self.config.adapter_mode],
            'adapter_mode': self.config.adapter_mode,
            'mapping_is_time_conditioned': True,
            'uses_pairwise_functional_relations': True,
            'uses_unbalanced_optimal_transport': self.config.adapter_mode != 'query_attention',
            'descriptor_dim': self.config.descriptor_dim,
            'descriptor_scales_seconds': [
                1.0,
                self.config.descriptor_window_seconds,
                self.config.descriptor_slow_seconds,
            ],
            'latent_neural_sources': self.config.latent_electrodes,
            'sinkhorn_iterations': self.config.sinkhorn_iterations,
            'transport_epsilon': self.config.transport_epsilon,
            'uses_target_labels_for_mapping': False,
            'uses_target_test_statistics': False,
            'uses_electrode_coordinates': False,
            'preserves_native_electrode_gradient_path': True,
            'permutation_invariant_clip_prediction': True,
        }


class LowRankGlobalFourierOperator(nn.Module):
    def __init__(self, embed_dim: int, maximum_modes: int, rank: int) -> None:
        super().__init__()
        self.maximum_modes = int(maximum_modes)
        self.v_real = nn.Parameter(torch.empty(maximum_modes, rank, embed_dim))
        self.v_imag = nn.Parameter(torch.empty(maximum_modes, rank, embed_dim))
        self.u_real = nn.Parameter(torch.empty(maximum_modes, embed_dim, rank))
        self.u_imag = nn.Parameter(torch.empty(maximum_modes, embed_dim, rank))
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        for parameter in (self.v_real, self.v_imag, self.u_real, self.u_imag):
            nn.init.normal_(parameter, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_count = int(tokens.shape[2])
        original_dtype = tokens.dtype
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            spectrum = torch.fft.rfft(tokens.float(), dim=2, norm='ortho')
            modes = min(self.maximum_modes, int(spectrum.shape[2]))
            v = torch.complex(self.v_real[:modes], self.v_imag[:modes]).to(spectrum.dtype)
            u = torch.complex(self.u_real[:modes], self.u_imag[:modes]).to(spectrum.dtype)
            compressed = torch.einsum(
                'bkmd,mrd->bkmr', spectrum[:, :, :modes], v.conj()
            )
            mixed_low = torch.einsum('bkmr,mdr->bkmd', compressed, u)
            if modes < spectrum.shape[2]:
                mixed_spectrum = F.pad(
                    mixed_low, (0, 0, 0, int(spectrum.shape[2]) - modes)
                )
            else:
                mixed_spectrum = mixed_low
            mixed = torch.fft.irfft(
                mixed_spectrum, n=patch_count, dim=2, norm='ortho'
            )
        return self.output_projection(mixed.to(dtype=original_dtype))


class LocalTemporalOperator(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, latent_sources, patches, embed_dim = tokens.shape
        temporal = tokens.reshape(
            batch * latent_sources, patches, embed_dim
        ).transpose(1, 2)
        temporal = self.pointwise(F.gelu(self.depthwise(temporal)))
        return temporal.transpose(1, 2).reshape(
            batch, latent_sources, patches, embed_dim
        )


class AdaptiveSpectralLocalOperator(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.global_operator = LowRankGlobalFourierOperator(
            config.embed_dim, config.fourier_modes, config.fourier_rank
        )
        self.local_operator = LocalTemporalOperator(
            config.embed_dim, config.local_kernel_size
        )
        self.scale_gate = nn.Sequential(
            nn.LayerNorm(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_state = self.global_operator(tokens)
        local_state = self.local_operator(tokens)
        gate = self.scale_gate(tokens)
        return gate * local_state + (1.0 - gate) * global_state, gate


class AxialTemporalAttention(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.embed_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, latent_sources, patches, embed_dim = tokens.shape
        temporal = tokens.reshape(batch * latent_sources, patches, embed_dim)
        temporal = self.attention(
            temporal, temporal, temporal, need_weights=False
        )[0].reshape(batch, latent_sources, patches, embed_dim)
        return temporal, torch.full_like(temporal, 0.5)


class LatentNeuralFieldBlock(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.field_norm = nn.LayerNorm(config.embed_dim)
        self.temporal_mixer = (
            AdaptiveSpectralLocalOperator(config)
            if config.temporal_mixer == 'spectral_local'
            else AxialTemporalAttention(config)
        )
        self.source_attention = nn.MultiheadAttention(
            config.embed_dim,
            config.heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        field = self.field_norm(tokens)
        temporal, scale_gate = self.temporal_mixer(field)
        batch, latent_sources, patches, embed_dim = field.shape
        spatial = field.permute(0, 2, 1, 3).reshape(
            batch * patches, latent_sources, embed_dim
        )
        spatial = self.source_attention(
            spatial, spatial, spatial, need_weights=False
        )[0]
        spatial = spatial.reshape(
            batch, patches, latent_sources, embed_dim
        ).permute(0, 2, 1, 3)
        tokens = tokens + (temporal + spatial) / math.sqrt(2.0)
        return tokens + self.ffn(self.ffn_norm(tokens)), scale_gate


class DurationAwareEvidenceHead(nn.Module):
    def __init__(self, config: EpiTransOpConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embed_dim
        bottleneck = config.residual_bottleneck_dim
        self.residual_down = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, bottleneck), nn.GELU()
        )
        self.residual_up = nn.Linear(bottleneck, dim)
        self.residual_gate = nn.Linear(dim, 1)
        self.task_token_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.duration_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.evidence_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.residual_enabled = False

    def _patch_logits(self, tokens: torch.Tensor, head: nn.Module) -> torch.Tensor:
        token_logits = head(tokens).squeeze(-1)
        return torch.logsumexp(token_logits, dim=1) - math.log(
            float(token_logits.shape[1])
        )

    def _accumulate(
        self,
        patch_logits: torch.Tensor,
        duration_weights: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = float(self.config.pooling_temperature)
        peak = temperature * (
            torch.logsumexp(patch_logits / temperature, dim=-1)
            - math.log(float(patch_logits.shape[-1]))
        )
        persistent = (
            duration_weights * patch_logits
        ).sum(dim=-1) / duration_weights.sum(dim=-1).clamp_min(1e-6)
        return gate * peak + (1.0 - gate) * persistent, peak, persistent

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shared = tokens
        residual = self.residual_up(self.residual_down(shared))
        pooled = shared.mean(dim=(1, 2))
        residual_gate = torch.sigmoid(self.residual_gate(pooled))[:, None, None]
        task_field = (
            shared + residual_gate * residual
            if self.residual_enabled else shared
        )
        duration_weights = torch.sigmoid(
            self.duration_head(task_field.mean(dim=1)).squeeze(-1)
        )
        kernel = min(self.config.evidence_smoothing_kernel, duration_weights.shape[-1])
        if kernel % 2 == 0:
            kernel -= 1
        if kernel > 1:
            duration_weights = F.avg_pool1d(
                duration_weights[:, None],
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
                count_include_pad=False,
            ).squeeze(1)
        evidence_gate = torch.sigmoid(
            self.evidence_gate(task_field.mean(dim=(1, 2))).squeeze(-1)
        )
        patch_logits = self._patch_logits(task_field, self.task_token_head)
        logits, peak, persistent = self._accumulate(
            patch_logits, duration_weights, evidence_gate
        )
        return logits, {
            'patch_logits': patch_logits,
            'shared_field': shared,
            'residual_field': residual,
            'task_field': task_field,
            'residual_gate': residual_gate.flatten(1),
            'duration_weights': duration_weights,
            'evidence_gate': evidence_gate,
            'peak_evidence': peak,
            'persistent_evidence': persistent,
        }


def sinusoidal_time_encoding(
    patch_count: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(patch_count, device=device, dtype=torch.float32)[:, None]
    exponent = torch.arange(0, embed_dim, 2, device=device, dtype=torch.float32)
    frequency = torch.exp(-math.log(10000.0) * exponent / max(embed_dim, 1))
    encoding = torch.zeros(patch_count, embed_dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * frequency)
    if embed_dim > 1:
        encoding[:, 1::2] = torch.cos(
            position * frequency[: encoding[:, 1::2].shape[1]]
        )
    return encoding.to(dtype=dtype)


class EpiTransOp(nn.Module):
    def __init__(self, config: EpiTransOpConfig | None = None) -> None:
        super().__init__()
        self.config = config or EpiTransOpConfig()
        self.patch_encoder = SharedChannelPatchEncoder(
            self.config.patch_points, self.config.embed_dim
        )
        self.electrode_transport = ObservabilityGuidedTemporalTransport(self.config)
        self.latent_embedding = nn.Parameter(
            torch.empty(1, self.config.latent_electrodes, 1, self.config.embed_dim)
        )
        self.blocks = nn.ModuleList(
            [LatentNeuralFieldBlock(self.config) for _ in range(self.config.depth)]
        )
        self.final_norm = nn.LayerNorm(self.config.embed_dim)
        self.head = DurationAwareEvidenceHead(self.config)
        self.register_buffer(
            'source_slot_prototypes',
            torch.zeros(2, self.config.latent_electrodes, self.config.embed_dim),
        )
        self.register_buffer('source_slot_initialized', torch.zeros(2, dtype=torch.bool))
        nn.init.normal_(self.latent_embedding, std=0.02)
        self.apply(self._initialize_module)
        nn.init.zeros_(self.head.residual_up.weight)
        nn.init.zeros_(self.head.residual_up.bias)
        nn.init.constant_(
            self.head.residual_gate.bias,
            -1.5 if self.config.task == 'prediction' else -1.0,
        )
        nn.init.constant_(
            self.head.evidence_gate[-1].bias,
            -1.0 if self.config.task == 'prediction' else 1.0,
        )

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    @staticmethod
    def _finite_tensor(tensor: torch.Tensor, limit: float = 1e4) -> torch.Tensor:
        if torch.isfinite(tensor).all():
            return tensor
        return torch.nan_to_num(
            tensor,
            nan=0.0,
            posinf=float(limit),
            neginf=-float(limit),
        ).clamp(min=-float(limit), max=float(limit))

    @classmethod
    def _finite_details(cls, details: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: cls._finite_tensor(value)
            if torch.is_tensor(value) else value
            for key, value in details.items()
        }

    def _merge_views(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 5:
            raise ValueError(
                f'EpiTransOp expects [batch, view, channel, patch, point], got {tuple(eeg.shape)}'
            )
        _, views, _, _, points = eeg.shape
        if views != 1:
            raise ValueError(
                'EpiTransOp requires view=1 because the protocols do not define a temporal multi-view axis'
            )
        if points != self.config.patch_points:
            raise ValueError(
                f'EpiTransOp patch contract is {self.config.patch_points}, received {points}'
            )
        signal = eeg[:, 0]
        if not torch.isfinite(signal).all():
            signal = torch.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        return signal

    @staticmethod
    def smooth_mass_confidence(
        source_mass: torch.Tensor,
        reference_mass: float,
        temperature: float,
    ) -> torch.Tensor:
        log_mass_ratio = (
            source_mass.float().clamp_min(1e-8).log()
            - math.log(float(reference_mass))
        )
        return torch.sigmoid(log_mass_ratio / float(temperature))

    def forward_features(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        return_transport: bool = False,
        return_details: bool = False,
    ):
        signal = self._merge_views(eeg)
        if channel_mask.shape != signal.shape[:2]:
            raise ValueError('EpiTransOp channel mask does not match the input channel axis')
        mask = channel_mask.to(device=signal.device, dtype=torch.bool)
        channel_tokens = self._finite_tensor(self.patch_encoder(signal))
        descriptors = self._finite_tensor(
            self.electrode_transport.signal_descriptors(signal, mask)
        )
        tokens, details = self.electrode_transport(channel_tokens, descriptors, mask)
        tokens = self._finite_tensor(tokens)
        details = self._finite_details(details)
        details['transport_slot_features'] = tokens.mean(dim=2)
        time_encoding = sinusoidal_time_encoding(
            tokens.shape[2], tokens.shape[3], tokens.device, tokens.dtype
        )
        if self.config.mass_aware_transport:
            reference_mass = 1.0 / float(self.config.latent_electrodes)
            source_mass = details['temporal_source_mass'].float()
            confidence = self.smooth_mass_confidence(
                source_mass,
                reference_mass,
                self.config.electrode_mass_temperature,
            ).to(dtype=tokens.dtype)
            confidence = confidence.permute(0, 2, 1).unsqueeze(-1)
            prior = self.latent_embedding.to(tokens.dtype)
            tokens = confidence * tokens + (1.0 - confidence) * prior
            tokens = self._finite_tensor(tokens)
            details['mass_confidence'] = confidence.squeeze(-1)
        else:
            details['mass_confidence'] = torch.ones(
                tokens.shape[0], tokens.shape[1], tokens.shape[2],
                device=tokens.device, dtype=tokens.dtype,
            )
        tokens = tokens + time_encoding[None, None]
        scale_gates = []
        for block in self.blocks:
            tokens, scale_gate = block(tokens)
            tokens = self._finite_tensor(tokens)
            scale_gate = self._finite_tensor(scale_gate, limit=1.0)
            scale_gates.append(scale_gate.mean(dim=-1))
        tokens = self._finite_tensor(self.final_norm(tokens))
        details['slot_features'] = tokens.mean(dim=2)
        details['temporal_scale_gate'] = torch.stack(scale_gates).mean(dim=0)
        if return_details:
            return tokens, details
        if return_transport:
            return tokens, details['transport']
        return tokens

    def forward(
        self,
        eeg: torch.Tensor,
        channel_mask: torch.Tensor,
        return_transport: bool = False,
        return_details: bool = False,
    ):
        tokens, details = self.forward_features(eeg, channel_mask, return_details=True)
        logits, head_details = self.head(tokens)
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=20.0,
            neginf=-20.0,
        ).clamp(min=-20.0, max=20.0)
        head_details = self._finite_details(head_details)
        details.update(head_details)
        details['slot_features'] = details['shared_field'].mean(dim=2)
        details['shared_slot_features'] = details['slot_features']
        details['residual_slot_features'] = details['residual_field'].mean(dim=2)
        if return_details:
            return logits, details
        if return_transport:
            return logits, details['transport']
        return logits

    @torch.no_grad()
    def update_source_slot_prototypes(
        self,
        slot_features: torch.Tensor,
        labels: torch.Tensor,
        source_mask: torch.Tensor,
        momentum: float,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError('Prototype momentum must be in [0, 1)')
        for class_index in range(2):
            selected = source_mask & (labels.long() == class_index)
            if not torch.any(selected):
                continue
            prototype = F.normalize(
                slot_features[selected].detach().mean(dim=0), dim=-1
            )
            if self.source_slot_initialized[class_index]:
                prototype = F.normalize(
                    momentum * self.source_slot_prototypes[class_index]
                    + (1.0 - momentum) * prototype,
                    dim=-1,
                )
            self.source_slot_prototypes[class_index].copy_(prototype)
            self.source_slot_initialized[class_index] = True

    def class_conditional_slot_alignment_loss(
        self,
        slot_features: torch.Tensor,
        labels: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        normalized = F.normalize(slot_features, dim=-1)
        for class_index in range(2):
            selected = target_mask & (labels.long() == class_index)
            if not torch.any(selected) or not self.source_slot_initialized[class_index]:
                continue
            target_prototype = F.normalize(normalized[selected].mean(dim=0), dim=-1)
            source_prototype = self.source_slot_prototypes[class_index].detach()
            losses.append(
                1.0 - (target_prototype * source_prototype).sum(dim=-1).mean()
            )
        if not losses:
            return slot_features.sum() * 0.0
        return torch.stack(losses).mean()

    def configuration(self) -> dict[str, Any]:
        return asdict(self.config)

    def model_information(self) -> dict[str, Any]:
        return {
            'Architecture': 'RIVER',
            'Task': self.config.task,
            'Sampling frequency': self.config.sampling_frequency,
            'Patch points': self.config.patch_points,
            'Descriptor scales seconds': (
                f'1,{self.config.descriptor_window_seconds:g},'
                f'{self.config.descriptor_slow_seconds:g}'
            ),
            'Descriptor channel chunk size': self.config.descriptor_channel_chunk_size,
            'Latent neural sources': self.config.latent_electrodes,
            'Signal descriptor dimensions per scale': self.config.descriptor_dim,
            'Embedding dimensions': self.config.embed_dim,
            'Operator depth': self.config.depth,
            'Attention heads': self.config.heads,
            'Feedforward dimensions': self.config.ff_dim,
            'Fourier modes': self.config.fourier_modes,
            'Fourier rank': self.config.fourier_rank,
            'Local kernel size': self.config.local_kernel_size,
            'Transport mode': self.config.adapter_mode,
            'Sinkhorn iterations': self.config.sinkhorn_iterations,
            'Transport epsilon': self.config.transport_epsilon,
            'Mass relaxation': self.config.electrode_mass_relaxation,
            'Mass aware transport': self.config.mass_aware_transport,
            'Mass confidence': 'smooth_log_mass_ratio_sigmoid',
            'Residual bottleneck dimensions': self.config.residual_bottleneck_dim,
            'Residual initialization': 'zero_output_projection',
            'Temporal mixer': self.config.temporal_mixer,
            'Evidence pooling': 'duration_aware_peak_persistent',
            'Source objective': 'mean_task_bce_original_channel_drop',
            'Target objective': 'task_bce_optional_class_alignment',
            'Dropout': self.config.dropout,
            'Training EMA decay': float(getattr(self, 'training_ema_decay', 0.0)),
            'Training precision': str(getattr(self, 'training_precision', 'fp32')),
            'Pretraining': 'disabled until a pretraining dataset is approved',
        }

    def cross_modal_adapter_contract(self) -> dict[str, Any]:
        return self.electrode_transport.contract()
