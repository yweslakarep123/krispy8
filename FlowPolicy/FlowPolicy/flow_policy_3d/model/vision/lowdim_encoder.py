"""Simple MLP encoder for low-dim state-based environments (e.g. Franka Kitchen).

Replaces the point-cloud FlowPolicyEncoder; only consumes the `agent_pos`
observation key (robot state optionally concatenated with a flattened goal).
"""

from typing import Dict, Sequence, Type
import torch
import torch.nn as nn
from termcolor import cprint


def _create_mlp(input_dim: int,
                output_dim: int,
                hidden_dims: Sequence[int],
                activation_fn: Type[nn.Module] = nn.ReLU,
                use_layernorm: bool = True) -> nn.Sequential:
    layers = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        if use_layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation_fn())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class LowdimEncoder(nn.Module):
    """MLP encoder that maps `agent_pos` -> feature vector.

    The output shape mirrors FlowPolicyEncoder (single vector of size
    out_channel) so the rest of FlowPolicy can treat it identically.
    """

    state_key = 'agent_pos'

    def __init__(self,
                 observation_space: Dict,
                 out_channel: int = 256,
                 hidden_dims: Sequence[int] = (256, 256),
                 use_layernorm: bool = True,
                 ):
        super().__init__()
        assert self.state_key in observation_space, \
            f"LowdimEncoder requires '{self.state_key}' in observation_space, " \
            f"got keys={list(observation_space.keys())}"
        state_shape = observation_space[self.state_key]
        assert len(state_shape) == 1, \
            f"agent_pos must be 1D, got shape {state_shape}"

        self.state_dim = int(state_shape[0])
        self.n_output_channels = int(out_channel)

        self.mlp = _create_mlp(
            input_dim=self.state_dim,
            output_dim=self.n_output_channels,
            hidden_dims=tuple(hidden_dims),
            use_layernorm=use_layernorm,
        )

        cprint(f"[LowdimEncoder] state_dim={self.state_dim} "
               f"out_channel={self.n_output_channels} "
               f"hidden_dims={tuple(hidden_dims)}", 'yellow')

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = observations[self.state_key]
        assert x.ndim == 2, \
            f"agent_pos expected shape (B, D), got {tuple(x.shape)}"
        return self.mlp(x)

    def output_shape(self) -> int:
        return self.n_output_channels
