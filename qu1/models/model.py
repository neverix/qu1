from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx

from .config import RWKVConfig
from .time_mix import TimeMix
from .channel_mix import ChannelMix
from .common import VLinear, time_shift, vmap_any


class RWKVLayer(eqx.Module):
    layer_idx: int
    time_mix: TimeMix
    channel_mix: ChannelMix
    ln0: eqx.nn.LayerNorm
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(self, *, config: RWKVConfig, layer_idx: int, key: jax.random.PRNGKey):
        self.layer_idx = layer_idx
        self.time_mix = TimeMix(config=config, layer_idx=layer_idx, key=key)
        self.channel_mix = ChannelMix(config=config, layer_idx=layer_idx, key=key)
        self.ln0 = eqx.nn.LayerNorm(config.d_model)
        self.ln1 = eqx.nn.LayerNorm(config.d_model)
        self.ln2 = eqx.nn.LayerNorm(config.d_model)
    
    def __call__(self, x, v_first, state_update, state=None, new_mask=None):
        x = jnp.where(self.layer_idx == 0, vmap_any(self.ln0)(x), x)
        x = vmap_any(self.ln1)(x)
        with jax.named_scope("Time Mix"):
            x_attn, state, v_first = self.time_mix(x, time_shift(x), v_first,
                                                    first_layer=self.layer_idx == 0,
                                                    state_update=state_update,
                                                    state=state,
                                                    new_mask=new_mask)
        x = x + x_attn
        with jax.named_scope("Channel Mix"):
            x = x + self.channel_mix(vmap_any(self.ln2)(x))
        return x, state, v_first


class RWKV(eqx.Module):
    config: RWKVConfig = eqx.field(static=True)
    state_update: Callable = eqx.field(static=True)
    embedding: eqx.nn.Embedding
    layers: RWKVLayer | list[RWKVLayer]
    ln_out: eqx.nn.LayerNorm
    lm_head: VLinear
    
    def __init__(self, *, config: RWKVConfig, key: jax.random.PRNGKey, state_update: Callable):
        self.config = config
        self.state_update = state_update
        
        if config.layer_scan:
            def initialize_layer(key, layer_idx):
                key1, key2 = jax.random.split(key, 2)
                layer = RWKVLayer(config=config, layer_idx=layer_idx, key=key1)
                return key2, layer
            layer_indices = jnp.arange(config.n_layers)
            key, self.layers = jax.lax.scan(initialize_layer, key, layer_indices)
        else:
            self.layers = []
            for layer_idx in range(config.n_layers):
                key, subkey = jax.random.split(key)
                layer = RWKVLayer(config=config, layer_idx=layer_idx, key=subkey)
                self.layers.append(layer)

        self.embedding = eqx.nn.Embedding(config.vocab_size, config.d_model, key=key)
        # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L405
        self.embedding = eqx.tree_at(lambda m: m.weight, self.embedding, jax.random.uniform(key, (config.vocab_size, config.d_model), minval=-1e-4, maxval=1e-4))
        self.ln_out = eqx.nn.LayerNorm(config.d_model)
        self.lm_head = VLinear(config.d_model, config.vocab_size, key=key, initialization="ratio_orthogonal")

    def __call__(self, x, states: dict[int, jnp.ndarray] | jnp.ndarray | None = None, return_states: bool = False, new_mask=None):
        with jax.named_scope("Embedding"):
            x = self.embedding.weight[x]
        v_first = jnp.empty_like(x)
        
        if self.config.layer_scan:
            def forward(state, scanned):
                x, v_first = state
                layer, state = scanned
                with jax.named_scope("Layer forward"):
                    x, new_state, v_first = layer(x, v_first, state_update=self.state_update, state=state, new_mask=new_mask)
                return (x, v_first), (new_state if return_states else None)
            (x, v_first), new_states = jax.lax.scan(forward, (x, v_first), (self.layers, None if states is None else states), unroll=True)
        else:
            assert isinstance(self.layers, list)
            with jax.named_scope("Layer forward"):
                new_states = {} if states is None else {}
                for i, layer in enumerate(self.layers):
                    x, new_state, v_first = layer(x, v_first, state_update=self.state_update, state=states[i] if states is not None else None, new_mask=new_mask)
                    if return_states:
                        new_states[i] = new_state
        
        with jax.named_scope("Output"):
            y = vmap_any(self.ln_out)(x)
            y = self.lm_head(y)
        if return_states:
            return y, new_states
        else:
            return y
