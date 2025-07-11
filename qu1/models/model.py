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
    
    def __call__(self, x, v_first, state_update, new_mask=None):
        x = jnp.where(self.layer_idx == 0, vmap_any(self.ln0)(x), x)
        x = vmap_any(self.ln1)(x)
        v_first = v_first * 0
        x_attn, _state, v_first = self.time_mix(x, time_shift(x), v_first, first_layer=self.layer_idx == 0, state_update=state_update, new_mask=new_mask)
        x = x + x_attn
        x = x + self.channel_mix(vmap_any(self.ln2)(x))
        return x, v_first


class RWKV(eqx.Module):
    config: RWKVConfig = eqx.field(static=True)
    state_update: Callable = eqx.field(static=True)
    embedding: eqx.nn.Embedding
    layers: RWKVLayer
    ln_out: eqx.nn.LayerNorm
    lm_head: VLinear
    
    def __init__(self, *, config: RWKVConfig, key: jax.random.PRNGKey, state_update: Callable):
        self.config = config
        self.state_update = state_update
        
        def initialize_layer(key, layer_idx):
            key1, key2 = jax.random.split(key, 2)
            layer = RWKVLayer(config=config, layer_idx=layer_idx, key=key1)
            return key2, layer
        layer_indices = jnp.arange(config.n_layers)
        key, self.layers = jax.lax.scan(initialize_layer, key, layer_indices)

        self.embedding = eqx.nn.Embedding(config.vocab_size, config.d_model, key=key)
        # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L405
        self.embedding = eqx.tree_at(lambda m: m.weight, self.embedding, jax.random.uniform(key, (config.vocab_size, config.d_model), minval=-1e-4, maxval=1e-4))
        self.ln_out = eqx.nn.LayerNorm(config.d_model)
        self.lm_head = VLinear(config.d_model, config.vocab_size, key=key, initialization="ratio_orthogonal")

    def __call__(self, x, new_mask=None):
        x = vmap_any(self.embedding, n_dims=0)(x)
        v_first = jnp.zeros_like(x)
        def forward(layer_idx, state):
            x, v_first = state
            layer = jax.tree.map(lambda m: m[layer_idx], self.layers)
            x, v_first = layer(x, v_first, state_update=self.state_update, new_mask=new_mask)
            return x, v_first
        x, v_first = jax.lax.fori_loop(0, self.config.n_layers, forward, (x, v_first))
        y = vmap_any(self.ln_out)(x)
        y = self.lm_head(y)
        return y
