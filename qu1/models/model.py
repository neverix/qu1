from .config import RWKVConfig
import jax
import jax.numpy as jnp
import equinox as eqx
from .time_mix import TimeMix
from .channel_mix import ChannelMix
from .common import VLayerNorm, VLinear, vmap_any


class RWKVLayer(eqx.Module):
    layer_idx: int
    time_mix: TimeMix
    channel_mix: ChannelMix
    ln0: VLayerNorm
    ln1: VLayerNorm
    ln2: VLayerNorm

    def __init__(self, *, config: RWKVConfig, layer_idx: int, key: jax.random.PRNGKey):
        self.layer_idx = layer_idx
        self.time_mix = TimeMix(config=config, layer_idx=layer_idx, key=key)
        self.channel_mix = ChannelMix(config=config, layer_idx=layer_idx, key=key)
        key1, key2, key3 = jax.random.split(key, 3)
        self.ln0 = VLayerNorm(config.d_model, key=key1)
        self.ln1 = VLayerNorm(config.d_model, key=key2)
        self.ln2 = VLayerNorm(config.d_model, key=key3)
    
    def __call__(self, x, v_first, state_update):
        x = jnp.where(self.layer_idx == 0, self.ln0(x), x)
        x = self.ln1(x)
        x_attn, _state, v_first = self.time_mix(x, time_shift(x), v_first, first_layer=self.layer_idx == 0, state_update=state_update)
        x = x + x_attn
        x = x + self.channel_mix(self.ln2(x))
        return x, v_first


class RWKV(eqx.Module):
    config: RWKVConfig
    embedding: eqx.nn.Embedding
    layers: RWKVLayer
    ln_out: VLayerNorm
    lm_head: VLinear
    
    def __init__(self, *, config: RWKVConfig, key: jax.random.PRNGKey):
        def initialize_layer(key, layer_idx):
            key1, key2 = jax.random.split(key, 2)
            layer = RWKVLayer(config=config, layer_idx=layer_idx, key=key1)
            return key2, layer
        layer_indices = jnp.arange(config.n_layers)
        self.layers, key = jax.lax.scan(initialize_layer, key, layer_indices)
        self.embedding = eqx.nn.Embedding(config.vocab_size, config.d_model, key=key)
        # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L405
        self.embedding = jax.random.uniform(key, (config.vocab_size, config.d_model), minval=-1e-4, maxval=1e-4)
        self.ln_out = VLayerNorm(config.d_model, key=key)
        self.lm_head = VLinear(config.d_model, config.vocab_size, key=key, initialization="ratio_orthogonal")

    def __call__(self, x):
        x = vmap_any(self.embedding)(x)
        v_first = jnp.zeros_like(x)
        def forward(state, layer_idx):
            x, v_first = state
            x, v_first = self.layers[layer_idx](x, v_first, state_update=state_update)
            return x, v_first
        x, v_first = jax.lax.fori_loop(0, config.n_layers, forward, (x, v_first))
        y = self.ln_out(x)
        y = self.lm_head(y)
        return y
