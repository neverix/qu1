import jax
import jax.numpy as jnp
import equinox as eqx
from .common import VLinear, Lora, GroupNorm, weighter_init, time_shift
from .config import RWKVConfig


class ChannelMix(eqx.Module):
    d_model: int
    d_ff: int
    
    x_k: jax.Array
    
    def __init__(self, *, config: RWKVConfig, layer_idx: int, key: jax.random.PRNGKey):
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        
        self.x_k = weighter_init(config, layer_idx, 4.0)
        key1, key2 = jax.random.split(key, 2)
        self.key = VLinear(d_model, d_ff, key=key1, initialization="uniform")
        self.value = VLinear(d_ff, d_model, key=key2, initialization="zeros")
    
    def __call__(self, x):
        k = x + self.x_k * (time_shift(x) - x)
        k = jax.nn.relu(self.key(k)) ** 2
        return self.value(k)
