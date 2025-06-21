from .config import RWKVConfig
import jax
import jax.numpy as jnp
import equinox as eqx


class RWKV(eqx.Module):
    config: RWKVConfig
    layer: eqx.Module
    
    def __init__(self, *, config: RWKVConfig, key: jax.random.PRNGKey):
        pass
