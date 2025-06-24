from functools import wraps
from typing import Literal

import jax
import jax.numpy as jnp
import equinox as eqx


class VLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None = None

    def __init__(self, d_in: int, d_out: int, *, key: jax.random.PRNGKey, use_bias: bool = True,
                 initialization: Literal["xavier", "orthogonal", "zeros", "uniform", "ratio_orthogonal"] = "xavier"):
        if initialization == "xavier":
            self.weight = jax.nn.initializers.xavier_normal()(key, (d_out, d_in))
        elif initialization == "orthogonal":
            self.weight = jax.nn.initializers.orthogonal(scale=0.1)(key, (d_out, d_in))
        elif initialization == "zeros":
            self.weight = jnp.zeros((d_out, d_in))
        elif initialization == "uniform":
            self.weight = jax.nn.initializers.uniform(scale=0.5/(d_in ** 0.5))(key, (d_out, d_in))
        elif initialization == "ratio_orthogonal":
            self.weight = jax.nn.initializers.orthogonal(scale=0.5 * (d_out / d_in) ** 0.5)(key, (d_out, d_in))
        else:
            raise ValueError(f"Invalid initialization: {initialization}")
        
        if use_bias:
            self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias
        return y


class VLayerNorm(eqx.nn.LayerNorm):
    def __call__(self, x):
        return super().__call__(x.reshape(-1, x.shape[-1])).reshape(x.shape)


def vmap_any(f, n_dims: int = 1):
    @wraps(f)
    def fn_vmap(*args):
        fn = f
        for _ in range(args[0].ndim - n_dims):
            fn = eqx.filter_vmap(fn)
        return fn(*args)
    return fn_vmap


class Lora(eqx.Module):
    w1: VLinear
    w2: VLinear
    activation: Literal["none", "sigmoid", "tanh"] = eqx.field(static=True, default="none")

    def __init__(self, d_in: int, d_mid: int, bias_pre: bool = False, bias_post: bool = False, activation: Literal["none", "sigmoid", "tanh"] = "none", *, key: jax.random.PRNGKey):
        key1, key2 = jax.random.split(key, 2)
        self.w1 = VLinear(d_in, d_mid, key=key1, use_bias=bias_pre, initialization="zeros")
        self.w2 = VLinear(d_mid, d_in, key=key2, use_bias=bias_post, initialization="orthogonal")
        self.activation = activation

    def __call__(self, x):
        x = self.w1(x)
        match self.activation:
            case "sigmoid":
                x = jax.nn.sigmoid(x)
            case "tanh":
                x = jax.nn.tanh(x)
        x = self.w2(x)
        return x


class GroupNorm(eqx.Module):
    num_groups: int = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)

    def __init__(self, num_groups: int, num_channels: int, *, eps: float = 1e-5):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.weight = jnp.ones((num_groups, num_channels,))
        self.bias = jnp.zeros((num_groups, num_channels,))
        self.eps = eps

    def __call__(self, x):
        x = x.reshape(x.shape[:-1] + (self.num_groups, self.num_channels))
        x = x - x.mean(axis=-1, keepdims=True)
        x = x / jnp.sqrt(x.var(axis=-1, keepdims=True) + self.eps)
        x = x * self.weight + self.bias
        return x.reshape(x.shape[:-2] + (self.num_groups * self.num_channels,))


def weighter_init(config, layer_idx, power: float = 1.0):
    # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L96
    ratio_1_to_almost0 = 1.0 - (layer_idx / config.n_layers)  # 1 to ~0
    ddd = jnp.zeros(config.d_model) + jnp.arange(config.d_model) / config.d_model
    return 1.0 - jnp.power(ddd, power * ratio_1_to_almost0)

def patterned_bias(config, layer_idx, usage: Literal["w", "a", "v"]):
    # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L117
    c = config.d_model
    match usage:
        case "w":
            linear_weight, zigzag_weight, www_weight, bias = 0.5, 2.5, 0.0, 0.0
        case "a":
            linear_weight, zigzag_weight, www_weight, bias = 0.4, 0.3, 0.0, -0.19
        case "v":
            linear_weight, zigzag_weight, www_weight, bias = -0.4, 0.0, 0.73, 0.0
        case _:
            raise ValueError(f"Invalid usage: {usage}")
    n = jnp.arange(c)
    linear = n / (c - 1) - 0.5
    # x^2 * sign(x) for each of the heads
    zigzag = ((n % config.n_head) - ((config.n_head - 1) / 2)) / ((config.n_head - 1) / 2)
    zigzag = zigzag * jnp.abs(zigzag)
    ratio_0_to_1 = layer_idx / (config.n_layers - 1)
    www = -6 + 6 * (n / (c - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
    return linear * linear_weight + zigzag * zigzag_weight + www * www_weight + bias


def calculate_lora_dimensions(d_model):
    # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L126
    d_w = max(32, int(round(  (1.8*(d_model**0.5))  /32)*32))
    d_a = max(32, int(round(  (1.8*(d_model**0.5))  /32)*32))
    d_g = max(32, int(round(  (0.6*(d_model**0.8))  /32)*32))
    d_v = max(32, int(round(  (1.3*(d_model**0.5))  /32)*32))
    return d_w, d_a, d_g, d_v

def time_shift(x):
    return jnp.roll(x, shift=1, axis=-2).at[..., :1, :].set(0)
