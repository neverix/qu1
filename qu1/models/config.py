import equinox as eqx
from dataclasses import replace
from .common import calculate_lora_dimensions


class RWKVConfig(eqx.Module):
    n_layers: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)
    vocab_size: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    d_ff: int = None
    head_size: int = None
    d_w: int = None
    d_a: int = None
    d_g: int = None
    d_v: int = None
    eps: float = eqx.static_field(default=64e-5)
    
    def __post_init__(self):
        # https://github.com/BlinkDL/RWKV-LM/blob/49cfc1e5ddf348e7d07c08ec2ca527e32e4bcf9a/RWKV-v7/train_temp/src/model.py#L126
        d_w, d_a, d_g, d_v = calculate_lora_dimensions(self.d_model)
        if self.d_w is None:
            self.d_w = d_w
        if self.d_a is None:
            self.d_a = d_a
        if self.d_g is None:
            self.d_g = d_g
        if self.d_v is None:
            self.d_v = d_v
        if self.d_ff is None:
            self.d_ff = self.d_model * 4
        if self.head_size is None:
            assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
            self.head_size = self.d_model // self.n_head
        elif self.n_head is None:
            assert self.d_model % self.head_size == 0, "d_model must be divisible by head_size"
            self.n_head = self.d_model // self.head_size
        else:
            assert self.n_head * self.head_size == self.d_model, "n_head * head_size must be equal to d_model"
