import equinox as eqx


class RWKVConfig(eqx.Module):
    n_layers: int
    d_model: int
    d_ff: int
    n_head: int
    head_size: int
    d_w: int
    d_a: int
    d_g: int
    d_v: int
    eps: float = eqx.static_field()
