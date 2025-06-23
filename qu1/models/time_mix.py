import jax
import jax.numpy as jnp
import equinox as eqx
from .common import VLinear, Lora, GroupNorm, weighter_init, patterned_bias
from .config import RWKVConfig


class TimeMix(eqx.Module):
    n_head: int
    head_size: int

    rwkvag: jax.Array

    rw: VLinear
    w: Lora
    kw: VLinear
    vw: VLinear
    a: Lora
    g: Lora
    k_k: jax.Array
    k_a: jax.Array
    v: Lora
    r_k: jax.Array
    out_gn: GroupNorm
    out: VLinear

    def __init__(self, *, config: RWKVConfig, layer_idx: int, key: jax.random.PRNGKey):
        d_model, n_head, head_size, d_w, d_a, d_g, d_v = config.d_model, \
            config.n_head, config.head_size, config.d_w, config.d_a, config.d_g, config.d_v

        self.n_head, self.head_size = n_head, head_size

        self.rwkvag = jnp.stack([
            weighter_init(config, layer_idx, p)
            for p in (0.2, 0.9, 0.7, 0.7, 0.9, 0.2)
        ], axis=0)

        r_key, w_key, k_key, v_key, a_key, g_key = jax.random.split(key, 6)

        # eponymous rwkv
        # low-rank or dense projections of the input
        d_mid = n_head * head_size
        self.rw = VLinear(d_model, d_model, key=r_key, use_bias=False)
        self.w = Lora(d_model, d_w, key=w_key, activation="tanh", bias_post=True)

        self.kw = VLinear(d_model, d_mid, key=k_key, use_bias=False)
        self.vw = VLinear(d_model, d_mid, key=v_key, use_bias=False)
        # acceptance weight
        # used to create a probability that controls how much of kk is a bias
        # and used to create a right eigenvector for the update matrix (gating kk)
        self.a = Lora(d_model, d_a, key=a_key, bias_post=True)
        # gate weight
        self.g = Lora(d_model, d_g, key=g_key, activation="sigmoid")

        # basis for update matrix
        # will be used to create a unit vector from k
        self.k_k = jnp.ones((d_mid,))
        # an optional modulation factor
        # modulated itself by acceptance
        self.k_a = jnp.zeros((d_mid,))

        # gate for controlling how much of the first layer's v is added to this one
        self.v = Lora(d_model, d_v, key=v_key, bias_post=True)
        
        
        replace_bias = lambda m, *args, **kwargs: eqx.tree_at(m, lambda m: m.w2.bias, patterned_bias(config, layer_idx, *args, **kwargs))
        self.w = replace_bias(self.w, "w")
        self.a = replace_bias(self.a, "a")
        self.v = replace_bias(self.v, "v")
        self.g = replace_bias(self.g, "g")

        # which components of r+k used to control what heads write a plain copy of v to the output?
        self.r_k = jnp.ones((d_mid,))

        self.out_gn = GroupNorm(n_head, head_size, eps=config.eps)
        self.out = VLinear(d_mid, d_model, key=v_key, use_bias=False, initialization="zeros")

    def __call__(self, current, prev, v_first = None, first_layer: bool = False, *, state_update):
        diff = prev - current
        xr, xw, xk, xv, xa, xg = (a[..., 0, :] for a in jnp.split(current + self.rwkvag * diff[..., None, :], 6, axis=-2))

        bd = current.shape[:-1]
        add_heads = lambda x: x.reshape(bd + (self.n_head, self.head_size))
        rm_heads = lambda x: x.reshape(bd + (self.n_head * self.head_size,))

        # lora projections
        r = self.rw(xr)
        w = self.w(xw)
        k = self.kw(xk)
        v = self.vw(xv)
        a = jax.nn.sigmoid(self.a(xa))
        g = self.g(xg)

        # create "bone" for our update matrix, the left eigenvector
        kk = add_heads(k * self.k_k)
        kk = kk / jnp.linalg.norm(kk, axis=-1, keepdims=True)
        # optionally upweight some components of k
        k = k * (1 + (a-1) * self.k_a)

        # :/
        if v_first is None:
            v_first = v
        v_first = jnp.where(first_layer, v, v_first)
        v = v + (v_first - v) * jax.nn.sigmoid(self.v(xv))

        # exp(-exp(-softplus(w)))
        # w is used as the key to let the past state persist. i don't know why we need two exponentials
        w = jnp.exp(-0.606531 * jax.nn.sigmoid(w)) # 0.606531 = exp(-0.5)

        r_k = add_heads(r * k * self.r_k).sum(axis=-1, keepdims=True)

        r, w, k, v = map(add_heads, (r, w, k, v))
        a, b = -kk, kk * add_heads(a)

        state, out = state_update(state, r, w, k, v, a, b)

        out = rm_heads(out)
        out = self.out_gn(out)
        out = out + rm_heads(r_k * v)

        out = self.out(out * g)
        return out, state, v_first

