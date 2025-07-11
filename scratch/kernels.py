#%%
%load_ext autoreload
%autoreload 2
%cd ~/qu1
#%%
import jax
import jax.numpy as jnp
from functools import partial
from qu1 import rwkv_kernels
batch_size = 2
seq_len = 8192
n_heads = 8
d_head = 128

key = jax.random.key(0)
ks = jax.random.split(key, 6)
r = jax.random.normal(ks[0], (batch_size, seq_len, n_heads, d_head))
w = jax.random.normal(ks[1], (batch_size, seq_len, n_heads, d_head))
w = -jax.nn.softplus(w) - 0.5
k = jax.random.normal(ks[2], (batch_size, seq_len, n_heads, d_head))
v = jax.random.normal(ks[3], (batch_size, seq_len, n_heads, d_head))
a = jax.random.normal(ks[4], (batch_size, seq_len, n_heads, d_head))
b = jax.random.normal(ks[5], (batch_size, seq_len, n_heads, d_head))

a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
b = a * jax.nn.sigmoid(b)

gts, gto = rwkv_kernels.rwkv_update(r, w, k, v, a, b)

y = rwkv_kernels.rwkv_update(r, w, k, v, a, b, fn=rwkv_kernels.serial_rwkv)[1]
x = gto
print(jnp.corrcoef(x.flatten(), y.flatten()))

print(rwkv_kernels.benchmark(rwkv_kernels.rwkv_update)(r, w, k, v, a, b), rwkv_kernels.benchmark(partial(rwkv_kernels.rwkv_update, fn=rwkv_kernels.serial_rwkv))(r, w, k, v, a, b))
#%%
back_1 = rwkv_kernels.rwkv_backward(r, w, k, v, a, b)
back_2 = rwkv_kernels.rwkv_backward(r, w, k, v, a, b, fn=rwkv_kernels.serial_rwkv)
for x, y in zip(back_1, back_2):
    s = 20
    x, y = x[:, s:], y[:, s:]
    print(jnp.corrcoef(x.flatten(), y.flatten())[0, 1])
#%%
print(rwkv_kernels.benchmark(rwkv_kernels.rwkv_backward)(r, w, k, v, a, b))
print(rwkv_kernels.benchmark(partial(rwkv_kernels.rwkv_backward, fn=rwkv_kernels.serial_rwkv))(r, w, k, v, a, b))
#%%
from matplotlib import pyplot as plt
ww = w[:, :, 0, :]
plt.plot(ww.min(axis=-1).min(axis=0))
plt.plot(ww.max(axis=-1).max(axis=0))
plt.show()
# %%
