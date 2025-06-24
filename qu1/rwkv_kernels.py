from functools import partial

import timeit
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def w_transform(w):
    return jnp.exp(-jnp.exp(w))

def w_transform_inv(w):
    return jnp.exp(jnp.exp(w))

def w_transform_backward(w, w_exp, dw_exp):
    return w_exp * (-jnp.exp(w)) * dw_exp

def state_update(state, rwkvab):
    r, w, k, v, a, b = rwkvab
    w = w_transform(w)
    state = jnp.einsum("...xy, ...y -> ...xy", state, w) \
        + jnp.einsum("...xy, ...y, ...z->...xz", state, a, b) \
        + jnp.einsum("...x, ...y -> ...xy", v, k)
    out = jnp.einsum("...xy, ...y -> ...x", state, r)
    return state, out

def scanner(state, rwkvab):
    return jax.lax.scan(state_update, state, rwkvab)


def rwkv_update(r, w, k, v, a, b, state=None, *, fn=jax.vmap(scanner)):
    batch_size, seq_len, n_heads, d_head = r.shape
    reorder = lambda x: x.transpose((0, 2, 1, 3)).reshape((batch_size * n_heads, seq_len, d_head))
    r, w, k, v, a, b = map(reorder, (r, w, k, v, a, b))
    if state is None:
        state = jnp.zeros((batch_size * n_heads, d_head, d_head))
    state, out = fn(state, (r, w, k, v, a, b))
    state = state.reshape(batch_size, n_heads, d_head, d_head)
    out = out.reshape(batch_size, n_heads, seq_len, d_head).transpose(0, 2, 1, 3)
    return state, out



def serial_kernel(r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, state_ref, y_ref, ab_ref, out_state_ref, state_acc_ref, *, seq_len, save_ab=False):
    @pl.when(pl.program_id(1) == 0)
    def _():
        state_acc_ref[...] = state_ref[...]

    state = state_acc_ref[...]
    a = a_ref[...][0]

    ab = jnp.repeat(a, state.shape[1], axis=0).reshape(state.shape) * state
    ab = ab.reshape(-1, ab.shape[-1])
    averager = jnp.ones((ab.shape[-1], ab.shape[-1])) / ab.shape[-1]
    ab = ab @ averager
    ab = ab.reshape(*state.shape[:2], ab.shape[-1])

    if save_ab:
        ab_sum = ab.reshape(ab.shape[0] * ab.shape[1], -1).T[0].reshape(ab.shape[0], ab.shape[1])
        ab_ref[...] = ab_sum[None, :, :]

    new_state_1 = state * w_transform(w_ref[...][0][:, None, :])
    new_state_2 = ab * b_ref[...][0][:, None, :]
    new_state_3 = v_ref[...][0][:, :, None] * k_ref[...][0][:, None, :]

    new_state = new_state_1 + new_state_2 + new_state_3
    r = r_ref[...][0]

    new_state_for_out = new_state.reshape(-1, new_state.shape[2]).T
    r_for_out = jnp.repeat(r.T, new_state.shape[1], axis=1)
    out = (new_state_for_out * r_for_out).sum(axis=0).reshape(*new_state.shape[:2])

    state_acc_ref[...] = new_state

    y_ref[...] = out[None, :, :]


    @pl.when(pl.program_id(1) == seq_len - 1)
    def _():
        out_state_ref[...] = state_acc_ref[...]

def serial_rwkv_kernel(r, w, k, v, a, b, state, save_ab=False):
    seq_len, batch_size, d_head = r.shape
    c_b = 32
    n_batches = batch_size // c_b

    y, ab, state =  pl.pallas_call(
        partial(serial_kernel, seq_len=seq_len, save_ab=save_ab),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, c_b, d_head), lambda i, j: (j, i, 0)),
            ] * 6 + [
                pl.BlockSpec((c_b, d_head, d_head), lambda i, j: (i, 0, 0))
            ],
            out_specs=[
                pl.BlockSpec((1, c_b, d_head), lambda i, j: (j, i, 0)),
                pl.BlockSpec((1, c_b, d_head), lambda i, j: (j, i, 0)),
                pl.BlockSpec((c_b, d_head, d_head), lambda i, j: (i, 0, 0))
            ],
            scratch_shapes=[pltpu.VMEM((c_b, d_head, d_head), jnp.float32)],
            grid=(n_batches, seq_len),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((seq_len, batch_size, d_head), jnp.float32),
            jax.ShapeDtypeStruct((seq_len, batch_size, d_head), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, d_head, d_head), jnp.float32),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=False
    )(r, w, k, v, a, b, state)
    return y, ab, state

@jax.custom_vjp
def serial_kernel_rwkv(r, w, k, v, a, b, state):
    y, ab, state = serial_rwkv_kernel(r, w, k, v, a, b, state)
    return y, state

def serial_kernel_rwkv_forward(r, w, k, v, a, b, state):
    y, ab, state_new = serial_rwkv_kernel(r, w, k, v, a, b, state, save_ab=True)
    return (y, state_new), (r, w, k, v, a, b, ab, state_new)


def serial_backward(r_ref, w_ref, k_ref, v_ref, a_ref, b_ref, ab_ref, dy_ref, state_next_ref, dstate_ref,
                    dstate_prev_ref, dr_ref, dw_ref, dk_ref, dv_ref, da_ref, db_ref,
                    state_acc_ref, dstate_acc_ref,
                    *, seq_len):
    @pl.when(pl.program_id(1) == 0)
    def _():
        state_acc_ref[...] = state_next_ref[...]
        dstate_acc_ref[...] = dstate_ref[...]

    state = state_acc_ref[...]
    dy = dy_ref[...][0]
    dstate = dstate_acc_ref[...]

    a = a_ref[...][0]
    ab = ab_ref[...][0]
    b = b_ref[...][0]
    ab_contribution = ab[:, :, None] * b[:, None, :]
    v = v_ref[...][0]
    k = k_ref[...][0]
    vk_contribution = v[:, :, None] * k[:, None, :]
    
    w = w_ref[...][0]
    w_exp = w_transform(w)
    w_exp_inv = w_transform_inv(w)
    prev_state_w = state - ab_contribution - vk_contribution
    prev_state = prev_state_w * w_exp_inv[:, None, :]
    
    # we add some gradient from dy
    r = r_ref[...][0]
    dstate = dstate + dy[:, :, None] * r[:, None, :]
    # dstate flows through ab and w
    dstate_prev = dstate * w_exp[:, None, :] + \
        (dstate * b[:, None, :]).sum(axis=-1, keepdims=True) * a[:, None, :]
    
    # 1) dw
    dw_exp = jnp.sum(prev_state * dstate, axis=1)
    dw = w_transform_backward(w, w_exp, dw_exp)
    dw_ref[...] = dw[None, :, :]
    
    # 2) dr
    dr = (state * dy[:, :, None]).sum(axis=1)
    dr_ref[...] = dr[None, :, :]
    
    # 3) db
    db = (dstate * ab[:, :, None]).sum(axis=1)
    db_ref[...] = db[None, :, :]
    
    # 4) da
    dab = (dstate * b[:, None, :]).sum(axis=-1, keepdims=True)
    da = (dab * prev_state).sum(axis=1)
    da_ref[...] = da[None, :, :]
    
    # 5) dk
    dk = (dstate * v[:, :, None]).sum(axis=1)
    dk_ref[...] = dk[None, :, :]
    
    # 6) dv
    dv = (
        jnp.repeat(k, dstate.shape[1], axis=0)
        * dstate.reshape(-1, dstate.shape[-1])
    ).T.sum(axis=0).reshape(*dstate.shape[:2])
    dv_ref[...] = dv[None, :, :]
    
    state_acc_ref[...] = prev_state
    dstate_acc_ref[...] = dstate_prev

    @pl.when(pl.program_id(1) == seq_len - 1)
    def _():
        dstate_prev_ref[...] = dstate_acc_ref[...]

def serial_kernel_rwkv_backward(res, gradients):
    dy, dstate = gradients
    r, w, k, v, a, b, ab, state = res

    seq_len, batch_size, d_head = r.shape
    c_b = 16
    n_batches = batch_size // c_b

    dstate_prev, dr, dw, dk, dv, da, db = pl.pallas_call(
        partial(serial_backward, seq_len=seq_len),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((1, c_b, d_head), lambda i, j: (seq_len - j - 1, i, 0)),
            ] * 8 + [
                pl.BlockSpec((c_b, d_head, d_head), lambda i, j: (i, 0, 0))
            ] * 2,
            out_specs=[
                pl.BlockSpec((c_b, d_head, d_head), lambda i, j: (i, 0, 0))
            ] + [
                pl.BlockSpec((1, c_b, d_head), lambda i, j: (seq_len - j - 1, i, 0)),
            ] * 6,
            scratch_shapes=[pltpu.VMEM((c_b, d_head, d_head), jnp.float32)] * 2,
            grid=(n_batches, seq_len),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((batch_size, d_head, d_head), jnp.float32),
        ) + (
            jax.ShapeDtypeStruct((seq_len, batch_size, d_head), jnp.float32),
        ) * 6,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary")
        ),
        interpret=False
    )(r, w, k, v, a, b, ab, dy, state, dstate)
    return (dr, dw, dk, dv, da, db, dstate_prev)

serial_kernel_rwkv.defvjp(serial_kernel_rwkv_forward, serial_kernel_rwkv_backward)

def serial_rwkv(state, rwkvab):
    r, w, k, v, a, b = (x.transpose(1, 0, 2) for x in rwkvab)
    y, state = serial_kernel_rwkv(r, w, k, v, a, b, state)
    y = y.transpose(1, 0, 2)
    return state, y


def benchmark(f, ntrials: int = 20):
    f = jax.jit(f)
    def run(*args, **kwargs):
        # Compile function first
        jax.block_until_ready(f(*args, **kwargs))
        # Time function
        result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                                number=ntrials)
        time = result / ntrials
        # print(f"Time: {time}")
        return time
    return run


def rwkv_backward(r, w, k, v, a, b, fn=jax.vmap(scanner)):
    key = jax.random.key(0)
    batch_size, seq_len, n_heads, d_head = r.shape
    dstate = jax.random.normal(key, (batch_size, n_heads, d_head, d_head))
    dy = jax.random.normal(key, (batch_size, seq_len, n_heads, d_head))
    def rwkv_update_gradsum(r, w, k, v, a, b):
        state, y = rwkv_update(r, w, k, v, a, b, fn=fn)
        return jnp.sum(y * dy) + jnp.sum(state * dstate)

    dr, dw, dk, dv, da, db = jax.grad(rwkv_update_gradsum, argnums=(0, 1, 2, 3, 4, 5))(r, w, k, v, a, b)
    return dr, dw, dk, dv, da, db

# notes on chunked kernel
# (https://github.com/johanwind/wind_rwkv/blob/main/wind_rwkv/rwkv7/triton_bighead.py)
# operates in chunks.

# incl_pref: cumprod of w per chunk
# non_incl_pref: incl_pref / w
# inv_incl_pref: 1/incl_pref (correction for w)

# wq: r * incl_pref
# wa: a * non_incl_pref
# kwi: k * inv_incl_pref
# bwi: b * inv_incl_pref

# ab/ak masked with lower tri without diagonal,
# qb/qk masked with lower tri
# ab: wa @ bwi.T = a * non_incl_pref @ (b * inv_incl_pref).T
# so, we find the dot product of a and b for all pairs of timesteps
# and we multiply this by the product of w'ssince (except the last one)
# we are tracing b developing through steps, and w only matters when we have
# more than one step separating the b and the a
# ak: wa @ kwi.T = a * non_incl_pref @ (k * inv_incl_pref).T
# same but for a/k
# qb: wq @ bwi.T = r * non_incl_pref @ (b * inv_incl_pref).T
# same, but for contribution to output
# include the diagonal because we can write to state and output at the same state
# qk: wq @ kwi.T = r * non_incl_pref @ (k * inv_incl_pref).T
# ab_inv: tri_minv(ab)

# wa_state: a * non_incl_pref @ state
# wq_state: r * non_incl_pref @ state
# find contribution of starting state to new a/r
# we get a matrix of type T x A_rows
# then, to get the ab product of the starting state to all steps we just
# wa_state[:, :, None] * b[:, None, :]
# but that's inefficient. we just want to compute the contribution to r

# ab_u: ak @ v + wa_state
# straightforward. we know how much k forwards to a for each pair of steps,
# columns correspond to source timesteps (k), so we get a matrix of type
# T x A_rows again
# u: ab_inv @ ab_u
# how much each b should be reading in from all the previous a's
# yy: qk @ v + qb @ u + wq_state
# final output!
# 1) forward v's (each of type A_rows). straightforward
# = (r * non_incl_pref @ (k * inv_incl_pref).T) * mask2 @ v
# we matmul r with each past k
# and for each r, keep only the k's from the past/present
# then we add v for each of those.
# ...but wouldn't those v's be transformed by the ab?
# glad you asked!
# we handle this in ab_u; check how much each v will contribute
# 2) forward contributions transformed by ab's
# qb @ u = (qb * mask2) @ ab_inv @ (ak @ v + wa_state)
# qb: we write a b on each step and r reads from all of those
# ak @ v + wa_state: intake corresponding to added kv and wa_state
# we get one on each step! now the question is how we add them together
# it turns out that each past addition is transformed in a predictable way
# ab_inv is it. why? it is equal to ab + ab^2 + ... + ab^n + I
# remember, ab is strictly lower tri, so we need to add I!
# why do we need to raise it to an infinite power? so we can consider all hops
# 3) forward initial state contribution to r

# now update state and we're done

# state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
# fw: final weight. pass state along
# v.T @ kwi * fw = v.T @ k * inv_incl_pref * final_weight
# tldr we add contributions from kv everywhere
# u: how much we know each b writes. so u.T [!!] @ (b * inv_incl_pref) * fw = output to each channel of the state.
# not multiplying by q to get final output!

# ----

# backward pass !

# u: can be pulled from forward pass.
# du: derivative to u. u = ab_inv @ (ak @ v + wa_state). y = ... + qb @ u
# so du = qb.T @ dy. what is bwi_dw_dstate doing here?
# u is also used for determining the next state. specifically, state = ... + u.T @ (b * inv_incl_pref * fw)
# so we find du += (dstate.T @ (b * inv_incl_pref * fw)).T = bwi * fw @ dstate.T  (correct!!)
# dab_u: derivative to ak @ v + wa_state (the input a embedding) we just take du @ ab_inv.T

# dab/dak: derivative to ab/ak. dak = dab_u @ v.T obviously
# meanwhile getting dab from dab_u is complicated. dab_inv = du @ (ak @ v + wa_state).T
# dab = -ab_inv.T @ du @ (ak @ v + wa_state).T @ ab_inv.T =
# = -ab_inv.T @ du @ (ab_inv @ (ak @ v + wa_state)).T
# = -ab_inv.T @ du @ u = dab_u @ u
# waow

# for q, we don't need to worry about gradients from state!
# y = qk @ v + qb @ u + wq_state
# dqb/dqk: dqb = dy @ u.T; dqk = dy @ v.T
# ez


# compute gradient to previous state!
# dstate =
# fw * dstate_next
# + fw * dy.T @ (r * incl_pref)
# -- so we multiply each pair of dy and r * incl_pref
# -- basically, the previous state also feeds in here where it can pass thru r and then go to y directly
# + dab_u.T @ (a * incl_pref)
# -- all other state contributions flow thru a
# we know the derivative to the input a projection, we know what a weights were. a is created by multiplying
# initial state in part so we just forward it thru

# remember: state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)

# fw_u_dstate: fw * u @ dstate this is from the next dstate.
# u is the output to b. so this is contribution of u to next state.
# use this to decide how to update b. wire together u and next dstate

# fw_v_dstate: fw * v @ dstate_next. contribution of kwi to next state
# use this to decide how to wire together k. we already know how k decays.

# dab_u_state: dab_u @ state. used as gradient for a. i guess this is the gradient to past state, but for a?
# since wa_state = a * non_incl_pref @ state

# dy_state: dy @ state. gradient to wire together r from past state directly
# state_dstate: sum(state * dstate, axis=0). ????


# finale
# play hopes and dreams from undertale

# a gradient: non_incl_pref * (dab @ (b * inv_incl_pref) + dak @ (k * inv_incl_pref) + dab_u_state)
# remember: ab = a * non_incl_pref @ (b * inv_incl_pref).T; we know gradient to ab, it makes perfect sense
# next part. ak = a * non_incl_pref @ (k * inv_incl_pref).T. same thing.
# dab_u_state = dab_u @ state. remember ab_u = ak @ v + a * non_incl_pref @ state
# we computed gradient through ak, but we also need to include the @ state to be complete

# r gradient: incl_pref * (dqb @ (b * inv_incl_pref) + dqk @ (k * inv_incl_pref) + dy_state)
# qb = incl_pref @ (b * inv_incl_pref).T given that we care about immediate forwarding it makes sense
# qk - same thing
# dy_state = dy @ state. remember y = ... + r * non_incl_pref @ state

# b gradient: inv_incl_pref * (ab.T @ (a * non_incl_pref) + dqb.T @ (r * incl_pref) + fw_u_dstate)
# ab, qb: boring
# fw_u_dstate = fw * u @ dstate_next
# bwi = b * inv_incl_pref
# remember state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)
# so state = ... + u.T @ (b * inv_incl_pref * fw)
# ok

# k gradient: inv_incl_pref * (dak.T @ (a * non_incl_pref) + dqk.T @ (r * incl_pref) + fw_v_dstate)
# obvious
# kwi = k * inv_incl_pref
# state = ... + v.T @ (k * inv_incl_pref * fw)
# fw_v_dstate = fw * v @ dstate_next

# w gradient: ...

# dw0 - sum(state * dstate_next, axis=0) (don't need to multiply by fw, that's from exp)
# just the weight that should go to make the previous state influence next gradient
# then we loop over timesteps?..

#   fast_dw(dab, wa, bwi) + fast_dw(dak, wa, kwi)
# + fast_dw(dqb, wq, bwi) + fast_dw(dqk, dq, kwi)
# natural question: what is fast_dw?

# ?????????????????????????????????????????????????????????

# cumsum(v_dstate * (fw * kwi))
# cumsum(u_dstate * (fw * bwi))
# cumsum(dab_u_state * wa)
# cumsum(dy_state * wq)

# dw = fw * dw;
# dw += fast_dw<1>(dab,wa,bwi);
# dw += fast_dw<1>(dak,wa,kwi);
# dw += fast_dw<0>(dqb,wq,bwi);
# dw += fast_dw<0>(dqk,wq,kwi);
# FTile tmp;
# dw += cumsumv<0,0>(tmp = v_dstate*(fw*kwi));
# dw += cumsumv<0,0>(tmp = u_dstate*(fw*bwi));
# dw += cumsumv<0,1>(tmp = dab_u_state0*wa);
# dw += cumsumv<1,1>(tmp = dy_state0*wq);


#     dw0 = fw * state_dstate
#     for k in range(t0*dT,t0*dT+dT):
#         lmask = (t<k).trans()
#         A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
#         A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
#         A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
#         A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
#         dw = tl.sum(A, axis=0,keep_dims=True) + dw0

#         wk = tl.load(w_+IND4(bi,k,hi,j, T,H,C)).to(tl.float32)
#         dw *= -wk.exp()
#         tl.store(dw_+IND4(bi,k,hi,j, T,H,C), dw)


