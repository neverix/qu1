from pathlib import Path
from functools import partial
from dataclasses import replace
from typing import Literal
import os

from loguru import logger
from safetensors.numpy import save_file
import jax.numpy as jnp
from torch.utils.data import DataLoader
import equinox as eqx
import optax
import jax
from miditok import REMI
from simple_parsing import Serializable, parse
import wandb
from tqdm import trange
from jax import sharding
import numpy as np

from .models import RWKVConfig, RWKV
from .rwkv_kernels import rwkv_update, serial_rwkv
from .midi_data import MidiDataset


class TrainConfig(Serializable):
    tokens_path: Path = Path("data/tokens.dat")
    lengths_path: Path = Path("data/lengths.dat")
    seed: int = 42
    batch_size: int = 16
    max_tokens_in_batch: int = 4096
    max_steps: int = 1_000_000
    warmup_steps: int = 50
    learning_rate: float = 1e-5
    save_every: int = 1000
    save_dir: Path = Path("models")
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    architecture: RWKVConfig = RWKVConfig(
        n_layers=16,
        d_model=512,
        vocab_size=1024,
        n_head=4,
        d_ff=2048,
    )


def main():
    os.environ["LIBTPU_INIT_ARGS"] = (
        os.getenv("LIBTPU_INIT_ARGS", "") + " "
        "--xla_tpu_enable_latency_hiding_scheduler=true "
        "--xla_enable_async_collective_permute=true "
        "--xla_tpu_enable_ag_backward_pipelining=true "
        "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
        "--xla_tpu_data_parallel_opt_different_sized_ops=true "
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
        "--xla_tpu_overlap_compute_collective_tc=true "
        "--xla_enable_async_all_gather=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_megacore_fusion_allow_ags=true "
        "TPU_MEGACORE=MEGACORE_DENSE "
    )
    
    tokenizer = REMI()
    args = parse(TrainConfig)
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
    mesh = sharding.Mesh(np.array(jax.devices("tpu")).reshape(-1, 1), ("dp", "mp"))
    data_sharding = sharding.NamedSharding(mesh, sharding.PartitionSpec("dp", None))
    
    # Initialize wandb
    wandb.init(project="qu1", config=vars(args))
    
    logger.info(f"Loading dataset from {args.tokens_path} and {args.lengths_path}")
    logger.info(f"Batch size: {args.batch_size}, max tokens in batch: {args.max_tokens_in_batch}")
    logger.info(f"Architecture: {args.architecture}")
    
    dataset = MidiDataset(args.tokens_path, args.lengths_path,
                          max_tokens_in_batch=args.max_tokens_in_batch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = replace(
        args.architecture,
        vocab_size=tokenizer.vocab_size
    )
    with jax.default_device(jax.devices("cpu")[0]):
        rwkv = RWKV(config=config, key=jax.random.key(args.seed),
                    state_update=eqx.filter_checkpoint(partial(rwkv_update,
                                        fn=serial_rwkv
                                        )))
    with jax.sharding.use_mesh(mesh):
        rwkv = jax.tree.map(lambda x: jax.device_put(x) if isinstance(x, jax.Array) else x, rwkv)
    
    param_count = sum(p.size for p in jax.tree.flatten(eqx.filter(rwkv, eqx.is_array))[0] if isinstance(p, jax.Array))
    logger.info(f"Parameter count: {param_count}")
    approx_flops = args.batch_size * args.max_tokens_in_batch * param_count * 6
    
    optimizer = optax.chain(
        # optax.clip_by_global_norm(1.0),
        optax.adam(args.learning_rate),
        optax.scale_by_schedule(optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=1.0,
            warmup_steps=args.warmup_steps,
            decay_steps=args.max_steps - args.warmup_steps,
            end_value=0.0,
            exponent=1.0,
        ))
    )
    opt_state = optimizer.init(eqx.filter(rwkv, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(rwkv, tokens, restart_mask):
        logits = rwkv(tokens, new_mask=restart_mask)
        logprobs = jax.nn.log_softmax(logits)
        y_pred = logprobs[..., :-1, :]
        y_true = tokens[..., 1:]
        loss = -jnp.take_along_axis(y_pred, y_true[..., None], axis=-1).mean()
        return loss

    @eqx.filter_jit
    def train_step(rwkv, tokens, restart_mask, opt_state):
        loss, grads = loss_fn(rwkv, tokens, restart_mask)
        updates, opt_state = optimizer.update(grads, opt_state)
        rwkv = eqx.apply_updates(rwkv, updates)
        return loss, rwkv, opt_state
    
    for training_step, (tokens, restart_mask) in zip(bar := trange(args.max_steps), dataloader):
        if training_step % args.save_every == 0:
            state_dict_numpy = jax.tree.map(lambda x: np.asarray(x), eqx.filter(rwkv, eqx.is_array))
            state_dict_flat = jax.tree.flatten(state_dict_numpy)[0]
            args.save_dir.mkdir(parents=True, exist_ok=True)
            save_file({str(k): v for k, v in enumerate(state_dict_flat)}, args.save_dir / f"rwkv.safetensors")
        
        tokens, restart_mask = \
            jnp.asarray(tokens.numpy(), device=jax.devices("cpu")[0]), \
            jnp.asarray(restart_mask.numpy(), device=jax.devices("cpu")[0])
        with jax.sharding.use_mesh(mesh):
            tokens = jax.device_put(tokens, data_sharding)
            restart_mask = jax.device_put(restart_mask, data_sharding)
            loss, rwkv, opt_state = train_step(rwkv, tokens, restart_mask, opt_state)
        itps = bar.format_dict["rate"]
        if itps is not None:
            one_v4_chip_flops = 275 * 1e12
            mfu = (itps * approx_flops) / one_v4_chip_flops
        else:
            mfu = 0.0
        wandb.log({"step": training_step, "loss": float(loss), "mfu": mfu,
                   "tokens_processed": training_step * args.batch_size * args.max_tokens_in_batch},
                  step=training_step)
        bar.set_postfix({"mfu": mfu, "loss": float(loss)})
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
