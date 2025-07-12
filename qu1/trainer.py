from pathlib import Path
from functools import partial
from dataclasses import replace
from typing import Literal

from loguru import logger
import jax.numpy as jnp
from torch.utils.data import DataLoader
import equinox as eqx
import optax
import jax
from miditok import REMI
from simple_parsing import Serializable, parse
import wandb
from tqdm import trange

from .models import RWKVConfig, RWKV
from .rwkv_kernels import rwkv_update, serial_rwkv
from .midi_data import MidiDataset


class TrainConfig(Serializable):
    tokens_path: Path = Path("data/tokens.dat")
    lengths_path: Path = Path("data/lengths.dat")
    seed: int = 42
    batch_size: int = 4
    max_tokens_in_batch: int = 8192
    max_steps: int = 100000
    warmup_steps: int = 50
    learning_rate: float = 1e-5
    log_every_n_steps: int = 1000
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    architecture: RWKVConfig = RWKVConfig(
        n_layers=16,
        d_model=512,
        vocab_size=1024,
        n_head=4,
        d_ff=2048,
    )


def main():
    tokenizer = REMI()
    args = parse(TrainConfig)
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    
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
    rwkv = RWKV(config=config, key=jax.random.key(args.seed),
                state_update=partial(rwkv_update,
                                     fn=serial_rwkv
                                     ))
    
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
    
    log_dict_buffer = []
    for train_step, (tokens, restart_mask) in zip(bar := trange(args.max_steps), dataloader):
        tokens, restart_mask = jnp.asarray(tokens.numpy()), jnp.asarray(restart_mask.numpy())
        loss, rwkv, opt_state = train_step(rwkv, tokens, restart_mask, opt_state)
        itps = bar.format_dict["rate"]
        if itps is not None:
            one_v4_chip_flops = 275 * 1e12
            mfu = (itps * approx_flops) / one_v4_chip_flops
        else:
            mfu = 0.0
        log_dict_buffer.append(dict(
            step=train_step,
            tokens_processed=train_step * args.batch_size * args.max_tokens_in_batch,
            loss=loss,
            mfu=mfu,
        ))
        if train_step % args.log_every_n_steps == 0:
            for buf in log_dict_buffer:
                wandb.log(buf | {"loss": float(buf["loss"])}, step=buf["step"])
            log_dict_buffer = []
        bar.set_postfix({"mfu": mfu})
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
