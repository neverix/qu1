from pathlib import Path
from functools import partial
from dataclasses import replace

from loguru import logger
import jax.numpy as jnp
from torch.utils.data import DataLoader
import equinox as eqx
import optax
import jax
from miditok import REMI
from simple_parsing import Serializable, parse
import wandb

from .models import RWKVConfig, RWKV
from .rwkv_kernels import rwkv_update, serial_rwkv
from .midi_data import MidiDataset


class TrainConfig(Serializable):
    tokens_path: Path = Path("data/tokens.dat")
    lengths_path: Path = Path("data/lengths.dat")
    seed: int = 42
    batch_size: int = 1
    max_tokens_in_batch: int = 16384
    architecture: RWKVConfig = RWKVConfig(
        n_layers=12,
        d_model=512,
        vocab_size=1024,
        n_head=4,
        d_ff=2048,
    )


def main():
    tokenizer = REMI()
    args = parse(TrainConfig)
    
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
                state_update=partial(rwkv_update, fn=serial_rwkv))
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(rwkv, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(rwkv, tokens, restart_mask):
        logits = rwkv(tokens)
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
    
    for i, (tokens, restart_mask) in enumerate(dataloader):
        tokens, restart_mask = jnp.asarray(tokens.numpy()), jnp.asarray(restart_mask.numpy())
        loss, rwkv, opt_state = train_step(rwkv, tokens, restart_mask, opt_state)
        wandb.log({"loss": float(loss), "step": i})
    
    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
