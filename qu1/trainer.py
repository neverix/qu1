from pathlib import Path
from functools import partial
from dataclasses import replace

import jax.numpy as jnp
from torch.utils.data import DataLoader
import equinox as eqx
import optax
import jax
from miditok import REMI
from simple_parsing import Serializable, parse

from .models import RWKVConfig, RWKV
from .rwkv_kernels import rwkv_update
from .midi_data import MidiDataset


class TrainConfig(Serializable):
    tokens_path: Path = Path("data/tokens.dat")
    lengths_path: Path = Path("data/lengths.dat")
    seed: int = 42
    batch_size: int = 4
    max_tokens_in_batch: int = 32768
    architecture: RWKVConfig = RWKVConfig(
        n_layers=12,
        d_model=512,
        vocab_size=1024,
        n_head=8,
        d_ff=2048,
    )


def main():
    tokenizer = REMI()
    args = parse(TrainConfig)
    dataset = MidiDataset(args.tokens_path, args.lengths_path,
                          max_tokens_in_batch=args.max_tokens_in_batch)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    config = replace(
        args.architecture,
        vocab_size=tokenizer.vocab_size
    )
    rwkv = RWKV(config=config, key=jax.random.key(args.seed),
                state_update=partial(rwkv_update))
    optimizer = optax.adam(learning_rate=1e-4)
    opt_state = optimizer.init(rwkv)

    @eqx.filter_value_and_grad
    def loss_fn(rwkv, tokens, restart_mask):
        logits = rwkv(tokens)
        logprobs = jax.nn.log_softmax(logits)
        y_pred = logprobs[..., :-1, :]
        y_true = tokens[..., 1:]
        loss = jnp.take_along_axis(y_pred, y_true[..., None], axis=-1).mean()
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
        if i % 100 == 0:
            print(f"Step {i}, loss: {loss}")

if __name__ == "__main__":
    main()
