from dataclasses import dataclass, replace
from midi2audio import FluidSynth
from safetensors.numpy import load_file
from pathlib import Path
import equinox as eqx
from loguru import logger
import numpy as np
import jax
import jax.numpy as jnp
from tqdm import trange
from typing import Literal
from safetensors.numpy import load_file
from .models import RWKVConfig, RWKV
from .rwkv_kernels import rwkv_update, serial_rwkv
from .midi_data import MidiDataset
from .dumb_rng import dumb_prng_impl
from simple_parsing import Serializable, parse
from miditok import REMI

@dataclass
class EvalConfig(Serializable):
    restore_path: Path = Path("models/rwkv.safetensors")
    save_dir: Path = Path("samples")
    architecture: RWKVConfig = RWKVConfig(
        n_layers=16,
        d_model=512,
        vocab_size=1024,
        n_head=4,
        d_ff=2048,
    )
    seed: int = 143
    temperature: float = 0.5
    top_p: float = 0.9


def main():
    tokenizer = REMI()
    args = parse(EvalConfig)
    
    config = replace(args.architecture, vocab_size=tokenizer.vocab_size)
    logger.info(f"Creating model")
    model = RWKV(config=config, key=jax.random.key(0, impl=dumb_prng_impl), state_update=rwkv_update)
    model_params, model_def = eqx.partition(model, eqx.is_array)
    model_params_flat, model_params_def = jax.tree.flatten(model_params)
    logger.info(f"Loading model from {args.restore_path}")
    model_params_flat_loaded = load_file(args.restore_path)
    logger.info("Model loaded")
    model_params_flat_loaded = [jax.device_put(model_params_flat_loaded[str(i)]) for i, e in enumerate(model_params_flat)]
    model_params = jax.tree.unflatten(model_params_def, model_params_flat_loaded)
    model = eqx.combine(model_params, model_def)
    logger.info("Model ready")
    
    forward = eqx.filter_jit(lambda models, tokens, states: model(tokens, states, return_states=True))
    @eqx.filter_jit
    def sample(logits, key):
        logit = logits[0, -1, :]
        key, subkey = jax.random.split(key)
        logit = jax.nn.log_softmax(logit / args.temperature, axis=-1)
        probs = jax.nn.softmax(logit / args.temperature, axis=-1)
        probs_argsort = jnp.argsort(probs)
        reversed_probs_argsort = jnp.argsort(probs_argsort)
        probs_sorted = probs[probs_argsort]
        probs_cumsum = jnp.cumsum(probs_sorted)
        mask = probs_cumsum <= args.top_p
        mask = mask.at[0].set(True)
        logit = jnp.where(mask[reversed_probs_argsort], logit, -jnp.inf)
        logit = jax.nn.log_softmax(logit, axis=-1)
        return jax.random.categorical(subkey, logit, axis=-1), key
    
    tokens = [4]
    states = None
    key = jax.random.key(args.seed)
    logits_so_far = []
    try:
        for _ in (bar := trange(4096, desc="Generating")):
            tokens_jnp = jnp.array(tokens[-1:]).reshape(1, -1)
            logits, states = forward(model, tokens_jnp, states)
            logit_original = jax.nn.log_softmax(logits[0, -1, :], axis=-1)
            next_token, key = sample(logits, key)
            logits_so_far.append(float(logit_original[next_token]))
            tokens.append(int(next_token))
            bar.set_postfix(nll=np.mean((logits_so_far)))
    except KeyboardInterrupt:
        pass

    logger.info("Saving")
    decoded = tokenizer(np.array(tokens)[None])
    args.save_dir.mkdir(parents=True, exist_ok=True)
    decoded.dump_midi(args.save_dir / "sample.mid")  # type: ignore
    midi_path = args.save_dir / "sample.mid"
    wav_path = args.save_dir / "sample.wav"
    FluidSynth().midi_to_audio(str(midi_path), str(wav_path))

if __name__ == "__main__":
    main()
