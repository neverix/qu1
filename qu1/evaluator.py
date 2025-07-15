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
    
    @eqx.filter_jit
    def forward(model, tokens):
        return model(tokens)
    
    tokens = [4]
    key = jax.random.key(0)
    try:
        for _ in trange(128, desc="Generating"):
            tokens_jnp = jnp.array(tokens).reshape(1, -1)
            logits = forward(model, tokens_jnp)
            logit = logits[0, -1, :]
            key, subkey = jax.random.split(key)
            logit = logit + jax.random.gumbel(subkey, logit.shape)
            next_token = int(logit.argmax())
            tokens.append(next_token)
    except KeyboardInterrupt:
        pass

    logger.info("Saving")
    decoded = tokenizer(np.array(tokens)[None])
    args.save_dir.mkdir(parents=True, exist_ok=True)
    decoded.dump_midi(args.save_dir / "sample.mid")
    midi_path = args.save_dir / "sample.mid"
    wav_path = args.save_dir / "sample.wav"
    FluidSynth().midi_to_audio(str(midi_path), str(wav_path))

if __name__ == "__main__":
    main()
