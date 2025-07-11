#%%
import glob
from datasets import Dataset, load_dataset
from miditok import REMI
from symusic import Score
import os

tokenizer = REMI()

def tokenize_midi(midi_file_path):
    midi_file_path = midi_file_path["midi_file_path"]
    score = Score.from_midi(open(midi_file_path, "rb").read())
    tokens, = tokenizer(score)
    return {"music": midi_file_path, "tokens": tokens.ids}


# Find all midi files in the extracted directory
midi_files = glob.glob('data/aria-midi-v1-pruned-ext/**/*.mid', recursive=True)

# Filter out potential non-midi files or invalid paths if necessary
# For now, assuming glob finds only valid paths

# Create a Hugging Face Dataset from the list of file paths
dataset = Dataset.from_dict({"midi_file_path": midi_files})

tokens = tokenize_midi(dataset[0])
#%%
# Map the tokenization function over the dataset
tokenized_dataset = dataset.map(tokenize_midi, remove_columns=["midi_file_path"], num_proc=os.cpu_count())

# Print the first example to check
tokenized_dataset[0]
#%%
tokenized_dataset.push_to_hub("nev/aria-remi", private=True)
#%%
import datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
lengths = []
for x in tqdm(datasets.load_dataset("nev/aria-remi", split="train")):
    lengths.append(len(x["tokens"]))
#%%
import numpy as np
bins = np.logspace(2, 6, 100)
plt.xscale("log")
plt.hist(lengths, bins=bins)
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.title("Length Distribution of Aria MIDI Dataset")
plt.show()
#%%
import datasets
all_tokens = datasets.load_dataset("nev/aria-remi", split="train")["tokens"]
# %%
import numpy as np
lengths = np.array([len(x) for x in all_tokens], dtype=np.int32)
combined_tokens = np.array([x for y in all_tokens for x in y], dtype=np.int32)
# %%
tokens_mmap = np.memmap("../data/tokens.dat", dtype=np.int32, mode="w+", shape=(len(combined_tokens),))
tokens_mmap[:] = combined_tokens
#%%
lengths_mmap = np.memmap("../data/lengths.dat", dtype=np.int32, mode="w+", shape=(len(lengths),))
lengths_mmap[:] = lengths
# %%
