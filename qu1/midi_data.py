from torch.utils.data import Dataset
import numpy as np

class MidiDataset(Dataset):
    def __init__(self, tokens_path, lengths_path, pad_to=128, max_tokens_in_batch=32768, pad_token_id=0):
        self.tokens = np.memmap(tokens_path, dtype=np.int32, mode="r")
        self.lengths = np.memmap(lengths_path, dtype=np.int32, mode="r")
        self.lengths_cumsum = np.cumsum(self.lengths)
        self.pad_to = pad_to
        self.pad_token_id = pad_token_id
        self.max_tokens_in_batch = max_tokens_in_batch

    def __len__(self):
        return len(self.lengths)

    def at(self, idx):
        start = self.lengths_cumsum[idx] - self.lengths[idx]
        sliced = self.tokens[start:start + self.lengths[idx]]
        pad_n = (len(sliced) + self.pad_to - 1) // self.pad_to * self.pad_to
        return np.pad(sliced, (0, pad_n - len(sliced)), constant_values=self.pad_token_id)

    def __getitem__(self, idx):
        elements = [self.at(idx)]
        starts = [0]
        for _ in range(self.max_tokens_in_batch // self.pad_to):
            if sum(map(len, elements)) >= self.max_tokens_in_batch:
                break
            starts.append(sum(map(len, elements)))
            idx = np.random.randint(len(self))
            elements.append(self.at(idx))
        combined = np.concatenate(elements)[:self.max_tokens_in_batch]
        mask = np.zeros(self.max_tokens_in_batch, dtype=np.int32)
        mask[starts] = 1
        return combined, mask
