import numpy as np
from torch.utils.data import Dataset
import torch

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '-': 4}
IDX2BASE = {v: k for k, v in BASE2IDX.items()}


class NanoporeDataset(Dataset):
    def __init__(self, npz_path, max_len=2000):
        data = np.load(npz_path, allow_pickle=True)
        self.x_data = data['x']  # list of arrays
        self.y_data = data['y']  # list of arrays (str or char)
        self.max_len = max_len

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Normalize and pad
        x = torch.tensor(x, dtype=torch.float32)
        y = [BASE2IDX.get(c, 4) for c in y]

        # Pad to max_len - handle 1D arrays correctly
        if len(x) > self.max_len:
            x_padded = x[:self.max_len]
        else:
            # For 1D tensors, padding is (padding_left, padding_right)
            x_padded = torch.nn.functional.pad(x, (0, self.max_len - len(x)))

        y_padded = torch.tensor(y + [4] * (self.max_len - len(y)))[:self.max_len]

        # Return with proper dimensions
        return x_padded.unsqueeze(-1), y_padded  # (L, 1), (L,)


def nanopore_collate(batch):
    """Custom collate function that keeps track of original lengths"""
    xs, ys = zip(*batch)

    # Get original lengths before padding
    x_lens = torch.tensor([len(x) for x in xs])
    y_lens = torch.tensor([torch.sum(y != 4) for y in ys])

    # Stack into batches
    xs = torch.stack(xs)  # (B, L, 1)
    ys = torch.stack(ys)  # (B, L)

    return xs, ys, x_lens, y_lens