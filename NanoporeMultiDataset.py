import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import torch

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, '': 4}
IDX2BASE = {v: k for k, v in BASE2IDX.items()}


class NanoporeDataset(Dataset):
    def __init__(self, npz_path, max_len=2000):
        data = np.load(npz_path, allow_pickle=True)
        self.x_data = data['x']  # list of arrays
        self.y_data = data['y']  # list of arrays (str or char)
        self.max_len = max_len
        print(f"Loaded {len(self.x_data)} sequences from {npz_path}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # Convert to torch tensor
        x = torch.tensor(x, dtype=torch.float32)

        # Convert labels to indices
        y = [BASE2IDX.get(c, 4) for c in y]

        # Handle padding for input sequence
        if len(x) > self.max_len:
            # Truncate if longer than max_len
            x_padded = x[:self.max_len]
        else:
            # Create a new tensor with the right size and copy the data
            x_padded = torch.zeros(self.max_len, dtype=torch.float32)
            x_padded[:len(x)] = x

        # Create and pad the label tensor
        y_padded = torch.tensor(y + [4] * (self.max_len - len(y)), dtype=torch.long)[:self.max_len]

        # Add channel dimension for model input
        return x_padded.unsqueeze(-1), y_padded  # (L, 1), (L)


class MultiFileNanoporeDataset(ConcatDataset):
    """Dataset that combines multiple NPZ files"""

    def __init__(self, data_dir, file_pattern="data_{}.npz", start_idx=0, end_idx=99, max_len=2000):
        # Create a list of datasets
        datasets = []

        for i in range(start_idx, end_idx + 1):
            npz_path = os.path.join(data_dir, file_pattern.format(i))
            if os.path.exists(npz_path):
                datasets.append(NanoporeDataset(npz_path, max_len=max_len))
            else:
                print(f"Warning: File not found: {npz_path}")

        if not datasets:
            raise ValueError(f"No valid datasets found in range {start_idx} to {end_idx}")

        super().__init__(datasets)
        print(f"Combined {len(datasets)} files with {len(self)} total sequences")


def nanopore_collate(batch):
    """Custom collate function that keeps track of original lengths"""
    xs, ys = zip(*batch)

    # Get original lengths before padding
    x_lens = torch.tensor([min(len(x), x.size(0)) for x in xs], dtype=torch.long)
    y_lens = torch.tensor([torch.sum(y != 4).item() for y in ys], dtype=torch.long)

    # Stack into batches
    xs = torch.stack(xs)  # (B, L, 1)
    ys = torch.stack(ys)  # (B, L)

    return xs, ys, x_lens, y_lens


# Function to create train/val/test split from multiple files
def create_data_splits(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                       start_idx=0, end_idx=99, max_len=2000):
    """
    Create train/validation/test splits from multiple NPZ files

    Args:
        data_dir: Directory containing the NPZ files
        train_ratio, val_ratio, test_ratio: Split ratios (should sum to 1)
        start_idx, end_idx: Range of file indices to include
        max_len: Maximum sequence length

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"

    # Count total files
    file_count = 0
    for i in range(start_idx, end_idx + 1):
        if os.path.exists(os.path.join(data_dir, f"data_{i}.npz")):
            file_count += 1

    if file_count == 0:
        raise ValueError(f"No data files found in {data_dir}")

    # Calculate split indices
    train_end = start_idx + int(file_count * train_ratio)
    val_end = train_end + int(file_count * val_ratio)

    # Create the datasets
    train_dataset = MultiFileNanoporeDataset(data_dir, start_idx=start_idx, end_idx=train_end - 1, max_len=max_len)
    val_dataset = MultiFileNanoporeDataset(data_dir, start_idx=train_end, end_idx=val_end - 1, max_len=max_len)
    test_dataset = MultiFileNanoporeDataset(data_dir, start_idx=val_end, end_idx=end_idx, max_len=max_len)

    print(f"Created data splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset