import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm

from NanoporeDataset import NanoporeDataset, IDX2BASE, nanopore_collate

# Import from our S5 implementation
from S5Basecaller import S5Basecaller


def decode_ctc(predictions, blank=4):
    """Simple greedy CTC decoding"""
    pred = predictions.argmax(dim=-1).cpu().numpy()  # (B, L)
    decoded = []
    for p in pred:
        result = []
        prev = None
        for idx in p:
            if idx != blank and idx != prev:
                result.append(IDX2BASE[idx])
            prev = idx
        decoded.append("".join(result))
    return decoded


def train(npz_path, save_path="s5_basecaller.pt", num_epochs=10, batch_size=32,
          learning_rate=1e-3, model_dim=128, num_classes=5, depth=4, max_len=2000):
    """Train the S5 basecaller model"""

    # Load dataset
    dataset = NanoporeDataset(npz_path, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=nanopore_collate)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = S5Basecaller(input_dim=1, model_dim=model_dim, num_classes=num_classes, depth=depth).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss = nn.CTCLoss(blank=4, reduction='mean', zero_infinity=True)

    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for i, (x, y, x_lens, y_lens) in enumerate(tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            x = x.to(device)  # (B, L, 1)
            y = y.to(device)  # (B, L)

            # Forward pass
            log_probs = model(x)  # (B, L, C)

            # Prepare for CTC loss
            log_probs_transposed = log_probs.permute(1, 0, 2)  # (L, B, C) as required by CTCLoss

            # Create target sequences without padding
            targets = []
            target_lengths = []
            for seq in y:
                target = seq[seq != 4]  # Remove padding
                targets.append(target)
                target_lengths.append(len(target))

            targets = torch.cat(targets)
            target_lengths = torch.tensor(target_lengths, device=device)
            input_lengths = x_lens  # Use actual sequence lengths

            # Compute loss
            loss = ctc_loss(log_probs_transposed, targets, input_lengths, target_lengths)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            epoch_loss += loss.item()

            # Print every 20 batches
            if (i + 1) % 20 == 0:
                sample_idx = 0  # Show first example in batch
                pred_seq = decode_ctc(log_probs)[sample_idx]
                true_seq = "".join([IDX2BASE[idx.item()] for idx in y[sample_idx] if idx.item() != 4])
                print(f"Batch {i + 1} loss: {loss.item():.4f}")
                print(f"Sample prediction: {pred_seq}")
                print(f"Sample ground truth: {true_seq}")

        # End of epoch
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1} complete in {epoch_time:.2f}s, avg loss: {avg_loss:.4f}")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"{save_path}.ckpt.{epoch + 1}")

    # Save final model
    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Model saved to {save_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train S5 Basecaller model")
    parser.add_argument("--npz_path", required=True, help="Path to .npz dataset")
    parser.add_argument("--save_path", default="s5_basecaller.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--depth", type=int, default=4, help="Number of S5 blocks")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length")

    args = parser.parse_args()

    train(
        npz_path=args.npz_path,
        save_path=args.save_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_dim=args.model_dim,
        depth=args.depth,
        max_len=args.max_len
    )


# Training: python train.py --npz_path data_0.npz --epochs 15 --batch_size 32
# Evaluation: python evaluate.py --model_path s5_basecaller.pt --npz_path test_data.npz