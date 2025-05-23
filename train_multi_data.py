import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm

# Import our custom modules
from S5Basecaller import S5Basecaller
from NanoporeMultiDataset import MultiFileNanoporeDataset, nanopore_collate, create_data_splits, IDX2BASE


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


def train_with_multiple_files(
        data_dir,
        save_dir="models",
        model_name="s5_basecaller",
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        model_dim=128,
        num_classes=5,
        depth=4,
        max_len=2000,
        start_idx=0,
        end_idx=99,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        use_tensorboard=False
):
    """Train the S5 basecaller model using multiple NPZ files"""

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)

    print(f"Setting up data from {data_dir} (files {start_idx}-{end_idx})")

    # Create train/val/test splits
    train_dataset, val_dataset, test_dataset = create_data_splits(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        start_idx=start_idx,
        end_idx=end_idx,
        max_len=max_len
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=nanopore_collate,
        num_workers=4,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=nanopore_collate,
        num_workers=2
    )

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model
    model = S5Basecaller(input_dim=1, model_dim=model_dim, num_classes=num_classes, depth=depth).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Setup optimizer, loss and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ctc_loss = nn.CTCLoss(blank=4, reduction='mean', zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Setup tensorboard if requested
    if use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # Training loop
    print(f"Starting training for {num_epochs} epochs")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        start_time = time.time()

        # Process batches
        for i, (x, y, x_lens, y_lens) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")):
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

            # Use actual input lengths, clipped to the maximum sequence length
            input_lengths = torch.clamp(x_lens, max=log_probs.size(1)).to(device)

            # Compute loss
            loss = ctc_loss(log_probs_transposed, targets, input_lengths, target_lengths)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()

            # Print every 50 batches
            if (i + 1) % 50 == 0:
                sample_idx = 0  # First example in batch
                pred_seq = decode_ctc(log_probs)[sample_idx]
                true_seq = "".join([IDX2BASE[idx.item()] for idx in y[sample_idx] if idx.item() != 4])
                print(f"Batch {i + 1} loss: {loss.item():.4f}")
                print(f"Sample prediction: {pred_seq}")
                print(f"Sample ground truth: {true_seq}")

        # End of training epoch
        train_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for x, y, x_lens, y_lens in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                log_probs = model(x)
                log_probs_transposed = log_probs.permute(1, 0, 2)

                # Prepare targets
                targets = []
                target_lengths = []
                for seq in y:
                    target = seq[seq != 4]
                    targets.append(target)
                    target_lengths.append(len(target))

                targets = torch.cat(targets)
                target_lengths = torch.tensor(target_lengths, device=device)
                input_lengths = torch.clamp(x_lens, max=log_probs.size(1)).to(device)

                # Compute validation loss
                loss = ctc_loss(log_probs_transposed, targets, input_lengths, target_lengths)
                val_loss += loss.item()

        # End of validation
        val_time = time.time() - val_start_time
        avg_val_loss = val_loss / len(val_loader)

        # Update LR scheduler
        scheduler.step(avg_val_loss)

        # Log metrics
        print(f"Epoch {epoch + 1} complete:")
        print(f"  Train: loss={avg_train_loss:.4f}, time={train_time:.2f}s")
        print(f"  Val: loss={avg_val_loss:.4f}, time={val_time:.2f}s")

        if use_tensorboard:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, f"{save_path}.ckpt.{epoch + 1}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{save_path}.best.pt")
            print(f"  New best model saved with val_loss={best_val_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), f"{save_path}.final.pt")
    print(f"Training complete. Final model saved to {save_path}.final.pt")
    print(f"Best validation loss: {best_val_loss:.4f}")

    if use_tensorboard:
        writer.close()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train S5 Basecaller model with multiple files")
    parser.add_argument("--data_dir", required=True, help="Directory containing NPZ files")
    parser.add_argument("--save_dir", default="models", help="Directory to save models")
    parser.add_argument("--model_name", default="s5_basecaller", help="Base name for saved model files")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model_dim", type=int, default=128, help="Model dimension")
    parser.add_argument("--depth", type=int, default=4, help="Number of S5 blocks")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length")
    parser.add_argument("--start_idx", type=int, default=0, help="First file index")
    parser.add_argument("--end_idx", type=int, default=99, help="Last file index")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Proportion for training")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Proportion for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Proportion for testing")
    parser.add_argument("--tensorboard", action="store_true", help="Enable tensorboard logging")

    args = parser.parse_args()

    train_with_multiple_files(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_dim=args.model_dim,
        depth=args.depth,
        max_len=args.max_len,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_tensorboard=args.tensorboard
    )