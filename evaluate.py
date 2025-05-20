import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from NanoporeDataset import NanoporeDataset, IDX2BASE, nanopore_collate
# Import from our files
from S5Basecaller import (S5Basecaller)


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


def string_from_targets(targets, blank=4):
    """Convert target tensors to strings by removing blanks"""
    strings = []
    for seq in targets:
        s = [IDX2BASE[i.item()] for i in seq if i.item() != blank]
        strings.append("".join(s))
    return strings


def levenshtein(s1, s2):
    """Calculate Levenshtein edit distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous[j + 1] + 1
            deletions = current[j] + 1
            substitutions = previous[j] + (c1 != c2)
            current.append(min(insertions, deletions, substitutions))
        previous = current

    return previous[-1]


def evaluate(model_path, npz_path, max_len=2000, num_examples_to_print=5):
    """Evaluate a trained S5 basecaller model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = S5Basecaller()

    # Handle both direct state dict and checkpoint formats
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model state dict directly")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Load evaluation dataset
    dataset = NanoporeDataset(npz_path, max_len=max_len)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=nanopore_collate)
    print(f"Loaded evaluation dataset with {len(dataset)} sequences")

    # Metrics to track
    total_lev_dist = 0
    total_length = 0
    correct = 0
    total = 0
    base_errors = {'insertions': 0, 'deletions': 0, 'substitutions': 0}

    examples_printed = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for x, y, x_lens, y_lens in tqdm(loader, desc="Evaluating"):
            x = x.to(device)  # (B, L, 1)
            y = y.to(device)  # (B, L)

            # Forward pass
            log_probs = model(x)  # (B, L, C)

            # Decode predictions
            decoded_preds = decode_ctc(log_probs)

            # Get ground truth strings
            y_strs = string_from_targets(y)

            # Compute metrics
            for pred, gt in zip(decoded_preds, y_strs):
                dist = levenshtein(pred, gt)
                total_lev_dist += dist
                total_length += len(gt)
                correct += int(pred == gt)
                total += 1

                # Calculate error types (simplified approximation)
                len_diff = len(pred) - len(gt)
                if len_diff > 0:
                    base_errors['insertions'] += len_diff
                elif len_diff < 0:
                    base_errors['deletions'] += abs(len_diff)
                # Assume remaining edit distance is due to substitutions
                base_errors['substitutions'] += max(0, dist - abs(len_diff))

                # Print example predictions
                if examples_printed < num_examples_to_print:
                    print(f"\n[Example {examples_printed + 1}]")
                    print(f"Prediction: {pred}")
                    print(f"Ground Truth: {gt}")
                    print(f"Edit Distance: {dist}")
                    print(f"Length: pred={len(pred)}, gt={len(gt)}")
                    examples_printed += 1

    # Calculate final metrics
    cer = total_lev_dist / total_length if total_length > 0 else 0
    acc = correct / total if total > 0 else 0
    error_breakdown = {k: v / total_lev_dist if total_lev_dist > 0 else 0 for k, v in base_errors.items()}

    print("\n====== Evaluation Results ======")
    print(f"Total Sequences: {total}")
    print(f"Sequence Accuracy: {acc:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Error Breakdown:")
    print(f"  Insertions: {error_breakdown['insertions']:.2%}")
    print(f"  Deletions: {error_breakdown['deletions']:.2%}")
    print(f"  Substitutions: {error_breakdown['substitutions']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate S5 Basecaller model")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.pt)")
    parser.add_argument("--npz_path", required=True, help="Path to .npz dataset")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to print")

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        npz_path=args.npz_path,
        max_len=args.max_len,
        num_examples_to_print=args.examples
    )