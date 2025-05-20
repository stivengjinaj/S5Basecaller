import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import our custom modules
from s5_basecaller import S5Basecaller
from multi_file_dataset import MultiFileNanoporeDataset, nanopore_collate, IDX2BASE


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


def evaluate_multi_file(
        model_path,
        data_dir,
        output_dir="results",
        start_idx=90,  # Assuming files 90-99 are for testing
        end_idx=99,
        max_len=2000,
        batch_size=8,
        num_examples_to_print=5
):
    """Evaluate a trained S5 basecaller model on multiple files"""
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "evaluation_results.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = S5Basecaller(input_dim=1, model_dim=128, num_classes=5, depth=4)

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
    test_dataset = MultiFileNanoporeDataset(
        data_dir=data_dir,
        start_idx=start_idx,
        end_idx=end_idx,
        max_len=max_len
    )

    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=nanopore_collate,
        num_workers=2
    )
    print(f"Loaded evaluation dataset with {len(test_dataset)} sequences")

    # Metrics to track
    total_lev_dist = 0
    total_length = 0
    correct = 0
    total = 0
    base_errors = {'insertions': 0, 'deletions': 0, 'substitutions': 0}

    # Store detailed results for later analysis
    all_results = []

    examples_printed = 0

    print("Starting evaluation...")
    with torch.no_grad(), open(results_file, 'w') as f:
        f.write("Prediction,Ground Truth,Edit Distance,Accuracy\n")

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
                accuracy = 1 - (dist / len(gt) if len(gt) > 0 else 0)

                total_lev_dist += dist
                total_length += len(gt)
                correct += int(pred == gt)
                total += 1

                # Write to results file
                f.write(f"\"{pred}\",\"{gt}\",{dist},{accuracy:.4f}\n")

                # Store for analysis
                all_results.append({
                    'pred': pred,
                    'gt': gt,
                    'dist': dist,
                    'acc': accuracy
                })

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
                    print(f"Accuracy: {accuracy:.2%}")
                    examples_printed += 1

    # Calculate final metrics
    cer = total_lev_dist / total_length if total_length > 0 else 0
    seq_acc = correct / total if total > 0 else 0
    error_breakdown = {k: v / total_lev_dist if total_lev_dist > 0 else 0 for k, v in base_errors.items()}

    # Write summary to results file
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        f.write("====== Evaluation Results ======\n")
        f.write(f"Total Sequences: {total}\n")
        f.write(f"Sequence Accuracy: {seq_acc:.4f}\n")
        f.write(f"Character Error Rate (CER): {cer:.4f}\n")
        f.write(f"Error Breakdown:\n")
        f.write(f"  Insertions: {error_breakdown['insertions']:.2%}\n")
        f.write(f"  Deletions: {error_breakdown['deletions']:.2%}\n")
        f.write(f"  Substitutions: {error_breakdown['substitutions']:.2%}\n")

    # Print summary
    print("\n====== Evaluation Results ======")
    print(f"Total Sequences: {total}")
    print(f"Sequence Accuracy: {seq_acc:.4f}")
    print(f"Character Error Rate (CER): {cer:.4f}")
    print(f"Error Breakdown:")
    print(f"  Insertions: {error_breakdown['insertions']:.2%}")
    print(f"  Deletions: {error_breakdown['deletions']:.2%}")
    print(f"  Substitutions: {error_breakdown['substitutions']:.2%}")
    print(f"\nDetailed results saved to {results_file}")
    print(f"Summary saved to {os.path.join(output_dir, 'summary.txt')}")

    # Generate sequence length analysis
    try:
        import matplotlib.pyplot as plt
        import pandas as pd

        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(all_results)

        # Add sequence length columns
        results_df['gt_len'] = results_df['gt'].apply(len)
        results_df['pred_len'] = results_df['pred'].apply(len)

        # Group by ground truth length and calculate average accuracy
        length_acc = results_df.groupby(pd.cut(results_df['gt_len'], bins=10))['acc'].mean()

        # Plot
        plt.figure(figsize=(10, 6))
        length_acc.plot(kind='bar')
        plt.title('Accuracy vs Sequence Length')
        plt.xlabel('Sequence Length')
        plt.ylabel('Average Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_length.png'))
        print(f"Generated length analysis plot at {os.path.join(output_dir, 'accuracy_vs_length.png')}")
    except Exception as e:
        print(f"Could not generate plots: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate S5 Basecaller model with multiple files")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.pt)")
    parser.add_argument("--data_dir", required=True, help="Directory containing NPZ files")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--start_idx", type=int, default=90, help="First file index for evaluation")
    parser.add_argument("--end_idx", type=int, default=99, help="Last file index for evaluation")
    parser.add_argument("--max_len", type=int, default=2000, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to print")

    args = parser.parse_args()

    evaluate_multi_file(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_examples_to_print=args.examples
    )