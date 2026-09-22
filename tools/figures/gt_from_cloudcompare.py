import argparse
import os
import glob
import numpy as np
from typing import Optional


def _iter_data_lines(path: str):
    """Yield non-empty, non-comment lines."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("//"):
                continue
            yield s


def _try_parse_floats(tokens):
    try:
        return [float(t) for t in tokens]
    except Exception:
        return None


def load_ascii_xyz_label(
    input_path: str,
    label_col: int,
    delimiter: Optional[str],
):
    """Load ASCII file with at least XYZ + label column.

    - Supports optional header line (non-numeric). If the first data line is non-numeric, it is skipped.
    - label_col is 0-based column index.
    """
    rows = []
    lines = _iter_data_lines(input_path)

    first = None
    for first in lines:
        break

    if first is None:
        raise ValueError(f"Empty file: {input_path}")

    # Detect header: if cannot parse floats, treat as header and read next line.
    tokens = first.split(delimiter) if delimiter else first.split()
    parsed = _try_parse_floats(tokens)
    if parsed is None:
        # Skip header and proceed
        first = None
        for first in lines:
            break
        if first is None:
            raise ValueError(f"No numeric data lines found in: {input_path}")
        tokens = first.split(delimiter) if delimiter else first.split()
        parsed = _try_parse_floats(tokens)
        if parsed is None:
            raise ValueError("Failed to parse numeric data after header.")

    # Process first numeric line
    if len(parsed) <= max(2, label_col):
        raise ValueError(f"Not enough columns in first data row. Need label_col={label_col}.")
    rows.append(parsed)

    for s in lines:
        tokens = s.split(delimiter) if delimiter else s.split()
        parsed = _try_parse_floats(tokens)
        if parsed is None:
            continue
        if len(parsed) <= max(2, label_col):
            continue
        rows.append(parsed)

    data = np.asarray(rows, dtype=np.float64)
    xyz = data[:, :3]
    label_raw = data[:, label_col]
    # Normalize to 0/1: any non-zero -> 1
    label = (label_raw != 0).astype(np.int64)
    return xyz, label


def save_gt_txt(output_path: str, xyz: np.ndarray, label: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = np.concatenate([xyz.astype(np.float64), label.reshape(-1, 1).astype(np.int64)], axis=1)
    # Use a compact format: 6 decimals for xyz, integer for label
    fmt = ["%.6f", "%.6f", "%.6f", "%d"]
    np.savetxt(output_path, out, fmt=fmt)


def save_gt_npy(output_path: str, xyz: np.ndarray, label: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out = np.concatenate([xyz.astype(np.float64), label.reshape(-1, 1).astype(np.int64)], axis=1)
    np.save(output_path, out)


def main():
    p = argparse.ArgumentParser(
        description="Convert CloudCompare ASCII export to evaluation ground-truth (_gt.txt/_gt.npy)."
    )
    p.add_argument("--input", required=True, help="CloudCompare exported ASCII/TXT file or glob pattern")
    p.add_argument("--output", required=True, help="Output path: .txt/.npy or directory if multiple inputs")
    p.add_argument(
        "--label-col",
        type=int,
        default=3,
        help="0-based column index of semantic_label in the input file (default: 3 => 4th col)",
    )
    p.add_argument(
        "--delimiter",
        default=None,
        help="Optional delimiter (default: whitespace). Example: ','",
    )
    args = p.parse_args()

    # Expand input if it's a glob or directory
    input_pattern = args.input
    inputs = []
    
    if os.path.isdir(input_pattern):
        # If input is a directory, look for .txt files
        inputs = sorted(glob.glob(os.path.join(input_pattern, "*.txt")))
    elif any(ch in input_pattern for ch in ["*", "?", "["]):
        # Glob pattern
        inputs = sorted(glob.glob(input_pattern, recursive=True))
    else:
        # Single file
        inputs = [input_pattern]

    if len(inputs) == 0:
        raise FileNotFoundError(f"No input files found for: {args.input}")

    # If multiple inputs, output must be a directory
    # Robust check for trailing slashes on Windows/Linux mixed environment
    out_is_trailing = args.output.endswith("/") or args.output.endswith("\\")
    out_is_dir = out_is_trailing or os.path.isdir(args.output)

    if len(inputs) > 1 and not out_is_dir:
        # If output doesn't exist yet but looks like a file path (no trailing slash), it's ambiguous.
        # But if we have multiple inputs, we MUST output to a directory.
        raise ValueError(f"Processing {len(inputs)} files, but --output does not look like a directory (must end with / or \\). Got: {args.output}")

    if out_is_dir:
        os.makedirs(args.output, exist_ok=True)

    for inp in inputs:
        try:
            print(f"Processing: {inp}")
            xyz, label = load_ascii_xyz_label(inp, label_col=args.label_col, delimiter=args.delimiter)
        except Exception as e:
            print(f"Error reading {inp}: {e}")
            continue

        if out_is_dir:
            # Output to directory with suffix
            base = os.path.splitext(os.path.basename(inp))[0]
            # If input ends in _gt, avoid double _gt_gt
            if base.endswith("_gt"):
                out_name = base + ".txt"
            else:
                out_name = base + "_gt.txt"
            
            output_path = os.path.join(args.output, out_name)
            save_gt_txt(output_path, xyz, label)
            print(f"  -> Saved: {output_path}")
            
            # Optional: Report stats
            n1 = int(label.sum())
            n0 = int(len(label) - n1)
            print(f"     Points: {len(label)}  Cabbage: {n1}  Background: {n0}")
            
        else:
            # Single file explicit output
            out_ext = os.path.splitext(args.output)[1].lower()
            if out_ext == ".txt":
                save_gt_txt(args.output, xyz, label)
            elif out_ext == ".npy":
                save_gt_npy(args.output, xyz, label)
            else:
                raise ValueError("--output must end with .txt or .npy")
            
            print(f"  -> Saved: {args.output}")
            n1 = int(label.sum())
            n0 = int(len(label) - n1)
            print(f"     Points: {len(label)}  Cabbage: {n1}  Background: {n0}")



if __name__ == "__main__":
    main()
