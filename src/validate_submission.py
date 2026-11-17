# =============================================================================
# Format Validator for MTAH 2027 Possession Submission Files
#
# Adapted from official competition starter code:
# Copyright (c) 2026-2027 Metrica Sports (MIT License)
# =============================================================================
"""Check a submission file before you send it.

Run this on every file you are about to submit. It applies the exact formatting and
schema validation that the official scoring platform applies:
- Exactly 199,319 data rows
- Monotonically increasing integer frames 0..199318
- Headers: 'frame,label'
- Labels strictly in {'H', 'A', 'D', 'X'}
- No empty or blank cells

Usage:
    python -m src.validate_submission submission/Placeholders.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

LABELS = ("H", "A", "D", "X")
H, A, D, X = 0, 1, 2, 3
POSSESSION_LABELS = (H, A)
DEFAULT_TOTAL_FRAMES = 199319
DEFAULT_FPS = 25.0


def read_label_csv(path: str) -> pd.DataFrame:
    """Read a frame,label CSV and return DataFrame with normalized column names."""
    frame = pd.read_csv(path, dtype={"label": "object"})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "label" not in frame.columns or "frame" not in frame.columns:
        raise ValueError("expected columns 'frame' and 'label', got {}".format(list(frame.columns)))
    return frame


def validate_submission_table(frame: pd.DataFrame, n_expected: int):
    """Return None if table satisfies all schema requirements, else a one-line failure reason."""
    if len(frame) != n_expected:
        return "row count {} != {}".format(len(frame), n_expected)

    frames = frame["frame"].values
    if not np.issubdtype(frames.dtype, np.integer):
        return "'frame' column is not integer (dtype {})".format(frames.dtype)
    if not np.array_equal(frames, np.arange(n_expected)):
        return "'frame' column is not 0..{} in order".format(n_expected - 1)

    labels = frame["label"].values
    blank = pd.isna(labels)
    if blank.any():
        return "empty label in {} rows; write X to abstain".format(int(blank.sum()))

    cleaned = np.array([str(v).strip().upper() for v in labels], dtype=object)
    bad = sorted(set(cleaned) - set(LABELS))
    if bad:
        return "labels outside {}: {}".format(list(LABELS), bad[:5])
    return None


def label_codes(frame_or_labels) -> np.ndarray:
    """Map label strings to 0..3."""
    values = frame_or_labels
    if isinstance(values, pd.DataFrame):
        values = values["label"].values
    values = np.array([str(v).strip().upper() for v in values], dtype=object)

    codes = np.full(len(values), -1, dtype=np.int8)
    for index, name in enumerate(LABELS):
        codes[values == name] = index
    if (codes < 0).any():
        raise ValueError("labels outside {}: {}".format(list(LABELS), sorted(set(values[codes < 0]))))
    return codes


def valid_runs(codes: np.ndarray):
    """Find contiguous gameplay spans where labels are not X."""
    graded = (codes != X).astype(np.int8)
    edges = np.diff(np.concatenate(([0], graded, [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segments(codes: np.ndarray, runs):
    """Extract constant-label intervals within each active run."""
    out = []
    for start, stop in runs:
        window = codes[start:stop]
        if not len(window):
            continue
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            out.append((start + int(bounds[i]), start + int(bounds[i + 1]), int(window[bounds[i]])))
    return out


def possession_spells(codes: np.ndarray, runs):
    """Extract maximal continuous H or A possession spells."""
    return [(a, b, k) for a, b, k in segments(codes, runs) if k in POSSESSION_LABELS]


def describe(codes: np.ndarray, fps: float = DEFAULT_FPS):
    """Summarizes reported possession spells and average durations."""
    runs = valid_runs(codes)
    spells = possession_spells(codes, runs)
    n_alive = int(np.sum((codes == H) | (codes == A)))
    mean_duration = (n_alive / float(fps) / len(spells)) if len(spells) else 0.0
    return len(spells), mean_duration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="the frame,label CSV to check")
    parser.add_argument("--rows", type=int, default=DEFAULT_TOTAL_FRAMES, help="expected row count (default: 199319)")
    args = parser.parse_args()

    if not os.path.isfile(args.submission):
        print("FAIL  no such file: {}".format(args.submission))
        sys.exit(1)

    try:
        table = read_label_csv(args.submission)
    except Exception as error:
        print("FAIL  {}".format(os.path.basename(args.submission)))
        print("      could not read: {}".format(error))
        print("\nExpected a CSV with a header line 'frame,label' and one row per video frame.")
        sys.exit(1)

    reason = validate_submission_table(table, args.rows)
    if reason:
        print("FAIL  {}".format(os.path.basename(args.submission)))
        print("      {}".format(reason))
        print("\nSee SUBMISSION.md for the contract. The usual causes are an off-by-one at the "
              "start or end of the file, a missing header, or blank cells.")
        sys.exit(1)

    codes = label_codes(table)
    n_spells, mean_duration = describe(codes, DEFAULT_FPS)

    print("OK    {}".format(os.path.basename(args.submission)))
    print("      {} rows, frame 0..{}".format(len(codes), len(codes) - 1))
    print("\n      label counts")
    for index, name in enumerate(LABELS):
        count = int(np.sum(codes == index))
        print("        {}  {:>7}  {:5.1f}%".format(name, count, 100.0 * count / len(codes)))
    print("\n      reports {} possessions of mean {:.1f} s".format(n_spells, mean_duration))
    if int(np.sum(codes == X)):
        print("      (counted with each X as a gap)")
    print("\nValid. File satisfies all structural, column, row count, and alphabet requirements.")
