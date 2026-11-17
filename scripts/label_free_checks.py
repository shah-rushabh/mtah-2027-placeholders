"""Label-free checks for one or more submission CSVs. Uses no ground truth.

Usage:
    python scripts/label_free_checks.py submission/Placeholders.csv submission/Placeholders_improved.csv

For each file it prints, per graded half and in total:
  spells      number of H/A possession spells (a run of one team, ball alive)
  mean_s      mean spell length in seconds
  lt2s        spells shorter than 2 s
  aba_lt3s    A-B-A sandwiches where the middle spell is shorter than 3 s
  dead        number of D spells
  H/A/D %     share of graded frames per label
If two or more files are given, it also prints how many frames differ from the first file.
"""

import sys

import numpy as np
import pandas as pd

FPS = 25.0
HALVES = {"half1": (11185, 90651), "half2": (110808, 190197)}


def runs(labels):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    return [(labels[s], e - s) for s, e in zip(starts, stops)]


def stats(labels):
    segs = runs(labels)
    live = [n for v, n in segs if v in ("H", "A")]
    aba = sum(
        1
        for i in range(1, len(segs) - 1)
        if segs[i][0] in ("H", "A")
        and segs[i - 1][0] == segs[i + 1][0]
        and segs[i - 1][0] in ("H", "A")
        and segs[i - 1][0] != segs[i][0]
        and segs[i][1] < 3 * FPS
    )
    n = len(labels)
    return {
        "spells": len(live),
        "mean_s": round(float(np.mean(live)) / FPS, 1) if live else 0.0,
        "lt2s": sum(1 for x in live if x < 2 * FPS),
        "aba_lt3s": aba,
        "dead": sum(1 for v, _ in segs if v == "D"),
        "H%": round(100 * float(np.mean(labels == "H")), 1) if n else 0.0,
        "A%": round(100 * float(np.mean(labels == "A")), 1) if n else 0.0,
        "D%": round(100 * float(np.mean(labels == "D")), 1) if n else 0.0,
    }


def main(paths):
    base = None
    rows = []
    for path in paths:
        labels = pd.read_csv(path)["label"].astype(str).to_numpy()
        parts = {name: labels[a : b + 1] for name, (a, b) in HALVES.items()}
        for name, part in parts.items():
            rows.append({"file": path, "part": name, **stats(part)})
        total = {"file": path, "part": "total"}
        h1, h2 = stats(parts["half1"]), stats(parts["half2"])
        for key in ("spells", "lt2s", "aba_lt3s", "dead"):
            total[key] = h1[key] + h2[key]
        graded = np.concatenate(list(parts.values()))
        both = stats(graded)
        for key in ("mean_s", "H%", "A%", "D%"):
            total[key] = both[key]
        rows.append(total)
        if base is None:
            base = labels
        else:
            print(f"{path}: {int(np.sum(labels != base)):,} frames differ from {paths[0]}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
