"""Label-free event checks that need the tracking data. Uses no ground truth.

Usage:
    python -m scripts.event_checks DATA_DIR base.csv other.csv [...]

For each CSV it prints, over the graded frames:

  Turnover timing (#3)
    ha_bounds   direct H<->A boundaries
    slow_n      boundaries with a ball-slowdown frame within 3 s
    slow_med    median offset (frames) boundary - nearest slowdown frame; 0 is best
    slow_mean   mean offset (frames)
    slow_le5    share of boundaries within 5 frames of a slowdown

    A ball-slowdown frame is the first frame of a run where the ball speed is at least 1 m/s
    lower than 3 frames (0.12 s) earlier, the earlier speed was at least 2 m/s, and a player of
    either team is within 2 m. A drop of 1 m/s in 0.12 s is about 8 m/s^2, ten times the
    rolling friction of grass, so it marks a touch. It does not use the team labels.

  Dead-ball timing (#4)
    d_spells    D spells (the D at the start of each half is not counted)
    d_lt3s      D spells shorter than 3 s
    st_restarts Smart Tagging set pieces in the graded halves
    live_no_rs  live spells that follow a D spell with no Smart Tagging restart near
                (set piece start - 6 s .. set piece start + 15 s)
    d_before_x  out-of-play set pieces (throw-in, corner, goal kick) whose D spell starts
                more than 1 s before the last observed line crossing in
                (set piece start - 30 s .. set piece start + 4 s)
"""

import os
import sys

import numpy as np
import pandas as pd

from src.loader import PITCH_LENGTH, PITCH_WIDTH, TOTAL_FRAMES, load_and_condition_tracking, parse_smart_tagging

FPS = 25
HALVES = [(11185, 90651), (110808, 190197)]
SLOWDOWN_LAG = 3
SLOWDOWN_MIN_DROP = 1.0
SLOWDOWN_MIN_SPEED = 2.0
SLOWDOWN_PLAYER_RADIUS = 2.0
SLOWDOWN_SEARCH = 75
OUT_OF_PLAY_TYPES = ("Throw-in", "Corner Kick", "Goal Kick")


def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for a, b in HALVES:
        mask[a : b + 1] = True
    return mask


def runs(labels):
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(labels)]))
    return list(zip(starts, stops, labels[starts]))


def slowdown_frames(tr):
    v = tr.ball_velocity
    valid = tr.ball_valid
    near = np.fmin(np.fmin(tr.d_H, tr.d_A), np.fmin(tr.d_H_gk, tr.d_A_gk))
    prev_v = np.concatenate((np.zeros(SLOWDOWN_LAG), v[:-SLOWDOWN_LAG]))
    prev_valid = np.concatenate((np.zeros(SLOWDOWN_LAG, bool), valid[:-SLOWDOWN_LAG]))
    with np.errstate(invalid="ignore"):
        drop = (
            valid & prev_valid & (prev_v >= SLOWDOWN_MIN_SPEED)
            & (prev_v - v >= SLOWDOWN_MIN_DROP) & (near <= SLOWDOWN_PLAYER_RADIUS)
        )
    edges = np.diff(np.concatenate(([0], drop.astype(np.int8))))
    return np.flatnonzero(edges == 1)


def line_crossings(tr, length=PITCH_LENGTH, width=PITCH_WIDTH):
    obs = tr.ball_observed
    x, y = tr.ball_x_metric, tr.ball_y_metric
    with np.errstate(invalid="ignore"):
        inside = obs & (x >= 0) & (x <= length) & (y >= 0) & (y <= width)
    outside = obs & ~inside
    idx = np.flatnonzero(obs)
    pairs = [(a, b) for a, b in zip(idx[:-1], idx[1:]) if inside[a] and outside[b]]
    return np.array([a + 1 for a, _ in pairs], dtype=int)


def turnover_stats(labels, graded, slow):
    bounds = [i for i in range(1, len(labels)) if graded[i] and labels[i - 1] in "HA" and labels[i] in "HA" and labels[i - 1] != labels[i]]
    offs = []
    for b in bounds:
        lo, hi = np.searchsorted(slow, [b - SLOWDOWN_SEARCH, b + SLOWDOWN_SEARCH + 1])
        cand = slow[lo:hi]
        if len(cand):
            offs.append(int(b - cand[np.argmin(np.abs(cand - b))]))
    offs = np.array(offs)
    return {
        "ha_bounds": len(bounds),
        "slow_n": len(offs),
        "slow_med": float(np.median(offs)) if len(offs) else np.nan,
        "slow_mean": round(float(np.mean(offs)), 1) if len(offs) else np.nan,
        "slow_le5": round(float(np.mean(np.abs(offs) <= 5)), 2) if len(offs) else np.nan,
    }


def dead_stats(labels, graded, set_pieces, crossings):
    segs = runs(labels)
    half_starts = {a for a, _ in HALVES}
    d_spells = [(s, e) for s, e, v in segs if v == "D" and graded[s] and s not in half_starts]
    sp_starts = np.array([p["start"] for p in set_pieces if graded[min(p["start"], len(graded) - 1)]])
    live_no_rs = 0
    for i in range(1, len(segs)):
        s, e, v = segs[i]
        if v in ("H", "A") and segs[i - 1][2] == "D" and graded[s] and s not in half_starts:
            if not np.any((sp_starts - 6 * FPS <= s) & (s <= sp_starts + 15 * FPS)):
                live_no_rs += 1
    d_before_x = 0
    for p in set_pieces:
        if p["type"] not in OUT_OF_PLAY_TYPES or not graded[p["start"]]:
            continue
        cand = crossings[(crossings >= p["start"] - 30 * FPS) & (crossings <= p["start"] + 4 * FPS)]
        if not len(cand):
            continue
        crossing = int(cand[-1])
        if labels[crossing] != "D":
            continue
        s = crossing
        while s > 0 and labels[s - 1] == "D":
            s -= 1
        if crossing - s > FPS:
            d_before_x += 1
    return {
        "d_spells": len(d_spells),
        "d_lt3s": sum(1 for s, e in d_spells if e - s < 3 * FPS),
        "st_restarts": len(sp_starts),
        "live_no_rs": live_no_rs,
        "d_before_x": d_before_x,
    }


def main(data_dir, paths):
    tr = load_and_condition_tracking(
        os.path.join(data_dir, "BAR-ATH_tactical_atd.txt"),
        os.path.join(data_dir, "BAR-ATH_tactical_atd.xml"),
        os.path.join(data_dir, "BAR-ATH_tactical_smart_tagging.xml"),
        TOTAL_FRAMES,
        (PITCH_LENGTH, PITCH_WIDTH),
        12,
        "active_bridging",
    )
    set_pieces = parse_smart_tagging(os.path.join(data_dir, "BAR-ATH_tactical_smart_tagging.xml"))["set_pieces"]
    graded = graded_mask(TOTAL_FRAMES)
    slow = slowdown_frames(tr)
    crossings = line_crossings(tr)
    rows = []
    for path in paths:
        labels = pd.read_csv(path)["label"].astype(str).to_numpy()
        rows.append({"file": os.path.basename(path), **turnover_stats(labels, graded, slow), **dead_stats(labels, graded, set_pieces, crossings)})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2:])
