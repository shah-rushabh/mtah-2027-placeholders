"""Stage 6: Symmetric RLE Minimum-Dwell Pruner & Graded Formatter.

Merges spurious tracking micro-spells, performs context-aware dead-ball pruning
(distinguishing same-team tempo shielding H-D-H from inter-team turnover boundaries H-D-A),
enforces our internal Zero-'X' coverage heuristic on graded match halves, and exports the certified
199,319-row CSV.

Minimum-dwell thresholds and merge criteria are empirical heuristics grounded in
human athletic response limits, ball deceleration latency, and tempo maintenance.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 1. STATE CONSTANTS & MAPPINGS
# =============================================================================
STATE_H = 0  # Home Possession (FC Barcelona)
STATE_A = 1  # Away Possession (Athletic Club)
STATE_D = 2  # Dead Ball / Stoppage (Out of bounds, foul, set-piece preparation)
STATE_X = 3  # Abstention / Ungraded Non-Gameplay Window

CODE_TO_CHAR = {STATE_H: "H", STATE_A: "A", STATE_D: "D", STATE_X: "X"}
CHAR_TO_CODE = {"H": STATE_H, "A": STATE_A, "D": STATE_D, "X": STATE_X}

# =============================================================================
# 2. CERTIFIED MATCH WINDOW INTERVALS (BAR-ATH_tactical_atd.xml)
# =============================================================================
# Extracted directly from official match metadata XML:
# - Half 1 Graded Window: Frame 11185 (kickoff) to Frame 90651 (halftime whistle)
# - Half 2 Graded Window: Frame 110808 (tracking start) to Frame 190197 (fulltime whistle)
# - Total Match Frames: 199,319 frames at 25.0 fps (approx. 132.88 minutes total timeline)
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
TOTAL_FRAMES = 199319

# =============================================================================
# 3. DWELL FILTERING & PRUNING THRESHOLDS (Sports Science Derivations)
# =============================================================================
# 1. Minimum possession dwell: 30 frames = 1.20 seconds at 25 Hz.
# Our heuristic threshold: sustained team possession is assumed to require a settled dwell
# of >= 1.2s to distinguish controlled team agency from transient loose ball deflections or optical noise.
DEFAULT_MIN_POSSESSION_DWELL_FRAMES = 30

# 2. Minimum dead-ball dwell: 35 frames = 1.40 seconds at 25 Hz.
# Our heuristic threshold: dead-ball intervals are constrained to >= 1.4s based on the physical
# latency required for ball recovery and restart setup following a whistle or stoppage.
DEFAULT_MIN_DEAD_DWELL_FRAMES = 35

# 3. Minimum same-team dead dwell: 45 frames = 1.80 seconds at 25 Hz.
# When the same team has possession before and after a brief dead flick (H-D-H or A-D-A)
# inside the pitch, this represents a slow tempo pause or shield ("La Pausa"). Merging it
# prevents spurious possession spell fragmentation during natural pauses in play.
DEFAULT_MIN_SAME_TEAM_DEAD_DWELL_FRAMES = 45

# 4. First-time intentional action protection: 10 frames = 0.40 seconds at 25 Hz.
# An intentional first-time pass, flick, clearance, or shot confirmed in Stage 3 is
# genuine football possession and is protected from being pruned down to 10 frames.
INTENTIONAL_ACTION_PROTECT_MIN_FRAMES = 10

# 5. Legal restart protection: 15 frames = 0.60 seconds at 25 Hz.
# A legal restart kick corroborated by p_restart > 0.50 is protected down to 15 frames.
RESTART_PROTECT_MIN_FRAMES = 15

# 6. Stoppage setup touch merger thresholds:
# Players regularly tap or pass the ball during an active stoppage to set up a restart:
# - Merged if duration < 45 frames (1.8s)
# - Merged if duration < 100 frames (4.0s) and players are walking (P75 speed < 1.95 m/s = 8:33 min/km)
SETUP_TOUCH_MAX_SHORT_FRAMES = 45
SETUP_TOUCH_MAX_WALKING_FRAMES = 100
SETUP_TOUCH_WALKING_SPEED_CEILING = 1.95  # 7.02 km/h = 8:33 min/km (walking ceiling)

# 7. Inter-team micro-D midpoint split threshold: 25 frames = 1.0 second.
INTER_TEAM_MICRO_DEAD_THRESHOLD_FRAMES = 25

# 8. Relaxation loop convergence cap:
RELAXATION_MAX_ITERATIONS = 6

# 8b. Turnover confirmation (opt-in with confirm_turnover_frames):
# CHALLENGE.md: a change of possession needs *controlled* possession, for example
# "a player keeps the ball for a few seconds". We use 2 seconds = 50 frames.
# A short opponent spell between two spells of the same team (A-B-A) that is shorter
# than this is a touch, not a turnover, and goes back to team A.
DEFAULT_TURNOVER_CONFIRM_FRAMES = 50

# 8c. Turnover at the receiver (opt-in with move_turnovers_to_receiver):
# An H<->A boundary moves to the first frame where the new team controls the ball.
# Control means: a new-team player is within reach of the ball, nearer than any opponent,
# and the ball is slower than the human dribble ceiling (it has slowed down from a pass).
RECEIVER_MAX_SHIFT_FRAMES = 50       # 2.00 s: a boundary never moves further than this
RECEIVER_REACH_METERS = 1.8          # 1.80 m: lunging reach (same as control.py contact radius)
RECEIVER_SLOW_BALL_SPEED = 6.5       # 6.50 m/s: dribble ceiling; a faster ball is still in flight
RECEIVER_MIN_CONTROL_FRAMES = 3      # 0.12 s: control must last 3 frames, so a fly-by does not count
RECEIVER_MIN_SPELL_KEEP_FRAMES = 50  # 2.00 s: a move never makes a spell shorter than this
                                     # (a spell already shorter than this is not shortened)

# 9. Corroborating evidence search offsets & thresholds:
EVIDENCE_SEARCH_WINDOW_OFFSET = 5
STOPPAGE_CORROBORATING_PROBABILITY = 0.70
RESTART_CORROBORATING_PROBABILITY = 0.50

# 10. Normalized pitch boundary limits (IFAB Law 9):
NORM_COORD_MIN = 0.0
NORM_COORD_MAX = 1.0

# 11. 50-50 Duel abstention parameters:
# Used when strategic abstention strategy is enabled for contested 50-50 duels:
DUEL_ABSTENTION_DISTANCE_DELTA = 0.6  # 0.60m: Home and Away players equidistant within reach envelope
DUEL_ABSTENTION_PROXIMITY_CEILING = 2.0  # 2.00m: Both players actively contesting within immediate tackle radius
DUEL_ABSTENTION_BALL_SPEED_CEILING = 2.0  # 2.00 m/s = 7.20 km/h = 8:20 min/km (FIFA Zone 1/2 boundary)


# =============================================================================
# 4. RUN-LENGTH ENCODING & DECODING UTILITIES
# =============================================================================
def rle_encode(codes: np.ndarray) -> List[Tuple[int, int, int]]:
    """Encodes 1D state array into contiguous segments: [(start, stop_exclusive, code)].

    Runs in O(N) vectorized time using NumPy adjacent difference indexing.
    """
    if len(codes) == 0:
        return []
    diffs = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], diffs))
    stops = np.concatenate((diffs, [len(codes)]))
    values = codes[starts]
    return [(int(s), int(e), int(v)) for s, e, v in zip(starts, stops, values)]


def rle_decode(segments: List[Tuple[int, int, int]], length: int) -> np.ndarray:
    """Decodes contiguous segment list back into a dense 1D NumPy int8 array.

    Runs in O(S) time where S is the number of segments.
    """
    out = np.zeros(length, dtype=np.int8)
    for s, e, v in segments:
        out[s:e] = v
    return out


def _merge_adjacent(segments: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """Joins neighbouring segments that have the same state."""
    merged: List[Tuple[int, int, int]] = []
    for s, e, v in segments:
        if merged and merged[-1][2] == v:
            merged[-1] = (merged[-1][0], e, v)
        else:
            merged.append((s, e, v))
    return merged


def merge_unconfirmed_turnovers(
    segments: List[Tuple[int, int, int]],
    confirm_frames: int,
    p_restart: Optional[np.ndarray] = None,
) -> List[Tuple[int, int, int]]:
    """Gives short A-B-A opponent spells back to team A.

    A spell of team B between two spells of team A is kept only if it lasts at least
    `confirm_frames`, or if a legal restart is detected inside it. Shortest spells are
    merged first, and the pass repeats until nothing changes.
    """
    segments = _merge_adjacent(segments)
    while True:
        best_index = None
        best_duration = None
        for i in range(1, len(segments) - 1):
            seg_start, seg_end, seg_state = segments[i]
            prev_state = segments[i - 1][2]
            next_state = segments[i + 1][2]
            if seg_state not in (STATE_H, STATE_A):
                continue
            if prev_state != next_state or prev_state not in (STATE_H, STATE_A) or prev_state == seg_state:
                continue
            duration = seg_end - seg_start
            if duration >= confirm_frames:
                continue
            if p_restart is not None and np.any(p_restart[seg_start:seg_end] > RESTART_CORROBORATING_PROBABILITY):
                continue
            if best_duration is None or duration < best_duration:
                best_index, best_duration = i, duration
        if best_index is None:
            return segments
        seg_start, seg_end, _ = segments[best_index]
        segments[best_index] = (seg_start, seg_end, segments[best_index - 1][2])
        segments = _merge_adjacent(segments)


def team_control_mask(
    d_team: np.ndarray,
    d_other: np.ndarray,
    ball_velocity: np.ndarray,
    ball_valid: np.ndarray,
    reach: float = RECEIVER_REACH_METERS,
    slow_speed: float = RECEIVER_SLOW_BALL_SPEED,
    min_frames: int = RECEIVER_MIN_CONTROL_FRAMES,
) -> np.ndarray:
    """True on frames where one team controls the ball.

    A frame counts when the ball position is known, the team's nearest player is within
    `reach`, no opponent is nearer, and the ball is slower than `slow_speed`. Only runs of at
    least `min_frames` such frames are kept, so a ball that flies past a player does not count.
    NaN distances (player not tracked) never count.
    """
    with np.errstate(invalid="ignore"):
        other = np.where(np.isnan(d_other), np.inf, d_other)
        raw = (
            np.asarray(ball_valid, dtype=bool)
            & ~np.isnan(d_team)
            & (d_team <= reach)
            & (d_team < other)
            & (ball_velocity <= slow_speed)
        )
    if min_frames <= 1 or not np.any(raw):
        return raw
    out = np.zeros_like(raw)
    edges = np.diff(np.concatenate(([0], raw.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start >= min_frames:
            out[start:stop] = True
    return out


def _find_reception(ctrl_new: np.ndarray, ctrl_old: np.ndarray, boundary: int, lo: int, hi: int) -> Optional[int]:
    """Frame where the new team takes control, searched in [lo, hi]; None if there is none.

    1. If the new team controls the ball at the boundary, or gets control before it with no
       old-team control in between, the turnover is the earliest such control frame (the
       decoder switched late).
    2. Otherwise the turnover is the first new-team control frame after the boundary (the
       decoder switched while the ball was still in flight from the passer).
    """
    # Case 1: walk back from the boundary while the old team has no control.
    earliest = None
    t = boundary
    if t <= hi and ctrl_new[t]:
        earliest = t
    t = boundary - 1
    while t >= lo and not ctrl_old[t]:
        if ctrl_new[t]:
            earliest = t
        t -= 1
    if earliest is not None:
        return earliest
    # Case 2: first new-team control after the boundary.
    after = np.flatnonzero(ctrl_new[boundary + 1 : hi + 1])
    if len(after):
        return boundary + 1 + int(after[0])
    return None


def move_turnovers_to_receiver(
    codes: np.ndarray,
    d_home: np.ndarray,
    d_away: np.ndarray,
    ball_velocity: np.ndarray,
    ball_valid: np.ndarray,
    max_shift: int = RECEIVER_MAX_SHIFT_FRAMES,
    min_keep: int = RECEIVER_MIN_SPELL_KEEP_FRAMES,
    reach: float = RECEIVER_REACH_METERS,
    slow_speed: float = RECEIVER_SLOW_BALL_SPEED,
    min_control_frames: int = RECEIVER_MIN_CONTROL_FRAMES,
) -> np.ndarray:
    """Moves each direct H<->A boundary to the frame where the new team controls the ball.

    The frames of a pass in flight stay with the passer's team, because the boundary moves to
    the reception, not to the kick. Only boundaries between two live spells move; a boundary
    next to D or X does not. A boundary moves at most `max_shift` frames, and never makes a
    spell shorter than min(its length, `min_keep`). If no control frame is found in range, the
    boundary stays.

    Args:
        codes: State array {0: H, 1: A, 2: D, 3: X}.
        d_home, d_away: Distance from the ball to the nearest player of each team, goalkeeper
            included (NaN when unknown).
        ball_velocity: Ball speed in m/s.
        ball_valid: True where the ball position is known (observed or bridged).
    """
    ctrl = {
        STATE_H: team_control_mask(d_home, d_away, ball_velocity, ball_valid, reach, slow_speed, min_control_frames),
        STATE_A: team_control_mask(d_away, d_home, ball_velocity, ball_valid, reach, slow_speed, min_control_frames),
    }
    segments = [list(seg) for seg in _merge_adjacent(rle_encode(codes))]
    for i in range(1, len(segments)):
        prev_seg, next_seg = segments[i - 1], segments[i]
        old_state, new_state = prev_seg[2], next_seg[2]
        if old_state not in (STATE_H, STATE_A) or new_state not in (STATE_H, STATE_A) or old_state == new_state:
            continue
        boundary = next_seg[0]
        prev_len = boundary - prev_seg[0]
        next_len = next_seg[1] - boundary
        lo = max(boundary - max_shift, prev_seg[0] + min(prev_len, min_keep))
        hi = min(boundary + max_shift, next_seg[1] - min(next_len, min_keep))
        if lo > hi:
            continue
        reception = _find_reception(ctrl[new_state], ctrl[old_state], boundary, lo, hi)
        if reception is None:
            continue
        reception = int(min(max(reception, lo), hi))
        prev_seg[1] = reception
        next_seg[0] = reception
    return rle_decode([tuple(seg) for seg in segments], len(codes))


# =============================================================================
# 5. CONTEXT-AWARE SYMMETRIC DWELL PRUNER
# =============================================================================
def apply_symmetric_dwell_pruning(
    codes: np.ndarray,
    ball_x_norm: Optional[np.ndarray] = None,
    ball_y_norm: Optional[np.ndarray] = None,
    player_p75_speed: Optional[np.ndarray] = None,
    min_poss_dwell: int = DEFAULT_MIN_POSSESSION_DWELL_FRAMES,
    min_dead_dwell: int = DEFAULT_MIN_DEAD_DWELL_FRAMES,
    min_same_team_dead_dwell: Optional[int] = None,
    bridge_dead_ball: bool = True,
    prune_inter_team_micro_d: bool = False,
    p_restart: Optional[np.ndarray] = None,
    is_deflection: Optional[np.ndarray] = None,
    p_dead: Optional[np.ndarray] = None,
    intentional_action_H: Optional[np.ndarray] = None,
    intentional_action_A: Optional[np.ndarray] = None,
    confirm_turnover_frames: Optional[int] = None,
) -> np.ndarray:
    """Applies context-aware dwell filtering and physical bridge healing:

    Args:
        codes: Initial decoded state array of shape (N,).
        ball_x_norm: Optional normalized ball x-coordinates for out-of-bounds verification.
        ball_y_norm: Optional normalized ball y-coordinates for out-of-bounds verification.
        player_p75_speed: Optional 75th percentile player speed array for walking pace checks.
        min_poss_dwell: Minimum duration for H/A possession spells (default: 30 frames = 1.2s).
        min_dead_dwell: Minimum duration for Dead Ball intervals (default: 35 frames = 1.4s).
        min_same_team_dead_dwell: Minimum duration for H-D-H / A-D-A dead intervals (default: 45 frames = 1.8s).
        bridge_dead_ball: Enable merging setup touches inside dead ball stoppages (default: True).
        prune_inter_team_micro_d: Enable midpoint splitting of H-D-A micro D gaps (default: False).
        p_restart: Candidate restart probabilities to preserve short legal restarts.
        is_deflection: Header/deflection flags.
        p_dead: Stoppage evidence probabilities to preserve genuine stoppages.
        intentional_action_H: Intentional first-time passes/shots by Home.
        intentional_action_A: Intentional first-time passes/shots by Away.
        confirm_turnover_frames: Opt-in. When set, (1) a short spell is no longer kept just
            because an intentional kick happened in it, and (2) A-B-A spells of team B shorter
            than this are given back to team A (see merge_unconfirmed_turnovers). None keeps
            the submitted behaviour.

    Returns:
        Pruned state array of shape (N,).
    """
    segments = rle_encode(codes)
    is_modified = True
    iteration_count = 0

    # Iterative relaxation loop: terminates upon convergence or maximum 6 iterations
    while is_modified and iteration_count < RELAXATION_MAX_ITERATIONS:
        is_modified = False
        iteration_count += 1
        new_segments = []
        total_segments_count = len(segments)

        for i, (seg_start, seg_end, seg_state) in enumerate(segments):
            segment_duration_frames = seg_end - seg_start
            previous_segment_state = segments[i - 1][2] if i > 0 else None
            next_segment_state = segments[i + 1][2] if i < total_segments_count - 1 else None

            # -----------------------------------------------------------------
            # CASE A: Dead-Ball Interval (STATE_D)
            # -----------------------------------------------------------------
            if seg_state == STATE_D:
                # Check whether ball was geometrically out-of-bounds during this segment:
                is_ball_out_of_bounds = False
                if ball_x_norm is not None and ball_y_norm is not None:
                    ball_x_normalized = ball_x_norm[seg_start:seg_end]
                    ball_y_normalized = ball_y_norm[seg_start:seg_end]
                    valid_coords_mask = ~np.isnan(ball_x_normalized) & ~np.isnan(ball_y_normalized)
                    if np.any(valid_coords_mask):
                        if np.any(
                            (ball_x_normalized[valid_coords_mask] < NORM_COORD_MIN)
                            | (ball_x_normalized[valid_coords_mask] > NORM_COORD_MAX)
                            | (ball_y_normalized[valid_coords_mask] < NORM_COORD_MIN)
                            | (ball_y_normalized[valid_coords_mask] > NORM_COORD_MAX)
                        ):
                            is_ball_out_of_bounds = True

                # Check if this dead-ball interval has strong stoppage or restart evidence:
                has_confirmed_stoppage_evidence = False
                if p_dead is not None:
                    search_window_start = max(0, seg_start)
                    search_window_end = min(len(p_dead), seg_end)
                    if search_window_end > search_window_start and np.mean(p_dead[search_window_start:search_window_end]) > STOPPAGE_CORROBORATING_PROBABILITY:
                        has_confirmed_stoppage_evidence = True

                if p_restart is not None and not has_confirmed_stoppage_evidence:
                    search_window_start = max(0, seg_start - EVIDENCE_SEARCH_WINDOW_OFFSET)
                    search_window_end = min(len(p_restart), seg_end + EVIDENCE_SEARCH_WINDOW_OFFSET)
                    if np.any(p_restart[search_window_start:search_window_end] > RESTART_CORROBORATING_PROBABILITY):
                        has_confirmed_stoppage_evidence = True

                # A1: Between SAME team (H-D-H or A-D-A) and NOT Out-of-Bounds:
                # If there is confirmed stoppage/restart evidence, PRESERVE D!
                # Only merge if it was an unevidenced micro-dip in open play.
                effective_dead_dwell = (
                    min_same_team_dead_dwell
                    if min_same_team_dead_dwell is not None
                    else DEFAULT_MIN_SAME_TEAM_DEAD_DWELL_FRAMES
                )
                if (
                    previous_segment_state == next_segment_state
                    and previous_segment_state in (STATE_H, STATE_A)
                    and not is_ball_out_of_bounds
                ):
                    if not has_confirmed_stoppage_evidence and segment_duration_frames < effective_dead_dwell:
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                        continue

                # A2: Between DIFFERENT teams (H-D-A or A-D-H) and NOT Out-of-Bounds:
                # If inter-team micro-D pruning is explicitly requested, split ultra-short gaps at midpoint:
                if (
                    prune_inter_team_micro_d
                    and previous_segment_state in (STATE_H, STATE_A)
                    and next_segment_state in (STATE_H, STATE_A)
                    and previous_segment_state != next_segment_state
                    and not is_ball_out_of_bounds
                ):
                    if segment_duration_frames < INTER_TEAM_MICRO_DEAD_THRESHOLD_FRAMES:
                        midpoint_frame = seg_start + segment_duration_frames // 2
                        new_segments.append((seg_start, midpoint_frame, previous_segment_state))
                        new_segments.append((midpoint_frame, seg_end, next_segment_state))
                        is_modified = True
                        continue

                # A3: Boundary Edge Dead-Ball Flicker (< min_dead_dwell):
                if not is_ball_out_of_bounds and not has_confirmed_stoppage_evidence and segment_duration_frames < min_dead_dwell:
                    if (
                        previous_segment_state == next_segment_state
                        and previous_segment_state is not None
                        and previous_segment_state != STATE_X
                    ):
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                        continue
                    elif (previous_segment_state is None or previous_segment_state == STATE_X) and next_segment_state in (STATE_H, STATE_A):
                        new_segments.append((seg_start, seg_end, next_segment_state))
                        is_modified = True
                        continue
                    elif (next_segment_state is None or next_segment_state == STATE_X) and previous_segment_state in (STATE_H, STATE_A):
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                        continue

                new_segments.append((seg_start, seg_end, seg_state))

            # -----------------------------------------------------------------
            # CASE B: Live Possession (STATE_H or STATE_A)
            # -----------------------------------------------------------------
            elif seg_state in (STATE_H, STATE_A):
                # B1: Stoppage Setup Touch Merger (D -> H/A -> D):
                # When players tap or pass the ball during an active stoppage to set up a
                # dead-ball restart, merge the touch into D:
                # - If touch duration < 45 frames (1.8s)
                # - If touch duration < 100 frames (4.0s) and players are walking (< 1.95 m/s)
                if bridge_dead_ball and previous_segment_state == STATE_D and next_segment_state == STATE_D:
                    has_confirmed_restart_kick = False
                    if p_restart is not None and segment_duration_frames >= RESTART_PROTECT_MIN_FRAMES:
                        search_window_start = max(0, seg_start - EVIDENCE_SEARCH_WINDOW_OFFSET)
                        search_window_end = min(len(p_restart), seg_end + EVIDENCE_SEARCH_WINDOW_OFFSET)
                        if np.any(p_restart[search_window_start:search_window_end] > RESTART_CORROBORATING_PROBABILITY):
                            has_confirmed_restart_kick = True

                    if not has_confirmed_restart_kick:
                        if segment_duration_frames < SETUP_TOUCH_MAX_SHORT_FRAMES:
                            new_segments.append((seg_start, seg_end, STATE_D))
                            is_modified = True
                            continue
                        elif segment_duration_frames < SETUP_TOUCH_MAX_WALKING_FRAMES and player_p75_speed is not None:
                            mean_player_speed = np.mean(player_p75_speed[seg_start:seg_end])
                            if mean_player_speed < SETUP_TOUCH_WALKING_SPEED_CEILING:
                                new_segments.append((seg_start, seg_end, STATE_D))
                                is_modified = True
                                continue

                # B2: Evidence-Aware Sub-Dwell Possession (< min_poss_dwell = 30 frames):
                # Micro-spells under 1.2s are preserved IF corroborated by:
                # 1. Confirmed legal restart (dur >= 15 frames)
                # 2. Confirmed intentional action (first-time pass, flick, clearance, or shot >= 10 frames = 0.4s)
                has_corroborating_possession_evidence = False
                if segment_duration_frames >= RESTART_PROTECT_MIN_FRAMES:
                    if p_restart is not None and np.any(p_restart[seg_start:seg_end] > RESTART_CORROBORATING_PROBABILITY):
                        has_corroborating_possession_evidence = True

                if (
                    confirm_turnover_frames is None
                    and segment_duration_frames >= INTENTIONAL_ACTION_PROTECT_MIN_FRAMES
                    and not has_corroborating_possession_evidence
                ):
                    if seg_state == STATE_H and intentional_action_H is not None and np.any(intentional_action_H[seg_start:seg_end]):
                        has_corroborating_possession_evidence = True
                    elif seg_state == STATE_A and intentional_action_A is not None and np.any(intentional_action_A[seg_start:seg_end]):
                        has_corroborating_possession_evidence = True

                if segment_duration_frames < min_poss_dwell and not has_corroborating_possession_evidence:
                    if previous_segment_state == next_segment_state and previous_segment_state is not None:
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                    elif previous_segment_state == STATE_D and next_segment_state in (STATE_H, STATE_A):
                        new_segments.append((seg_start, seg_end, next_segment_state))
                        is_modified = True
                    elif next_segment_state == STATE_D and previous_segment_state in (STATE_H, STATE_A):
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                    elif previous_segment_state is not None and previous_segment_state != STATE_X:
                        new_segments.append((seg_start, seg_end, previous_segment_state))
                        is_modified = True
                    elif next_segment_state is not None and next_segment_state != STATE_X:
                        new_segments.append((seg_start, seg_end, next_segment_state))
                        is_modified = True
                    else:
                        new_segments.append((seg_start, seg_end, seg_state))
                else:
                    new_segments.append((seg_start, seg_end, seg_state))

            else:
                new_segments.append((seg_start, seg_end, seg_state))

        # Re-merge adjacent identical segments for the next relaxation iteration:
        merged_segments = []
        for s, e, v in new_segments:
            if merged_segments and merged_segments[-1][2] == v:
                prev_s, prev_e, prev_v = merged_segments[-1]
                merged_segments[-1] = (prev_s, e, prev_v)
            else:
                merged_segments.append((s, e, v))
        segments = merged_segments

    if confirm_turnover_frames is not None:
        segments = merge_unconfirmed_turnovers(segments, confirm_turnover_frames, p_restart=p_restart)

    return rle_decode(segments, len(codes))


# =============================================================================
# 6. FORMATTER & SUBMISSION EXPORT
# =============================================================================
def format_final_submission(
    codes: np.ndarray,
    n_frames: int = TOTAL_FRAMES,
    abstention_strategy: Optional[str] = None,
    d_H: Optional[np.ndarray] = None,
    d_A: Optional[np.ndarray] = None,
    ball_velocity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clamps non-gameplay intervals to 'X' and enforces graded window policies.

    Compliance Rules:
    1. Pre-match, Halftime, and Post-match non-gameplay intervals are strictly set to 'X'.
    2. Inside graded gameplay intervals (Half 1 and Half 2), our pipeline adopts a complete
       coverage heuristic (mapping residual in-play abstentions to 'D') because in-play intervals
       represent continuous active match time.
    3. Optional 'duel_abstention' strategy allows strategic abstention on contested 50-50 duels.

    Args:
        codes: Input state array of shape (n_frames,).
        n_frames: Total number of frames (199,319).
        abstention_strategy: Optional 'duel_abstention' flag.
        d_H: Optional Home player distance array.
        d_A: Optional Away player distance array.
        ball_velocity: Optional ball velocity array.

    Returns:
        Formatted state array of shape (n_frames,).
    """
    out_codes = codes.copy()

    # Clamp non-gameplay intervals strictly to 'X':
    out_codes[0:H1_START] = STATE_X
    out_codes[H1_END + 1 : H2_START] = STATE_X
    out_codes[H2_END + 1 : n_frames] = STATE_X

    # Handle graded gameplay intervals:
    if abstention_strategy == "duel_abstention" and d_H is not None and d_A is not None:
        duel_mask = (
            (np.abs(d_H - d_A) < DUEL_ABSTENTION_DISTANCE_DELTA)
            & (np.minimum(d_H, d_A) < DUEL_ABSTENTION_PROXIMITY_CEILING)
            & (ball_velocity < DUEL_ABSTENTION_BALL_SPEED_CEILING if ball_velocity is not None else True)
        )
        for start_idx, end_idx in [(H1_START, H1_END), (H2_START, H2_END)]:
            half_window = out_codes[start_idx : end_idx + 1]
            sub_duel = duel_mask[start_idx : end_idx + 1]
            half_window[sub_duel & (half_window != STATE_D)] = STATE_X
            out_codes[start_idx : end_idx + 1] = half_window
    else:
        # Default Zero-'X' policy on graded intervals:
        for start_idx, end_idx in [(H1_START, H1_END), (H2_START, H2_END)]:
            half_window = out_codes[start_idx : end_idx + 1]
            unclassified_mask = half_window == STATE_X
            if np.any(unclassified_mask):
                half_window[unclassified_mask] = STATE_D
            out_codes[start_idx : end_idx + 1] = half_window

    return out_codes


def export_submission_csv(
    codes: np.ndarray,
    output_path: str,
) -> None:
    """Exports the official 199,319-row CSV with columns 'frame,label'.

    Validates:
    - Length exactly equals 199,319 rows.
    - Alphabet strictly in {'H', 'A', 'D', 'X'}.
    - CSV format matches the competition submission specification.
    """
    assert len(codes) == TOTAL_FRAMES, f"Expected {TOTAL_FRAMES} rows, got {len(codes)}"
    labels = [CODE_TO_CHAR[int(code_val)] for code_val in codes]
    submission_dataframe = pd.DataFrame({"frame": np.arange(len(codes)), "label": labels})
    submission_dataframe.to_csv(output_path, index=False)
