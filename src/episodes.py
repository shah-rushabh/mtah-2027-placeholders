"""Phase 2: Game Phase & Episode Evidence Engine.

Computes factorized probabilistic evidence for macroscopic football episodes:
  - Game Status: p_dead, p_live (with kinematic stoppage hysteresis & 90-min invariants)
  - Ball Motion: p_fast_transit (high-speed ball in unobstructed flight)
  - Restart Phase: p_restart_candidate, restart_team (set-piece impulse backtracking)
  - Control State: p_contested (50-50 physical challenges)

Does NOT make irreversible hard classifications; produces continuous evidence arrays
consumed by Stage 3 (Emissions) and Stage 4 (Viterbi Trellis Decoder).
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .loader import (
    AWAY_TEAM_CODE,
    FPS,
    HOME_TEAM_CODE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TOTAL_FRAMES,
    TrackingData,
)


# ==============================================================================
# PHYSICAL, GEOMETRIC & KINEMATIC CONSTANTS
# ==============================================================================
# Pitch Boundary & Geometry
PITCH_BOUNDARY_BUFFER_METERS = 1.0         # 1.0m tolerance beyond touchline absorbing optical parallax
PENALTY_BOX_DEPTH_METERS = 16.5            # FIFA 18-yard penalty box depth from goal line
PENALTY_BOX_WIDTH_MARGIN_METERS = 13.84     # Sideline distance to box edge on 68m pitch ((68 - 40.32) / 2)
GOAL_LINE_CROSSING_MARGIN_METERS = 2.5     # Goal line crossing proximity to confirm goal celebration

# Hysteresis & Temporal Quanta (frames @ 25 Hz)
STOPPAGE_ENTER_SUSTAINED_FRAMES = 20       # 0.80s: sustained collective slowdown before entering Dead Ball
STOPPAGE_EXIT_SUSTAINED_FRAMES = 12        # 0.48s: sustained ballistic movement before exiting Dead Ball
OOB_MEMORY_FRAMES = 150                    # 6.00s: countdown memory of out-of-bounds event
GK_HOLD_MAX_FRAMES = 200                   # 8.00s: legal goalkeeper hold limit under IFAB Law 12
GK_BALL_PROXIMITY_RADIUS_METERS = 1.5      # 1.50m: reaching/smothering distance to goalkeeper
RESTART_LIVE_BURST_FRAMES = 20             # 0.80s: immediate live burst post-restart guaranteeing zero latency
PRE_KICK_WINDOW_FRAMES = 12                # 0.48s: pre-kick stationary ball verification window
OOB_RETRIEVAL_LOOKBACK_FRAMES = 375        # 15.0s: lookback for ball retrieval before corner/goal kick clips
SET_PIECE_WHISTLE_LOOKBACK_FRAMES = 150     # 6.00s: lookback before XML clip start to catch foul whistle
SET_PIECE_SEARCH_WINDOW_FRAMES = 350       # 14.0s: search window within clip to pinpoint kick impulse
KICKOFF_CLIP_OFFSET_FRAMES = 50            # 2.00s: kickoff strike offset into kickoff clip
BRIDGE_HEALING_MAX_GAP_FRAMES = 75         # 3.00s: maximum false live gap healed inside dead ball

# Kinematic Sampling & Rolling Displacement Quanta
BALL_DISPLACEMENT_WINDOW_FRAMES = 25       # 1.00s: displacement measurement window across frames
BALL_DISPLACEMENT_STRIDE_FRAMES = 5        # 0.20s: stride interval for rolling displacement calculation
KICK_IMPULSE_SUSTAINED_FRAMES = 10         # 0.40s: sustained displacement evaluation window following foot strike
OOB_ROUTINE_FORWARD_FRAMES = 100           # 4.00s: forward window into set-piece routine to bridge out-of-bounds gap

# XML Event Search Windows & Proportion Ratios
GOAL_KICKOFF_PRE_WINDOW_FRAMES = 250       # 10.0s: search lookback before goal event for kickoff clip alignment
GOAL_KICKOFF_POST_WINDOW_FRAMES = 500      # 20.0s: search forward window after goal event for kickoff clip alignment
GOAL_CROSSING_SEARCH_FRAMES = 500          # 20.0s: forward search window from goal start to find goal-line crossing
GOAL_CROSSING_FALLBACK_RATIO = 0.25        # 25% into goal interval as fallback crossing frame if unobserved
FALLBACK_SET_PIECE_PREP_RATIO = 0.35       # 35% into clip duration assumed preparation phase if no kick impulse found

# Kinematic Velocity & Displacement Thresholds
BALL_STATIONARY_DISPLACEMENT_METERS = 3.0  # Over 2.0s, displacement < 3.0m indicates stationary ball
PLAYER_WALKING_SPEED_CEILING = 1.5         # 1.50 m/s (5.4 km/h, 11:07 min/km): 75th percentile walking pace
PLAYER_JOGGING_SPEED_FLOOR = 2.2           # 2.20 m/s (7.9 km/h, 7:35 min/km): player acceleration surge on restart
PLAYER_ACTIVE_FRACTION_CEILING = 0.15      # <= 15% active runners (> 2.5 m/s) required for stoppage
PRE_KICK_MAX_BALL_SPEED = 2.0              # 2.00 m/s (7.2 km/h): ball must be stationary prior to kick
KICK_IMPULSE_MIN_VELOCITY = 2.5            # 2.50 m/s (9.0 km/h): minimum velocity spike of a foot strike
KICK_IMPULSE_MIN_DISPLACEMENT = 1.0        # 1.00m: sustained flight displacement over 10 frames (0.4s)
THROW_IN_FIELD_MARGIN_METERS = 0.5         # 0.50m inside line: boundary penetration under IFAB Law 15
THROW_IN_MIN_VELOCITY = 2.0                # 2.00 m/s (7.2 km/h): hand throw velocity threshold
SHORT_RESTART_MIN_VELOCITY = 2.8           # 2.80 m/s (10.1 km/h): rolling velocity for short pass/goal kick
SHORT_RESTART_MIN_DISPLACEMENT = 1.2       # 1.20m: minimum travel for short pass restart
DRIVEN_RESTART_MIN_VELOCITY = 4.5          # 4.50 m/s (16.2 km/h): deliberate set-piece strike velocity
DRIVEN_RESTART_MIN_DISPLACEMENT = 2.0      # 2.00m: travel distance for driven kick
RESTART_BREAKOUT_BALL_DISPLACEMENT_METERS = 1.8  # 1.80m: multi-stride ball travel indicating live breakout
RESTART_BREAKOUT_ACTIVE_FRACTION_FLOOR = 0.20    # > 20% active runners required to confirm collective play resumption

# Fast Ball Transit Scaling (Step 2.6)
FAST_TRANSIT_VELOCITY_FLOOR = 6.5          # 6.50 m/s (23.4 km/h, 2:34 min/km): human dribble ceiling; faster = flight
FAST_TRANSIT_BASE_PROBABILITY = 0.30       # 0.30: baseline jump probability when dribbling ceiling is exceeded
FAST_TRANSIT_RAMP_SPAN = 5.0               # 5.00 m/s: scaling divisor, reaches p=1.0 at 10.0 m/s (36.0 km/h)

# 50-50 Contested Duel Bounds (Step 2.6)
DUEL_PHYSICAL_REACH_RADIUS = 1.8           # 1.80m: lunging tackle reach of a professional footballer
DUEL_EQUIDISTANCE_TOLERANCE = 0.8          # 0.80m: proximity difference below which duel is 50-50 contested

# Deflection Bounds (Step 2.6)
DEFLECTION_MIN_ANGLE_DEGREES = 35.0        # 35.0 deg: angular deflection indicating sharp ricochet
DEFLECTION_MIN_RESIDUAL_VELOCITY = 4.0     # 4.00 m/s (14.4 km/h): minimum post-contact rebound speed

# Probability Output Levels & Priors
PROB_DEAD_CONFIRMED = 0.95                 # Emitted during confirmed dead-ball state
PROB_DEAD_TRANSITION_BUFFER = 0.50         # Emitted during entry ramp-up (frame 1..19)
PROB_DEAD_LIVE_BASELINE = 0.05             # Default live play baseline
PROB_DEAD_GOAL_LOCKED = 0.99               # Locked during post-goal celebrations
PROB_DEAD_SET_PIECE_LOCKED = 0.96          # Locked during set-piece routine setups
PROB_DEAD_SET_PIECE_FALLBACK = 0.90        # Emitted when set-piece preparation is detected without impulse frame
PROB_DEAD_RESTART_BURST = 0.02             # Post-restart immediate live burst
PROB_DEAD_HYSTERESIS_ACTIVE_FLOOR = 0.85   # Probability threshold identifying established dead-ball intervals
DEFAULT_ACTIVE_FRACTION_FALLBACK = 0.20    # 20%: default active outfield runner fraction when uncomputed
DEFAULT_EPISODE_CONFIDENCE = 0.90          # 0.90: default baseline certainty for computed episodes

# Dead-ball anchors (opt-in with dead_ball_anchors=True, improvement #4)
# Start: D starts when the ball crosses a pitch line (IFAB Law 9), not a fixed time before the
# Smart Tagging clip. End: D ends at the restart found inside the Smart Tagging clip.
OUT_OF_PLAY_RESTART_TYPES = ("Throw-in", "Corner Kick", "Goal Kick")
CROSSING_LOOKBACK_FRAMES = 750             # 30.0s: a ball that went out longer ago than this is not this restart's cause
CROSSING_AFTER_CLIP_START_FRAMES = 100     # 4.00s: the ball goes out before the Smart Tagging clip starts, or within 4 s after
CROSSING_CONFIRM_DEPTH_METERS = 0.5        # 0.50m: the ball must go this far past the line, or be lost, to confirm the crossing
CROSSING_CONFIRM_GAP_FRAMES = 13           # 0.52s: a gap longer than the ballistic micro-gap after the line also confirms it
CROSSING_MAX_EXTRAPOLATION_FRAMES = 50     # 2.00s: longest straight-line extrapolation of a lost ball to a line
RESTART_ON_PITCH_MARGIN_METERS = 0.5       # 0.50m: a kicked ball is on the pitch, lines and parallax included
RESTART_ON_PITCH_WITHIN_FRAMES = 12        # 0.48s: a restarted ball is seen on the pitch this soon (an aerial ball
                                           # first seen in flight projects off the pitch for a few frames)
KICKOFF_CENTRE_MAX_DISTANCE_METERS = 5.0   # 5.00m: the kick-off ball passes this close to the centre spot (tracking error included)
POST_RESTART_CLEAR_MAX_FRAMES = 250        # 10.0s: longest dead-state carry-over cleared after a restart
THROW_IN_MIN_INWARD_METERS = 0.5           # 0.50m: a thrown ball moves at least this far into the pitch in RESTART_ON_PITCH_WITHIN_FRAMES
# Without a line crossing, a set-piece lock starts after the last frame of clear live play:
# the ball seen on the pitch faster than FAST_TRANSIT_VELOCITY_FLOOR, or players (75th
# percentile) faster than PLAYER_JOGGING_SPEED_FLOOR.


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class EpisodeEvidence:
    """Factorized probabilistic evidence per frame across the entire match."""
    p_dead: np.ndarray              # shape (N,), float32 in [0, 1]: probability match is in dead-ball stoppage
    p_live: np.ndarray              # shape (N,), float32 in [0, 1]: probability match is in active live play
    p_fast_transit: np.ndarray      # shape (N,), float32 in [0, 1]: probability ball is in unobstructed flight
    p_restart_candidate: np.ndarray # shape (N,), float32 in [0, 1]: impulse spike on legal set-piece kick
    p_contested: np.ndarray         # shape (N,), float32 in [0, 1]: probability of a 50-50 contested duel
    restart_team: np.ndarray        # shape (N,), int8: 0 for H, 1 for A, -1 for None
    confidence: np.ndarray          # shape (N,), float32 in [0, 1]: episode certainty
    is_deflection: Optional[np.ndarray] = None  # shape (N,), bool: True on V-bounce deflection/header
    # Filled only with dead_ball_anchors=True:
    line_crossings: Optional[np.ndarray] = None            # frames where the ball crosses a pitch line
    set_piece_restarts: Optional[Dict[int, Optional[int]]] = None  # Smart Tagging clip start -> restart frame


def compute_rolling_stats(
    arr: np.ndarray,
    window: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes rolling mean using uniform 1D convolution over a 1.0-second window (25 frames)."""
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    mean_roll = np.convolve(padded, kernel, mode="same")[pad:pad + len(arr)]
    return mean_roll.astype(np.float32)


# ==============================================================================
# DEAD-BALL ANCHOR DETECTORS (improvement #4, opt-in)
# ==============================================================================
def _on_pitch(x: float, y: float, pitch: Tuple[float, float], margin: float = 0.0) -> bool:
    """True when (x, y) is inside the pitch shrunk by `margin` metres (negative grows it)."""
    length, width = pitch
    return margin <= x <= length - margin and margin <= y <= width - margin


def _depth_outside(x: float, y: float, pitch: Tuple[float, float]) -> float:
    """How far (metres) a point is beyond the nearest pitch line; <= 0 means on the pitch."""
    length, width = pitch
    return max(-x, x - length, -y, y - width)


def _frames_to_line(x: float, y: float, vx: float, vy: float, pitch: Tuple[float, float]) -> Optional[float]:
    """Frames until a ball at (x, y) moving at (vx, vy) m/s reaches a pitch line; None if it never does."""
    length, width = pitch
    times = []
    if vx > 0:
        times.append((length - x) / vx)
    elif vx < 0:
        times.append(-x / vx)
    if vy > 0:
        times.append((width - y) / vy)
    elif vy < 0:
        times.append(-y / vy)
    times = [t for t in times if t >= 0]
    if not times:
        return None
    return min(times) * FPS


def detect_line_crossings(
    tracking: TrackingData,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    max_extrapolation: int = CROSSING_MAX_EXTRAPOLATION_FRAMES,
) -> np.ndarray:
    """Frames where the ball crosses a pitch line (IFAB Law 9), sorted.

    Only observed ball positions are used. For each last on-pitch observation `a`:
    - next observation `b` is off the pitch, one frame later: the crossing is `b`;
    - there is a gap before `b`: the crossing is where the ball, moving with its velocity at
      `a`, reaches the line (clipped to `b`); with no velocity, it is the straight line a->b;
    - the ball is lost and comes back on the pitch (or never comes back): the crossing is the
      extrapolated one, if the ball was heading out and reaches the line within
      `max_extrapolation` frames, before it is seen again.
    A crossing counts only if the ball then goes CROSSING_CONFIRM_DEPTH_METERS past the line or
    is lost for longer than a ballistic micro-gap. This removes jitter along the line.
    """
    obs = np.flatnonzero(tracking.ball_observed)
    if len(obs) < 2:
        return np.zeros(0, dtype=np.int64)
    x = tracking.ball_x_metric
    y = tracking.ball_y_metric
    vx = tracking.ball_vx if tracking.ball_vx is not None else np.zeros(len(x), dtype=np.float32)
    vy = tracking.ball_vy if tracking.ball_vy is not None else np.zeros(len(x), dtype=np.float32)
    on_pitch = np.array([_on_pitch(x[i], y[i], pitch) for i in obs])
    crossings = []
    for k in range(len(obs) - 1):
        if not on_pitch[k]:
            continue
        a, b = int(obs[k]), int(obs[k + 1])
        gap = b - a - 1
        if not on_pitch[k + 1]:
            # Seen off the pitch next. Confirm by depth or by a long gap in the off-pitch run.
            j = k + 1
            deepest = _depth_outside(x[b], y[b], pitch)
            longest_gap = gap
            while j + 1 < len(obs) and not on_pitch[j + 1]:
                longest_gap = max(longest_gap, int(obs[j + 1] - obs[j] - 1))
                j += 1
                deepest = max(deepest, _depth_outside(x[obs[j]], y[obs[j]], pitch))
            if deepest < CROSSING_CONFIRM_DEPTH_METERS and longest_gap < CROSSING_CONFIRM_GAP_FRAMES:
                continue
            if gap == 0:
                crossings.append(b)
                continue
            frames = _frames_to_line(x[a], y[a], vx[a], vy[a], pitch)
            if frames is None:
                # No velocity: where the straight line a->b meets the pitch boundary.
                d_in = -_depth_outside(x[a], y[a], pitch)
                d_out = _depth_outside(x[b], y[b], pitch)
                frames = (b - a) * d_in / max(d_in + d_out, 1e-6)
            crossings.append(int(min(b, a + max(1, int(math.ceil(frames))))))
        elif gap >= CROSSING_CONFIRM_GAP_FRAMES:
            # Lost on the pitch and seen on the pitch again later: did it go out in between?
            frames = _frames_to_line(x[a], y[a], vx[a], vy[a], pitch)
            if frames is not None and frames <= max_extrapolation and a + frames < b:
                crossings.append(int(a + max(1, int(math.ceil(frames)))))
    return np.array(sorted(set(crossings)), dtype=np.int64)


def find_kickoff_frame(
    tracking: TrackingData,
    search_start: int,
    search_end: int,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    max_distance: float = KICKOFF_CENTRE_MAX_DISTANCE_METERS,
) -> Optional[int]:
    """Kick-off frame: the ball's closest observed approach to the centre spot in the window.

    The tracker often loses the ball while it rests on the centre spot, and finds it again a
    metre or two away once it moves, so the closest observed approach is the best label-free
    estimate of the strike. None if the ball never comes within `max_distance` metres.
    """
    n = len(tracking.ball_observed)
    lo, hi = max(0, search_start), min(n, search_end)
    if hi <= lo:
        return None
    idx = lo + np.flatnonzero(tracking.ball_observed[lo:hi])
    if len(idx) == 0:
        return None
    cx, cy = pitch[0] / 2.0, pitch[1] / 2.0
    dist = np.hypot(tracking.ball_x_metric[idx] - cx, tracking.ball_y_metric[idx] - cy)
    best = int(np.argmin(dist))
    if dist[best] > max_distance:
        return None
    return int(idx[best])


def find_restart_frame(
    tracking: TrackingData,
    set_piece_type: str,
    search_start: int,
    search_end: int,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
) -> Optional[int]:
    """Restart frame of one Smart Tagging set piece, searched in [search_start, search_end).

    - Kick-off: see find_kickoff_frame.
    - Throw-in (IFAB Law 15): the ball, at rest before (mean speed < PRE_KICK_MAX_BALL_SPEED
      over PRE_KICK_WINDOW_FRAMES, unseen frames count as 0), moves faster than
      THROW_IN_MIN_VELOCITY, is seen inside the pitch within RESTART_ON_PITCH_WITHIN_FRAMES, and
      moves at least THROW_IN_MIN_INWARD_METERS into the pitch in that time.
    - Other kicks (IFAB Laws 13, 14, 16, 17: kicked and clearly moves): the same rest test, a
      speed above KICK_IMPULSE_MIN_VELOCITY, more than KICK_IMPULSE_MIN_DISPLACEMENT travelled in
      KICK_IMPULSE_SUSTAINED_FRAMES, and the ball seen on the pitch (RESTART_ON_PITCH_MARGIN_METERS
      tolerance) within RESTART_ON_PITCH_WITHIN_FRAMES.
    The on-pitch test rejects a ball that a ball boy throws or rolls back off the pitch.
    """
    n = len(tracking.ball_observed)
    if "kick off" in set_piece_type.lower():
        return find_kickoff_frame(tracking, search_start, search_end, pitch)
    obs = tracking.ball_observed
    x, y, v = tracking.ball_x_metric, tracking.ball_y_metric, tracking.ball_velocity
    is_throw = "throw" in set_piece_type.lower()
    min_speed = THROW_IN_MIN_VELOCITY if is_throw else KICK_IMPULSE_MIN_VELOCITY
    max_depth = 0.0 if is_throw else RESTART_ON_PITCH_MARGIN_METERS
    for f in range(max(1, search_start), min(n - RESTART_ON_PITCH_WITHIN_FRAMES - 1, search_end)):
        if not obs[f] or v[f] <= min_speed:
            continue
        if np.mean(v[max(0, f - PRE_KICK_WINDOW_FRAMES):f]) >= PRE_KICK_MAX_BALL_SPEED:
            continue
        if not is_throw:
            post = f + KICK_IMPULSE_SUSTAINED_FRAMES
            if not (obs[post] and math.hypot(x[post] - x[f], y[post] - y[f]) > KICK_IMPULSE_MIN_DISPLACEMENT):
                continue
        soon = np.arange(f, f + RESTART_ON_PITCH_WITHIN_FRAMES + 1)
        soon = soon[obs[soon]]
        if not any(_depth_outside(x[k], y[k], pitch) < max_depth for k in soon):
            continue
        if is_throw and _depth_outside(x[soon[-1]], y[soon[-1]], pitch) > _depth_outside(x[f], y[f], pitch) - THROW_IN_MIN_INWARD_METERS:
            continue  # a throw-in moves into the pitch, not along or towards the line
        return f
    return None


def derive_second_half_kickoff(
    tracking: TrackingData,
    smart_tags: Optional[Dict[str, List[Dict]]],
    second_half_start: int,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
) -> Optional[int]:
    """Second-half kick-off frame, from the first Smart Tagging 'Kick Off (Start)' clip that
    starts at or after `second_half_start` (from games.half_video_blocks), refined on the
    tracking with find_kickoff_frame. None if either step finds nothing."""
    if not smart_tags:
        return None
    for sp in smart_tags.get("set_pieces", []):
        if "start" in sp.get("type", "").lower() and "kick off" in sp.get("type", "").lower() and sp["start"] >= second_half_start:
            return find_kickoff_frame(tracking, sp["start"], sp["start"] + SET_PIECE_SEARCH_WINDOW_FRAMES, pitch)
    return None


# ==============================================================================
# EPISODE EVIDENCE ENGINE
# ==============================================================================
def compute_episode_evidence(
    tracking: TrackingData,
    smart_tags: Optional[Dict[str, List[Dict]]] = None,
    n_frames: Optional[int] = None,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    stoppage_enter_p75: float = PLAYER_WALKING_SPEED_CEILING,
    stoppage_exit_p75: float = PLAYER_JOGGING_SPEED_FLOOR,
    stoppage_ball_disp: float = BALL_STATIONARY_DISPLACEMENT_METERS,
    restart_impulse_vel: float = DRIVEN_RESTART_MIN_VELOCITY,
    fast_transit_vel: float = FAST_TRANSIT_VELOCITY_FLOOR,
    enable_smart_tag_restarts: bool = True,
    enable_kinematic_stoppage: bool = True,
    enable_fast_transit: bool = True,
    dead_ball_anchors: bool = False,
) -> EpisodeEvidence:
    """Estimates continuous game episode evidence across all frames with hysteresis.

    dead_ball_anchors (opt-in, improvement #4): D starts at the line crossing for throw-ins,
    corners and goal kicks, restarts are found with the stricter find_restart_frame, kick-offs
    after a goal end the goal lock at the detected kick, and a kinematic dead state that carries
    on after a restart is cleared. Off by default, so the submitted run is unchanged.
    """
    if n_frames is None:
        n_frames = len(tracking.ball_observed) if hasattr(tracking, "ball_observed") and tracking.ball_observed is not None else TOTAL_FRAMES
    pitch_length, pitch_width = pitch

    p_dead = np.full(n_frames, PROB_DEAD_LIVE_BASELINE, dtype=np.float32)
    p_fast_transit = np.zeros(n_frames, dtype=np.float32)
    p_restart_candidate = np.zeros(n_frames, dtype=np.float32)
    p_contested = np.zeros(n_frames, dtype=np.float32)
    restart_team = np.full(n_frames, -1, dtype=np.int8)
    confidence = np.full(n_frames, DEFAULT_EPISODE_CONFIDENCE, dtype=np.float32)

    # --------------------------------------------------------------------------
    # 1. KINEMATIC STOPPAGE DETECTION WITH HYSTERESIS & 90-MINUTE SAFEGUARDS
    # --------------------------------------------------------------------------
    if enable_kinematic_stoppage:
        player_speed_p75 = tracking.player_p75_speed if tracking.player_p75_speed is not None else tracking.player_mean_speed
        player_active_fraction = tracking.player_active_frac if tracking.player_active_frac is not None else np.full(n_frames, DEFAULT_ACTIVE_FRACTION_FALLBACK, dtype=np.float32)
        ball_velocity = tracking.ball_velocity
        ball_x_metric = tracking.ball_x_metric
        ball_y_metric = tracking.ball_y_metric
        ball_observed = tracking.ball_observed

        # Rolling 1-second (25 frame) metrics for net ball displacement
        displacement_window_frames = BALL_DISPLACEMENT_WINDOW_FRAMES
        half_window_frames = displacement_window_frames // 2

        ball_displacement = np.zeros(n_frames, dtype=np.float32)
        valid_ball_frames = np.flatnonzero(ball_observed)
        if len(valid_ball_frames) > 0:
            for center_frame_idx in range(half_window_frames, n_frames - half_window_frames, BALL_DISPLACEMENT_STRIDE_FRAMES):
                frame_start = center_frame_idx - half_window_frames
                frame_end = center_frame_idx + half_window_frames
                if ball_observed[frame_start] and ball_observed[frame_end]:
                    delta_x_window = ball_x_metric[frame_end] - ball_x_metric[frame_start]
                    delta_y_window = ball_y_metric[frame_end] - ball_y_metric[frame_start]
                    ball_displacement[frame_start:frame_end + 1] = math.sqrt(delta_x_window * delta_x_window + delta_y_window * delta_y_window)

        # Pre-compute XML tactical build-up / progression intervals
        is_buildup = np.zeros(n_frames, dtype=bool)
        if smart_tags and "active_phases" in smart_tags:
            for phase in smart_tags["active_phases"]:
                phase_code = phase.get("code", "")
                if phase_code in ("BUILD UP", "PROGRESSION"):
                    phase_start = max(0, phase.get("start", 0))
                    phase_end = min(n_frames - 1, phase.get("end", 0))
                    is_buildup[phase_start:phase_end + 1] = True

        # Precompute set-piece preparatory windows from XML annotations
        is_set_piece_prep = np.zeros(n_frames, dtype=bool)
        if smart_tags:
            for set_piece in smart_tags.get("set_pieces", []):
                set_piece_start = max(0, set_piece["start"] - SET_PIECE_WHISTLE_LOOKBACK_FRAMES)
                set_piece_end = min(n_frames - 1, set_piece["start"] + SET_PIECE_SEARCH_WINDOW_FRAMES)
                is_set_piece_prep[set_piece_start:set_piece_end] = True

        # Precompute recent out-of-bounds flag (within 150 frames = 6.0s)
        recent_oob_countdown = 0
        gk_hold_counter = 0

        # Hysteresis state machine counters
        in_dead_state = False
        stoppage_enter_counter = 0
        stoppage_exit_counter = 0

        for frame_idx in range(n_frames):
            # Decrement OOB memory countdown on every frame (expires in 150 frames = 6.0s)
            if recent_oob_countdown > 0:
                recent_oob_countdown -= 1

            # Out-of-Bounds Physical Gate (IFAB Law 9: Ball out of play across touchline/goal line)
            is_oob = False
            if ball_observed[frame_idx]:
                ball_x_curr = ball_x_metric[frame_idx]
                ball_y_curr = ball_y_metric[frame_idx]
                if (
                    ball_x_curr < -PITCH_BOUNDARY_BUFFER_METERS
                    or ball_x_curr > pitch_length + PITCH_BOUNDARY_BUFFER_METERS
                    or ball_y_curr < -PITCH_BOUNDARY_BUFFER_METERS
                    or ball_y_curr > pitch_width + PITCH_BOUNDARY_BUFFER_METERS
                ):
                    is_oob = True
                    recent_oob_countdown = OOB_MEMORY_FRAMES

            player_speed_p75_curr = player_speed_p75[frame_idx]
            ball_velocity_curr = ball_velocity[frame_idx]
            ball_displacement_curr = ball_displacement[frame_idx]
            player_active_fraction_curr = player_active_fraction[frame_idx]

            # ------------------------------------------------------------------
            # Safeguard B: Goalkeeper catch-and-hold inside 16.5m penalty box
            # When a GK catches or dives on a cross/shot in live play, he holds the ball
            # for 4-8s while outfield players reset.
            # Exceptions & IFAB Law 12 Refinements:
            # 1. Goal Kick / Set-Piece Prep: If an XML restart is being prepared or the
            #    ball went out of bounds recently, keeper contact is restart placement (DEAD).
            # 2. IFAB 8-Second Rule: Holding for > 200 frames (8.0s) inside box ceases
            #    to protect against dead ball detection.
            # ------------------------------------------------------------------
            is_gk_holding = False
            is_restart_context = is_set_piece_prep[frame_idx] or (recent_oob_countdown > 0)

            if ball_observed[frame_idx] and not is_oob and not is_restart_context:
                ball_x_curr = ball_x_metric[frame_idx]
                ball_y_curr = ball_y_metric[frame_idx]
                in_penalty_box = (
                    (ball_x_curr <= PENALTY_BOX_DEPTH_METERS or ball_x_curr >= pitch_length - PENALTY_BOX_DEPTH_METERS)
                    and (PENALTY_BOX_WIDTH_MARGIN_METERS <= ball_y_curr <= pitch_width - PENALTY_BOX_WIDTH_MARGIN_METERS)
                )
                if in_penalty_box:
                    dist_home_gk = tracking.d_H_gk[frame_idx] if tracking.d_H_gk is not None else np.nan
                    dist_away_gk = tracking.d_A_gk[frame_idx] if tracking.d_A_gk is not None else np.nan
                    if (not np.isnan(dist_home_gk) and dist_home_gk < GK_BALL_PROXIMITY_RADIUS_METERS) or (
                        not np.isnan(dist_away_gk) and dist_away_gk < GK_BALL_PROXIMITY_RADIUS_METERS
                    ):
                        gk_hold_counter += 1
                        if gk_hold_counter <= GK_HOLD_MAX_FRAMES:  # Within legal 8.0s limit
                            is_gk_holding = True
                    else:
                        gk_hold_counter = 0
                else:
                    gk_hold_counter = 0
            else:
                gk_hold_counter = 0

            # ------------------------------------------------------------------
            # Safeguard C: "La Pausa" (Center-back standing on ball during build-up)
            # During tactical build-up against a mid-block, center-backs pause with
            # foot on the ball. Outfield speed drops, but this is tactical patience, not a foul.
            # ------------------------------------------------------------------
            is_la_pausa = False
            if ball_observed[frame_idx] and not is_oob:
                if is_buildup[frame_idx]:
                    is_la_pausa = True

            # Entry condition: players walking (< 1.5 m/s), low ball displacement (< 3m), runners < 15%
            # Protected by Safeguards B & C against misclassifying GK holds and La Pausa
            is_stoppage_candidate = (
                is_oob or
                (
                    not is_gk_holding
                    and not is_la_pausa
                    and player_speed_p75_curr < stoppage_enter_p75
                    and (ball_displacement_curr < stoppage_ball_disp or not ball_observed[frame_idx])
                    and player_active_fraction_curr < PLAYER_ACTIVE_FRACTION_CEILING
                )
            )

            # ------------------------------------------------------------------
            # Safeguard D: IFAB Restart Legality & Ball Movement Law
            # IFAB Laws 8, 13, 14, 16, 17: A restart becomes live when "kicked and clearly moves".
            # Accommodates short goal kicks (e.g. 2m pass to center-back at 3.0 m/s)
            # and long kicks, while preventing substitution/water break jogs from
            # waking a dead ball when the ball remains stationary (ball_disp_t < 0.5m).
            # ------------------------------------------------------------------
            ball_moved = (
                (ball_velocity_curr > SHORT_RESTART_MIN_VELOCITY and ball_displacement_curr > SHORT_RESTART_MIN_DISPLACEMENT) or
                (ball_velocity_curr > restart_impulse_vel and ball_displacement_curr > DRIVEN_RESTART_MIN_DISPLACEMENT) or
                (
                    ball_displacement_curr > RESTART_BREAKOUT_BALL_DISPLACEMENT_METERS
                    and player_speed_p75_curr > stoppage_exit_p75
                    and player_active_fraction_curr > RESTART_BREAKOUT_ACTIVE_FRACTION_FLOOR
                )
            )
            is_exit_candidate = (
                not is_oob and
                ball_observed[frame_idx] and
                ball_moved
            )

            if not in_dead_state:
                if is_stoppage_candidate:
                    stoppage_enter_counter += 1
                    # Sustained slowdown before entering stoppage (20 frames = 0.80s)
                    if stoppage_enter_counter >= STOPPAGE_ENTER_SUSTAINED_FRAMES:
                        in_dead_state = True
                        stoppage_exit_counter = 0
                else:
                    stoppage_enter_counter = max(0, stoppage_enter_counter - 1)
            else:
                if is_exit_candidate:
                    stoppage_exit_counter += 1
                    # Sustained active restart before exiting stoppage (12 frames = 0.48s)
                    if stoppage_exit_counter >= STOPPAGE_EXIT_SUSTAINED_FRAMES:
                        in_dead_state = False
                        stoppage_enter_counter = 0
                else:
                    stoppage_exit_counter = max(0, stoppage_exit_counter - 1)

            if in_dead_state:
                p_dead[frame_idx] = PROB_DEAD_CONFIRMED
            elif is_stoppage_candidate:
                p_dead[frame_idx] = PROB_DEAD_TRANSITION_BUFFER

    # Dead-ball anchors: remember the kinematic dead state before the Smart Tagging overrides.
    kinematic_dead = (p_dead >= PROB_DEAD_CONFIRMED) if dead_ball_anchors else None
    line_crossings = detect_line_crossings(tracking, pitch) if dead_ball_anchors else None
    set_piece_restarts: Optional[Dict[int, Optional[int]]] = {} if dead_ball_anchors else None
    anchored_locks: List[Tuple[int, int]] = []  # [start, stop) frames locked dead by an anchor

    # --------------------------------------------------------------------------
    # 2. SET-PIECE & GOAL RESTART BACKTRACKING ENGINE
    # --------------------------------------------------------------------------
    if enable_smart_tag_restarts and smart_tags:
        ball_velocity = tracking.ball_velocity
        ball_observed = tracking.ball_observed
        ball_x_metric = tracking.ball_x_metric
        ball_y_metric = tracking.ball_y_metric
        player_speed_p75 = tracking.player_p75_speed if tracking.player_p75_speed is not None else tracking.player_mean_speed

        # A) GOALS Dead-Ball Celebration Locking:
        # A goal celebration begins when the ball crosses the goal line and ends on the subsequent kickoff.
        kickoffs_after_goal = [sp for sp in smart_tags.get("set_pieces", []) if "After Goal" in sp.get("type", "")]
        kickoffs_after_goal.sort(key=lambda x: x["start"])
        for goal_event in smart_tags.get("goals", []):
            goal_start_frame = goal_event["start"]
            goal_end_frame = goal_event["end"]
            if goal_start_frame >= n_frames:
                continue
            matching_kickoff = None
            for ko in kickoffs_after_goal:
                if ko["start"] >= goal_start_frame - GOAL_KICKOFF_PRE_WINDOW_FRAMES and ko["start"] <= goal_end_frame + GOAL_KICKOFF_POST_WINDOW_FRAMES:
                    matching_kickoff = ko
                    break
            # Find the exact moment the ball crosses the goal line (x < 2.5m or x > length - 2.5m)
            goal_crossing_frame = min(n_frames - 1, goal_start_frame + int(GOAL_CROSSING_FALLBACK_RATIO * (goal_end_frame - goal_start_frame)))
            for candidate_frame in range(goal_start_frame, min(n_frames - 1, goal_start_frame + GOAL_CROSSING_SEARCH_FRAMES)):
                if ball_observed[candidate_frame] and (
                    ball_x_metric[candidate_frame] < GOAL_LINE_CROSSING_MARGIN_METERS
                    or ball_x_metric[candidate_frame] > pitch_length - GOAL_LINE_CROSSING_MARGIN_METERS
                ):
                    goal_crossing_frame = candidate_frame
                    break
            # Kickoff occurs ~50 frames (2.0s) into the kickoff clip
            kickoff_frame = min(n_frames, matching_kickoff["start"] + KICKOFF_CLIP_OFFSET_FRAMES) if matching_kickoff else min(n_frames, goal_end_frame)
            detected_kickoff = None
            if dead_ball_anchors and matching_kickoff:
                detected_kickoff = find_kickoff_frame(
                    tracking, matching_kickoff["start"], matching_kickoff["start"] + SET_PIECE_SEARCH_WINDOW_FRAMES, pitch
                )
                set_piece_restarts[matching_kickoff["start"]] = detected_kickoff
                if detected_kickoff is not None:
                    kickoff_frame = detected_kickoff
            p_dead[goal_crossing_frame:kickoff_frame] = PROB_DEAD_GOAL_LOCKED
            if detected_kickoff is not None:
                anchored_locks.append((goal_crossing_frame, detected_kickoff))
                p_restart_candidate[detected_kickoff] = 1.0
                p_dead[detected_kickoff:min(n_frames, detected_kickoff + RESTART_LIVE_BURST_FRAMES)] = PROB_DEAD_RESTART_BURST

        # B) SET PIECES: Corners, Goal Kicks, Free Kicks, Throw-ins
        for sp in smart_tags.get("set_pieces", []):
            sp_type = sp.get("type", "")
            if "After Goal" in sp_type:
                continue
            clip_start = sp["start"]
            clip_end = sp["end"]
            if clip_start >= n_frames:
                continue
            team_label = sp.get("team", "")
            target_team = -1
            if "Barcelona" in team_label:
                target_team = HOME_TEAM_CODE
            elif any(k in team_label.lower() for k in ["athletic", "atletic"]):
                target_team = AWAY_TEAM_CODE

            # Whistle / stoppage occurred prior to XML clip start (150 frames = 6.0s buffer)
            whistle_frame = max(0, clip_start - SET_PIECE_WHISTLE_LOOKBACK_FRAMES)

            # Search strictly in the initial 14-second window [clip_start, clip_start + 350]
            # (Avoids locking onto downstream open-play passes in trailing 21 seconds of the 35s clip)
            restart_frame = None
            search_start = min(n_frames - 1, clip_start)
            search_end = min(n_frames - 1, clip_start + SET_PIECE_SEARCH_WINDOW_FRAMES)

            for candidate_frame in (range(0) if dead_ball_anchors else range(search_start, search_end - 5)):
                prev_speed_window = ball_velocity[max(0, candidate_frame - PRE_KICK_WINDOW_FRAMES):candidate_frame]
                # Pre-kick state must be stationary (mean speed < 2.0 m/s)
                if len(prev_speed_window) > 0 and np.mean(prev_speed_window) < PRE_KICK_MAX_BALL_SPEED:
                    # Kicks (IFAB Laws 8, 13, 14, 16, 17: kicked and clearly moves)
                    # Detectable kick impulse > 2.5 m/s with sustained displacement >= 1.0m over 10 frames (0.4s)
                    if ball_velocity[candidate_frame] > KICK_IMPULSE_MIN_VELOCITY:
                        post_kick_frame = min(n_frames - 1, candidate_frame + KICK_IMPULSE_SUSTAINED_FRAMES)
                        if ball_observed[candidate_frame] and ball_observed[post_kick_frame]:
                            delta_x_kick = ball_x_metric[post_kick_frame] - ball_x_metric[candidate_frame]
                            delta_y_kick = ball_y_metric[post_kick_frame] - ball_y_metric[candidate_frame]
                            displacement_kick = math.sqrt(delta_x_kick * delta_x_kick + delta_y_kick * delta_y_kick)
                            if displacement_kick > KICK_IMPULSE_MIN_DISPLACEMENT:
                                restart_frame = candidate_frame
                                break
                    # Throw-ins (IFAB Law 15: ball becomes live upon entering the field of play)
                    elif "Throw" in sp_type and ball_observed[candidate_frame]:
                        if (
                            THROW_IN_FIELD_MARGIN_METERS <= ball_x_metric[candidate_frame] <= pitch_length - THROW_IN_FIELD_MARGIN_METERS
                            and THROW_IN_FIELD_MARGIN_METERS <= ball_y_metric[candidate_frame] <= pitch_width - THROW_IN_FIELD_MARGIN_METERS
                        ):
                            if ball_velocity[candidate_frame] > THROW_IN_MIN_VELOCITY:
                                restart_frame = candidate_frame
                                break

            if dead_ball_anchors:
                anchored_on_crossing = False
                if any(t in sp_type for t in OUT_OF_PLAY_RESTART_TYPES):
                    # D starts when the ball crossed the line, not a fixed 6 s before the clip,
                    # and the restart is searched only after that crossing.
                    causes = line_crossings[
                        (line_crossings >= clip_start - CROSSING_LOOKBACK_FRAMES)
                        & (line_crossings <= clip_start + CROSSING_AFTER_CLIP_START_FRAMES)
                    ]
                    if len(causes):
                        whistle_frame = int(causes[-1])
                        search_start = max(search_start, whistle_frame + 1)
                        anchored_on_crossing = True
                restart_frame = find_restart_frame(tracking, sp_type, search_start, search_end, pitch)
                set_piece_restarts[clip_start] = restart_frame
                if not anchored_on_crossing:
                    # No crossing: do not lock frames of clear live play before the stoppage.
                    lock_end = restart_frame if restart_frame is not None else min(n_frames, clip_start)
                    window = np.arange(whistle_frame, max(whistle_frame, lock_end))
                    if len(window):
                        ball_fast_on_pitch = (
                            ball_observed[window]
                            & (ball_velocity[window] > FAST_TRANSIT_VELOCITY_FLOOR)
                            & (ball_x_metric[window] >= 0) & (ball_x_metric[window] <= pitch_length)
                            & (ball_y_metric[window] >= 0) & (ball_y_metric[window] <= pitch_width)
                        )
                        players_running = player_speed_p75[window] > PLAYER_JOGGING_SPEED_FLOOR
                        live_frames = window[ball_fast_on_pitch | players_running]
                        if len(live_frames):
                            whistle_frame = int(live_frames[-1]) + 1
                if restart_frame is not None:
                    anchored_locks.append((whistle_frame, restart_frame))

            # If an impulse frame was found:
            if restart_frame is not None:
                p_dead[whistle_frame:restart_frame] = np.maximum(p_dead[whistle_frame:restart_frame], PROB_DEAD_SET_PIECE_LOCKED)
                p_restart_candidate[restart_frame] = 1.0
                if target_team != -1:
                    restart_team[restart_frame] = target_team
                p_dead[restart_frame:min(n_frames, restart_frame + RESTART_LIVE_BURST_FRAMES)] = PROB_DEAD_RESTART_BURST
            else:
                prep_end = min(n_frames, clip_start + int(FALLBACK_SET_PIECE_PREP_RATIO * (clip_end - clip_start)))
                if dead_ball_anchors:
                    # No restart found: do not guess its time; lock only up to the clip start.
                    prep_end = min(n_frames, clip_start)
                p_dead[whistle_frame:prep_end] = np.maximum(p_dead[whistle_frame:prep_end], PROB_DEAD_SET_PIECE_FALLBACK)

        # C) Out-of-bounds Set-Piece Routine Bridging:
        # Goal kicks and corner kicks originate from out-of-bounds balls.
        # Bridge the preparation interval from out-of-bounds to the restart clip.
        for sp in smart_tags.get("set_pieces", []):
            if sp.get("type") in ("Goal Kick", "Corner Kick"):
                set_piece_start_frame = sp["start"]
                if set_piece_start_frame >= n_frames:
                    continue
                if dead_ball_anchors and set_piece_restarts.get(set_piece_start_frame) is not None:
                    continue  # the crossing-to-restart lock above already covers this stoppage
                # Look back up to 375 frames (15.0s) for the preceding out-of-bounds frame
                for candidate_start in range(max(0, set_piece_start_frame - OOB_RETRIEVAL_LOOKBACK_FRAMES), min(n_frames, set_piece_start_frame)):
                    if candidate_start < n_frames and p_dead[candidate_start] > PROB_DEAD_HYSTERESIS_ACTIVE_FLOOR:
                        forward_limit = min(n_frames, set_piece_start_frame + OOB_ROUTINE_FORWARD_FRAMES)
                        p_dead[candidate_start:forward_limit] = np.maximum(p_dead[candidate_start:forward_limit], PROB_DEAD_CONFIRMED)
                        break

    # Retroactive Dead-Ball Bridge Healing:
    # Eliminates brief false live breaks inside stoppages when players are walking
    player_speed_p75 = tracking.player_p75_speed if tracking.player_p75_speed is not None else tracking.player_mean_speed
    is_dead_ball_active = (p_dead > PROB_DEAD_HYSTERESIS_ACTIVE_FLOOR).astype(np.int8)
    dead_state_transitions = np.diff(is_dead_ball_active, prepend=0, append=0)
    dead_gap_starts = np.flatnonzero(dead_state_transitions == 1)
    dead_gap_ends = np.flatnonzero(dead_state_transitions == -1)

    for gap_idx in range(len(dead_gap_ends) - 1):
        gap_start_frame = dead_gap_ends[gap_idx]
        gap_end_frame = dead_gap_starts[gap_idx + 1]
        gap_length_frames = gap_end_frame - gap_start_frame
        if 0 < gap_length_frames <= BRIDGE_HEALING_MAX_GAP_FRAMES:
            if dead_ball_anchors and np.any(p_restart_candidate[gap_start_frame:gap_end_frame] > 0.5):
                continue  # a restart inside the gap is real live play
            if np.mean(player_speed_p75[gap_start_frame:gap_end_frame]) < stoppage_enter_p75:
                p_dead[gap_start_frame:gap_end_frame] = PROB_DEAD_CONFIRMED

    # Dead-ball anchors: end the stoppage at the restart. A kinematic dead state that carries on
    # after the restart (players still walking) is cleared, up to POST_RESTART_CLEAR_MAX_FRAMES,
    # the next line crossing, or the next anchored stoppage, whichever comes first. Frames where
    # the ball is seen off the pitch stay dead.
    if dead_ball_anchors and set_piece_restarts:
        locked = np.zeros(n_frames, dtype=bool)
        for lock_start, lock_stop in anchored_locks:
            locked[max(0, lock_start):max(0, lock_stop)] = True
        for restart in sorted(r for r in set_piece_restarts.values() if r is not None):
            begin = restart + RESTART_LIVE_BURST_FRAMES
            if begin >= n_frames or not kinematic_dead[min(n_frames - 1, begin)]:
                continue
            stop = min(n_frames, restart + POST_RESTART_CLEAR_MAX_FRAMES)
            later = line_crossings[line_crossings > restart]
            if len(later):
                stop = min(stop, int(later[0]))
            frame_idx = begin
            while frame_idx < stop and kinematic_dead[frame_idx] and not locked[frame_idx]:
                seen_off_pitch = tracking.ball_observed[frame_idx] and not _on_pitch(
                    tracking.ball_x_metric[frame_idx], tracking.ball_y_metric[frame_idx], pitch, -PITCH_BOUNDARY_BUFFER_METERS
                )
                if not seen_off_pitch:
                    p_dead[frame_idx] = min(p_dead[frame_idx], PROB_DEAD_LIVE_BASELINE)
                frame_idx += 1

    # --------------------------------------------------------------------------
    # 3. FAST BALL TRANSIT DETECTION
    # --------------------------------------------------------------------------
    if enable_fast_transit:
        ball_velocity = tracking.ball_velocity
        for frame_idx in range(n_frames):
            if ball_velocity[frame_idx] > fast_transit_vel:
                # Scale transit probability between fast_transit_vel (6.5 m/s) and driven pass ceiling (10.0 m/s)
                velocity_excess = ball_velocity[frame_idx] - fast_transit_vel
                prob_transit = min(1.0, velocity_excess / FAST_TRANSIT_RAMP_SPAN + FAST_TRANSIT_BASE_PROBABILITY)
                p_fast_transit[frame_idx] = float(prob_transit)

    # --------------------------------------------------------------------------
    # 4. CONTESTED 50-50 CHALLENGE DETECTION
    # --------------------------------------------------------------------------
    dist_home_nearest = tracking.d_H
    dist_away_nearest = tracking.d_A
    for frame_idx in range(n_frames):
        if not np.isnan(dist_home_nearest[frame_idx]) and not np.isnan(dist_away_nearest[frame_idx]):
            # Both players within physical tackle reach
            if dist_home_nearest[frame_idx] < DUEL_PHYSICAL_REACH_RADIUS and dist_away_nearest[frame_idx] < DUEL_PHYSICAL_REACH_RADIUS:
                duel_proximity_difference = abs(dist_home_nearest[frame_idx] - dist_away_nearest[frame_idx])
                if duel_proximity_difference < DUEL_EQUIDISTANCE_TOLERANCE:
                    p_contested[frame_idx] = float(1.0 - duel_proximity_difference / DUEL_EQUIDISTANCE_TOLERANCE)

    # Compute p_live as exact mathematical complement of p_dead
    p_live = np.clip(1.0 - p_dead, 0.0, 1.0).astype(np.float32)

    # Compute V-bounce deflection events: sharp deflection (> 35 deg) with high residual speed (> 4.0 m/s)
    is_deflection = np.zeros(n_frames, dtype=bool)
    if hasattr(tracking, "ball_deflection_angle") and tracking.ball_deflection_angle is not None:
        is_deflection = (
            (tracking.ball_deflection_angle > DEFLECTION_MIN_ANGLE_DEGREES)
            & (tracking.ball_velocity > DEFLECTION_MIN_RESIDUAL_VELOCITY)
        )

    return EpisodeEvidence(
        p_dead=p_dead,
        p_live=p_live,
        p_fast_transit=p_fast_transit,
        p_restart_candidate=p_restart_candidate,
        p_contested=p_contested,
        restart_team=restart_team,
        confidence=confidence,
        is_deflection=is_deflection,
        line_crossings=line_crossings,
        set_piece_restarts=set_piece_restarts,
    )


def export_episode_audit(
    evidence: EpisodeEvidence,
    tracking: TrackingData,
    output_path: str,
    subsample: int = 25,
) -> None:
    """Exports a diagnostic CSV for inspecting episode classifications across the match."""
    import pandas as pd

    frames = np.arange(0, len(evidence.p_dead), subsample)
    rows = []
    p75 = tracking.player_p75_speed if tracking.player_p75_speed is not None else tracking.player_mean_speed

    for sample_frame in frames:
        elapsed_seconds = sample_frame / FPS
        minutes, seconds = divmod(elapsed_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        timestamp_str = f"{int(hours):02d}:{int(minutes):02d}:{seconds:04.1f}"

        rows.append({
            "frame": sample_frame,
            "timestamp": timestamp_str,
            "p_live": round(float(evidence.p_live[sample_frame]), 3),
            "p_dead": round(float(evidence.p_dead[sample_frame]), 3),
            "p_fast_transit": round(float(evidence.p_fast_transit[sample_frame]), 3),
            "p_restart": round(float(evidence.p_restart_candidate[sample_frame]), 3),
            "p_contested": round(float(evidence.p_contested[sample_frame]), 3),
            "ball_velocity": round(float(tracking.ball_velocity[sample_frame]), 2),
            "player_p75_speed": round(float(p75[sample_frame]), 2),
            "ball_observed": int(tracking.ball_observed[sample_frame]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  ✓ Exported episode audit report: {output_path} ({len(df)} sampled rows)")
