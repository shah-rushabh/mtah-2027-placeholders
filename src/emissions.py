"""Stage 4: Conditioned Spatial Likelihoods & Stoppage Prior Engine.

Computes the log-emission matrix ln B(t, k) for frame t in [0..N-1] and state k in {H, A, D}
(or 5-state augmented space {H_ctrl, H_unctrl, A_ctrl, A_unctrl, D}).

Core Physical and Domain Invariants:
1. In-pitch active live play enforces a finite -40.0 nat barrier against Dead Ball,
   preventing open-play stoppage flickering without numerical instability.
2. Certified stoppages (out-of-bounds, stationary set-pieces, goal celebrations, walking players)
   emit a unified +5.0 / -10.0 nat consensus evidence favoring Dead Ball (15.0 nat margin).
3. Missing ball during active wing play emits neutral flat likelihood (ln B_H = ln B_A = 0.0,
   ln B_D = -1000.0), allowing HMM sequence inertia to coast possession cleanly across camera pans.
4. Goalkeepers are given full handling jurisdiction inside the 18-yard penalty box (IFAB Law 12)
   and a 12.0m active play radius outside the box (sweeper-keepers Ter Stegen and Padilla).
5. Open-space loose ball dampening (d_min > 2.8m) suppresses distance pulls down to 5% baseline.
6. Undeflected high-speed flight (v > 6.5 m/s, deflection < 25 deg) suppresses 2D ground proximity
   by 90%, preventing mid-air fly-bys from flipping possession to passive opponents.
7. Deflection events distribute a bounded budget (3.5 nats) across a strictly symmetric 5-frame kernel,
   capped at 1.0 nat for passive ricochets to prevent accidental turnovers.
8. Set-piece restart kicks apply an immediate +8.0 / -15.0 nat impulse on the strike frame,
   snapping the state out of Dead Ball into active live play with zero lag.

All constants are derived from IFAB Laws of the Game, certified pitch dimensions (105m x 68m),
human biomechanical movement ceilings, and label-free spatial possession dynamics.
"""

from typing import Dict, Optional

import numpy as np

from .episodes import EpisodeEvidence
from .loader import TrackingData
from .control import ControlEvidence

# =============================================================================
# 1. HMM STATE IDENTIFIERS & NUMERICAL BARRIERS
# =============================================================================
STATE_H = 0  # Home Possession (FC Barcelona)
STATE_A = 1  # Away Possession (Athletic Club)
STATE_D = 2  # Dead Ball / Stoppage (Out-of-bounds, foul, set-piece preparation)

# Strict impenetrable barrier for forbidden states in 32-bit float space:
# exp(-1000.0) evaluates to 0.0 without causing underflow/NaN exceptions.
NEG_INF = -1000.0

# =============================================================================
# 2. SPATIAL PROXIMITY & SOFTMIN AGGREGATION CONSTANTS
# =============================================================================
# Spatial decay scale: sigma = 4.0 meters.
# In exponential pitch control models (Spearman 2018), physical control decays as exp(-d / sigma).
# A jogging player (2.0 m/s = 7.2 km/h = 8:20 min/km) accelerating at 3-4 m/s^2 closes 4.0m in < 0.75s.
SIGMA_PROXIMITY_METERS = 4.0

# Softmin temperature parameter: tau = 1.5 meters.
# Matches physical adult athletic lunging reach and stride length (1.5m).
TAU_SOFTMIN_METERS = 1.5

# Large distance assigned to unobserved/NaN players in top-3 softmin aggregation:
FALLBACK_LARGE_DISTANCE_METERS = 100.0

# =============================================================================
# 3. IFAB LAW 1 PENALTY BOX & GOALKEEPER JURISDICTION CONSTANTS
# =============================================================================
# Under IFAB Law 1, the 18-yard penalty box extends 16.5m from each goal line:
# Pitch dimensions: length = 105.0m, width = 68.0m (center y = 34.0m).
# Penalty box width: 44 yards = 40.32m (y in [34.0 - 20.16, 34.0 + 20.16] = [13.84, 54.16]).
PENALTY_BOX_LENGTH_METERS = 16.5
PITCH_LENGTH_METERS = 105.0
AWAY_PENALTY_BOX_X_START = 88.5  # 105.0m - 16.5m = 88.5m (0.3m ball-radius parallax margin)
PENALTY_BOX_Y_MIN = 13.84
PENALTY_BOX_Y_MAX = 54.16

# Sweeper-keeper active play radius outside the penalty box: 12.0 meters.
# At elite keeper sprint speed (6.0 m/s = 21.6 km/h = 2:47 min/km), 12.0m is covered in 2.0s.
SWEEPER_KEEPER_ACTIVE_RADIUS_METERS = 12.0

# Immediate physical contact radius for fallback goalkeeper handling:
KEEPER_IMMEDIATE_CONTACT_RADIUS_METERS = 1.5

# =============================================================================
# 4. OPEN-SPACE LOOSE BALL DAMPENING CONSTANTS (Harmonized with Stage 3)
# =============================================================================
# Maximum physical reach and sliding tackle boundary in open grass:
LOOSE_BALL_DISTANCE_THRESHOLD = 2.8

# Linear ramp transition span from 2.8m to 5.8m:
LOOSE_BALL_RAMP_SPAN = 3.0

# Minimum baseline scale for spatial distance in open grass (95% suppression):
LOOSE_BALL_MIN_SCALE = 0.05

# =============================================================================
# 5. UNDEFLECTED FAST TRANSIT FLIGHT GATE (Harmonized with Stages 2 & 3)
# =============================================================================
# Maximum human foot dribbling velocity ceiling: 6.5 m/s = 23.4 km/h = 2:34 min/km.
FAST_TRANSIT_VELOCITY_THRESHOLD = 6.5

# Maximum directional deflection angle for clean undeflected flight: 25.0 degrees.
FAST_TRANSIT_MAX_DEFLECTION_DEG = 25.0

# Ground proximity retention factor during ballistic flight (90% suppression):
FAST_TRANSIT_PROXIMITY_RETENTION = 0.10

# =============================================================================
# 6. IN-PITCH LIVE PLAY INVARIANTS & STOPPAGE CONSENSUS EMISSIONS
# =============================================================================
# Maximum dead-ball probability for confirmed live open play:
LIVE_PLAY_MAX_DEAD_PROBABILITY = 0.25

# Live play dead penalty:
# Strong penalty against predicting Dead Ball during active running/dribbling in open play:
LIVE_PLAY_DEAD_PENALTY = -40.0

# Minimum dead-ball probability for confirmed stoppage consensus:
STOPPAGE_CONFIRMED_DEAD_PROBABILITY = 0.70

# Confirmed stoppage penalties and bonuses (whistle blown, out-of-bounds, set-piece setup):
# +5.0 bonus for Dead Ball, -10.0 penalty against live play:
CONFIRMED_STOPPAGE_DEAD_BONUS = 5.0
CONFIRMED_STOPPAGE_LIVE_PENALTY = -10.0

# Ambiguous stoppage interpolation band [0.25, 0.70]:
AMBIGUOUS_DEAD_PROB_MIN = 0.25
AMBIGUOUS_DEAD_PROB_MAX = 0.70
UNCONFIRMED_STOPPAGE_PENALTY = -5.0
LIKELY_STOPPAGE_BONUS = 2.0

# =============================================================================
# 7. SYMMETRIC CONTACT KERNEL & CONTROL EVIDENCE BOOST (Harmonized with Stage 3)
# =============================================================================
# Strictly symmetric 5-frame kernel [-2..+2] spanning 0.20s:
CONTACT_KERNEL_PROFILE = np.array([0.10, 0.20, 0.40, 0.20, 0.10], dtype=np.float32)

# Contact evidence touch bonuses:
DEFLECTION_ACTIVE_TOUCH_BONUS = 3.5
DEFLECTION_IMPUTED_TOUCH_BONUS = 1.75
DEFLECTION_PASSIVE_TOUCH_BONUS = 1.0  # Capped for passive ricochets to prevent false turnovers

# Controlled possession bonus for active dribbling/carrying:
CONTROLLED_POSSESSION_BONUS = 2.5

# 50-50 physical duel contest suppression:
CONTESTED_DUEL_PROB_THRESHOLD = 0.40
CONTESTED_DUEL_DAMPING_FACTOR = 0.40

# Fallback deflection parameters (when Stage 3 control_evidence is absent):
FALLBACK_DEFLECTION_MIN_ANGLE = 35.0
FALLBACK_DEFLECTION_MIN_VELOCITY = 4.0  # 14.4 km/h = 4:10 min/km
FALLBACK_DEFLECTION_MAX_DISTANCE = 1.8  # Harmonized with DUEL_PHYSICAL_REACH_RADIUS
FALLBACK_DEFLECTION_ADVANTAGE_MARGIN = 0.5

# =============================================================================
# 8. SET-PIECE RESTART IMPULSE CONSTANTS
# =============================================================================
# Confirmed restart kick probability threshold:
RESTART_CONFIRMED_PROB_THRESHOLD = 0.80

# Strike frame impulse snapping match out of Dead Ball with zero lag:
# +8.0 bonus for kicking team, -15.0 penalty for opponent and dead ball:
RESTART_KICK_TEAM_BOOST = 8.0
RESTART_OPPONENT_AND_DEAD_PENALTY = -15.0

# =============================================================================
# 9. MISSING BALL GAP CATEGORIZATION EMISSIONS
# =============================================================================
# Halftime / pre-match penalty against live play:
HALFTIME_DEAD_BASELINE = 0.0
HALFTIME_LIVE_PENALTY = -50.0

# Unclear missing gap penalties and bonuses:
UNCLEAR_GAP_DEAD_BONUS = 1.0
UNCLEAR_GAP_LIVE_PENALTY = -2.0

# Active missing gap parameters:
GAP_ACTIVE_MAX_DEAD_PROBABILITY = 0.60
GAP_ACTIVE_LIVE_EMISSION = 0.0

# =============================================================================
# 10. SMART TAGGING STRUCTURAL PRIORS & STATICS
# =============================================================================
# Set-piece stationary preparation window (first 350 frames = 14.0s of XML clip):
SET_PIECE_PREP_MAX_FRAMES = 350
SET_PIECE_BALL_STATIONARY_VELOCITY = 1.0  # 3.6 km/h = 16:40 min/km
SET_PIECE_UNCONTESTED_DISTANCE = 2.0

# Goal celebration window constants:
GOAL_CROSSING_SEARCH_FRAMES = 750  # First 30.0s of goal clip
GOAL_CELEBRATION_MAX_FRAMES = 2500  # Up to 100.0s celebration/VAR window
GOAL_LINE_X_HOME = 2.0
GOAL_LINE_X_AWAY = 103.0
GOAL_BUILDUP_LIVE_BONUS = 1.0
GOAL_CELEBRATION_DEAD_BONUS = 10.0

# Capped active event priors:
EVENT_PRIOR_DEFAULT_CAP = 2.5
REFEREE_EVENT_PER_FRAME_BONUS = 0.02

# =============================================================================
# 11. CERTIFIED MATCH PERIOD BOUNDARIES (BAR-ATH_tactical_atd.xml)
# =============================================================================
H1_START, H1_END = 11185, 90651
H2_START, H2_END = 110808, 190197
# Second-half kick-off. This literal is a frame of this match, so it is used ONLY to reproduce
# the submitted file (flags off). With dead_ball_anchors the pipeline passes h2_kickoff, derived
# at run time from the Smart Tagging 'Kick Off (Start)' clip and the ball track
# (episodes.derive_second_half_kickoff).
H2_KICKOFF = 113742

# Dead-ball anchors (improvement #4): a ball lost on the pitch is live until its extrapolated
# line crossing, if that crossing is within this many frames of the start of the gap.
GAP_CROSSING_MAX_LIVE_FRAMES = 50

# =============================================================================
# 12. AUGMENTED 5-STATE BASELINE CONSTANTS
# =============================================================================
AUGMENTED_UNCONTROLLED_BASE_OFFSET = -1.0
AUGMENTED_UNCONTROLLED_LOOSE_BONUS = 1.5
AUGMENTED_UNCONTROLLED_TRANSIT_BONUS = 1.5
AUGMENTED_UNCONTROLLED_CONTEST_BONUS = 0.5


def compute_log_emissions(
    tracking: TrackingData,
    smart_tags: Optional[Dict] = None,
    n_frames: int = 199319,
    sigma: float = SIGMA_PROXIMITY_METERS,
    ownership_strategy: str = "softmin_multi",  # 'softmin_multi' or 'single_nearest'
    tau_softmin: float = TAU_SOFTMIN_METERS,
    event_prior_cap: float = EVENT_PRIOR_DEFAULT_CAP,
    episode_evidence: Optional[EpisodeEvidence] = None,
    control_evidence: Optional[ControlEvidence] = None,
    augmented_5state: bool = False,
    dead_ball_anchors: bool = False,
    h2_kickoff: Optional[int] = None,
) -> np.ndarray:
    """Computes shape (n_frames, 3) log-emission matrix for H, A, D states.

    Args:
        tracking: TrackingData containing player/ball trajectories and metric distances.
        smart_tags: Optional dictionary of Smart Tagging annotations (set-pieces, goals).
        n_frames: Total number of frames in match timeline (199,319).
        sigma: Proximity exponential decay scale in meters (default: 4.0m).
        ownership_strategy: 'single_nearest' or 'softmin_multi' player aggregation.
        tau_softmin: Temperature parameter for softmin distance aggregation (1.5m).
        event_prior_cap: Maximum total log-prior accumulated per event window (2.5 nats).
        episode_evidence: Optional macro kinematic episode evidence from Stage 2.
        control_evidence: Optional micro interaction and turnover evidence from Stage 3.
        augmented_5state: Whether to return 5-state augmented matrix {H_c, H_u, A_c, A_u, D}.
        dead_ball_anchors: Opt-in (improvement #4). A stoppage gap is live until the ball's
            extrapolated line crossing, set-piece preparation stops at the detected restart, and
            a goal window ends at the detected kick-off. Needs episode_evidence built with
            dead_ball_anchors=True.
        h2_kickoff: Second-half kick-off frame derived at run time. None keeps the legacy
            H2_KICKOFF literal (reproduces the submitted file).

    Returns:
        log_B: np.ndarray of shape (n_frames, 3) or (n_frames, 5) containing ln B(t, k).
    """
    log_B = np.zeros((n_frames, 3), dtype=np.float32)

    # =========================================================================
    # STEP 4.1: OUTFIELD SPATIAL PROXIMITY EMISSIONS
    # =========================================================================
    # Exponential pitch control decay: ln B(t, k) = -d_eff(t) / sigma
    has_valid_outfield_distances = ~np.isnan(tracking.d_H) & ~np.isnan(tracking.d_A)

    if ownership_strategy == "softmin_multi" and hasattr(tracking, "d_H_top3") and tracking.d_H_top3 is not None:
        # Softmin multi-player aggregation over top 3 closest outfield players:
        # Normalized softmin: d_eff = -tau * ln( (1 / N_valid) * sum_{i=1}^3 exp(-d_i / tau) )
        # tau = 1.5m matches human stride length and lunging reach, properly capturing
        # collective numerical overloads without an artificial team-density headcount bonus.
        has_valid_home_top3 = ~np.isnan(tracking.d_H_top3[:, 0])
        home_top3_distances = np.where(
            np.isnan(tracking.d_H_top3[has_valid_home_top3]),
            FALLBACK_LARGE_DISTANCE_METERS,
            tracking.d_H_top3[has_valid_home_top3],
        )
        n_valid_home_players = np.sum(~np.isnan(tracking.d_H_top3[has_valid_home_top3]), axis=1)
        dist_home_softmin = -tau_softmin * np.log(
            np.sum(np.exp(-home_top3_distances / tau_softmin), axis=1) / np.maximum(1, n_valid_home_players)
        )
        log_B[has_valid_home_top3, STATE_H] = -dist_home_softmin / sigma

        has_valid_away_top3 = ~np.isnan(tracking.d_A_top3[:, 0])
        away_top3_distances = np.where(
            np.isnan(tracking.d_A_top3[has_valid_away_top3]),
            FALLBACK_LARGE_DISTANCE_METERS,
            tracking.d_A_top3[has_valid_away_top3],
        )
        n_valid_away_players = np.sum(~np.isnan(tracking.d_A_top3[has_valid_away_top3]), axis=1)
        dist_away_softmin = -tau_softmin * np.log(
            np.sum(np.exp(-away_top3_distances / tau_softmin), axis=1) / np.maximum(1, n_valid_away_players)
        )
        log_B[has_valid_away_top3, STATE_A] = -dist_away_softmin / sigma
    else:
        # Standard single nearest outfield player model:
        log_B[has_valid_outfield_distances, STATE_H] = -tracking.d_H[has_valid_outfield_distances] / sigma
        log_B[has_valid_outfield_distances, STATE_A] = -tracking.d_A[has_valid_outfield_distances] / sigma

    # =========================================================================
    # STEP 4.2: GOALKEEPER JURISDICTION & SWEEPER-KEEPER ACTIVATION
    # =========================================================================
    # Under IFAB Law 1, the 18-yard penalty area spans 16.5m from each goal line:
    ball_x_metric = tracking.ball_x_metric
    ball_y_metric = tracking.ball_y_metric
    ball_inside_either_penalty_box = (
        (ball_x_metric <= PENALTY_BOX_LENGTH_METERS) | (ball_x_metric >= AWAY_PENALTY_BOX_X_START)
    ) & (ball_y_metric >= PENALTY_BOX_Y_MIN) & (ball_y_metric <= PENALTY_BOX_Y_MAX)

    if episode_evidence is not None:
        # Dual Goalkeeper Jurisdiction:
        # a) Inside either 18-yard box: Keeper has full handling rights under IFAB Law 12.
        # b) Outside the box: Sweeper-keepers (Ter Stegen, Padilla) act within active radius 12.0m.
        is_home_keeper_active = ~np.isnan(tracking.d_H_gk) & (
            ball_inside_either_penalty_box | (tracking.d_H_gk < SWEEPER_KEEPER_ACTIVE_RADIUS_METERS)
        )
        log_B[is_home_keeper_active, STATE_H] = np.maximum(
            log_B[is_home_keeper_active, STATE_H], -tracking.d_H_gk[is_home_keeper_active] / sigma
        )

        is_away_keeper_active = ~np.isnan(tracking.d_A_gk) & (
            ball_inside_either_penalty_box | (tracking.d_A_gk < SWEEPER_KEEPER_ACTIVE_RADIUS_METERS)
        )
        log_B[is_away_keeper_active, STATE_A] = np.maximum(
            log_B[is_away_keeper_active, STATE_A], -tracking.d_A_gk[is_away_keeper_active] / sigma
        )
    else:
        # Fallback: Goalkeeper in immediate physical contact radius (< 1.5m):
        is_home_keeper_active = ~np.isnan(tracking.d_H_gk) & (tracking.d_H_gk < KEEPER_IMMEDIATE_CONTACT_RADIUS_METERS)
        log_B[is_home_keeper_active, STATE_H] = np.maximum(
            log_B[is_home_keeper_active, STATE_H], -tracking.d_H_gk[is_home_keeper_active] / sigma
        )

        is_away_keeper_active = ~np.isnan(tracking.d_A_gk) & (tracking.d_A_gk < KEEPER_IMMEDIATE_CONTACT_RADIUS_METERS)
        log_B[is_away_keeper_active, STATE_A] = np.maximum(
            log_B[is_away_keeper_active, STATE_A], -tracking.d_A_gk[is_away_keeper_active] / sigma
        )

    # =========================================================================
    # STEP 4.3: OPEN-SPACE LOOSE BALL DAMPENING & UNDEFLECTED FAST TRANSIT GATE
    # =========================================================================
    # A. Open-Space Loose Ball Dampening (d_min > 2.8m):
    # When the ball is rolling/bouncing in open grass with no player within tackling distance,
    # spatial evidence is dampened over [2.8m, 5.8m] down to 5% baseline, allowing HMM sequence
    # inertia to maintain possession smoothly without premature distance-based flipping.
    dist_home_safe = tracking.d_H if tracking.d_H is not None else np.full(n_frames, np.nan)
    dist_away_safe = tracking.d_A if tracking.d_A is not None else np.full(n_frames, np.nan)
    dist_home_gk_safe = tracking.d_H_gk if tracking.d_H_gk is not None else np.full(n_frames, np.nan)
    dist_away_gk_safe = tracking.d_A_gk if tracking.d_A_gk is not None else np.full(n_frames, np.nan)
    all_player_distances_matrix = np.column_stack([dist_home_safe, dist_away_safe, dist_home_gk_safe, dist_away_gk_safe])
    all_distances_nan = np.all(np.isnan(all_player_distances_matrix), axis=1)
    min_distance_to_ball_all_players = np.full(n_frames, np.nan, dtype=np.float32)
    if not np.all(all_distances_nan):
        min_distance_to_ball_all_players[~all_distances_nan] = np.nanmin(
            all_player_distances_matrix[~all_distances_nan], axis=1
        )

    is_loose_ball_in_open_space = ~np.isnan(min_distance_to_ball_all_players) & (
        min_distance_to_ball_all_players > LOOSE_BALL_DISTANCE_THRESHOLD
    )
    if np.any(is_loose_ball_in_open_space):
        loose_ball_dampening_factor = np.clip(
            1.0 - (min_distance_to_ball_all_players[is_loose_ball_in_open_space] - LOOSE_BALL_DISTANCE_THRESHOLD) / LOOSE_BALL_RAMP_SPAN,
            LOOSE_BALL_MIN_SCALE,
            1.0,
        )
        log_B[is_loose_ball_in_open_space, STATE_H] *= loose_ball_dampening_factor
        log_B[is_loose_ball_in_open_space, STATE_A] *= loose_ball_dampening_factor

    # B. Undeflected Fast Transit Gate (Decoupled from Ground Proximity):
    # When ball is in high-speed flight (v > 6.5 m/s) with directional deflection < 25.0 deg,
    # it is traveling ballistically. Suppress ground-plane spatial evidence by 90% to prevent
    # mid-air fly-bys from erroneously flipping possession to passive opposing players below.
    is_undeflected_fast_transit = np.zeros(n_frames, dtype=bool)
    if (
        hasattr(tracking, "ball_velocity")
        and tracking.ball_velocity is not None
        and hasattr(tracking, "ball_deflection_angle")
        and tracking.ball_deflection_angle is not None
    ):
        is_undeflected_fast_transit = (
            (tracking.ball_velocity > FAST_TRANSIT_VELOCITY_THRESHOLD)
            & (tracking.ball_deflection_angle < FAST_TRANSIT_MAX_DEFLECTION_DEG)
        )
        log_B[is_undeflected_fast_transit, STATE_H] *= FAST_TRANSIT_PROXIMITY_RETENTION
        log_B[is_undeflected_fast_transit, STATE_A] *= FAST_TRANSIT_PROXIMITY_RETENTION

    # =========================================================================
    # STEP 4.4: IN-PITCH LIVE PLAY INVARIANTS & STOPPAGE PRIORS
    # =========================================================================
    ball_observed_inside_pitch = (
        (tracking.ball_x_norm >= 0.0)
        & (tracking.ball_x_norm <= 1.0)
        & (tracking.ball_y_norm >= 0.0)
        & (tracking.ball_y_norm <= 1.0)
        & tracking.ball_observed
    )

    if episode_evidence is not None:
        prob_dead_ball = episode_evidence.p_dead

        # A. High-Confidence Stoppage (p_dead > 0.70):
        # Consensus between stationary ball and walking players emits unified stoppage evidence (+5.0 / -10.0 nats).
        is_high_confidence_stoppage = prob_dead_ball > STOPPAGE_CONFIRMED_DEAD_PROBABILITY
        log_B[is_high_confidence_stoppage, STATE_D] = CONFIRMED_STOPPAGE_DEAD_BONUS
        log_B[is_high_confidence_stoppage, STATE_H] = CONFIRMED_STOPPAGE_LIVE_PENALTY
        log_B[is_high_confidence_stoppage, STATE_A] = CONFIRMED_STOPPAGE_LIVE_PENALTY

        # B. High-Confidence Live Play Invariant (p_dead < 0.25 inside pitch):
        # Active sprinting/dribbling applies a finite -40.0 nat penalty against Dead Ball.
        is_high_confidence_live_play = (prob_dead_ball < LIVE_PLAY_MAX_DEAD_PROBABILITY) & ball_observed_inside_pitch
        log_B[is_high_confidence_live_play, STATE_D] = LIVE_PLAY_DEAD_PENALTY

        # C. Moderate Uncertainty Band (0.25 <= p_dead <= 0.70):
        # Linear interpolation from -5.0 to +2.0 nats across the ambiguous zone:
        is_ambiguous_stoppage_zone = (
            (prob_dead_ball >= AMBIGUOUS_DEAD_PROB_MIN)
            & (prob_dead_ball <= AMBIGUOUS_DEAD_PROB_MAX)
            & ball_observed_inside_pitch
        )
        if np.any(is_ambiguous_stoppage_zone):
            prob_span = AMBIGUOUS_DEAD_PROB_MAX - AMBIGUOUS_DEAD_PROB_MIN
            emission_span = LIKELY_STOPPAGE_BONUS - UNCONFIRMED_STOPPAGE_PENALTY
            normalized_prob = (prob_dead_ball[is_ambiguous_stoppage_zone] - AMBIGUOUS_DEAD_PROB_MIN) / prob_span
            log_B[is_ambiguous_stoppage_zone, STATE_D] = UNCONFIRMED_STOPPAGE_PENALTY + emission_span * normalized_prob

        # D. Flight Transit Attenuation (remaining fast transit frames):
        remaining_flight_mask = (episode_evidence.p_fast_transit > 0.5) & ~is_undeflected_fast_transit
        if np.any(remaining_flight_mask):
            flight_damping = 1.0 - 0.90 * episode_evidence.p_fast_transit[remaining_flight_mask]
            log_B[remaining_flight_mask, STATE_H] *= flight_damping
            log_B[remaining_flight_mask, STATE_A] *= flight_damping

        # E. Symmetric 5-Frame Contact Kernel (Deflections & Touches):
        if control_evidence is not None and np.any(control_evidence.is_deflection_event):
            deflection_peak_frames = np.flatnonzero(control_evidence.is_deflection_event)
            for peak_frame_idx in deflection_peak_frames:
                deflection_team = control_evidence.deflection_team[peak_frame_idx]
                if deflection_team in (0, 1):
                    is_passive_rebound = (
                        control_evidence.is_passive_deflection_H[peak_frame_idx]
                        if deflection_team == 0
                        else control_evidence.is_passive_deflection_A[peak_frame_idx]
                    )
                    # If passive deflection, apply only mild contact evidence (+1.0 nat) rather than control boost (+3.5 nats)
                    if is_passive_rebound:
                        budget_evidence = DEFLECTION_PASSIVE_TOUCH_BONUS
                    elif tracking.ball_observed[peak_frame_idx]:
                        budget_evidence = DEFLECTION_ACTIVE_TOUCH_BONUS
                    else:
                        budget_evidence = DEFLECTION_IMPUTED_TOUCH_BONUS

                    target_state = STATE_H if deflection_team == 0 else STATE_A
                    for offset, profile_fraction in enumerate(CONTACT_KERNEL_PROFILE, start=-2):
                        frame_idx = peak_frame_idx + offset
                        if 0 <= frame_idx < n_frames:
                            log_B[frame_idx, target_state] += budget_evidence * profile_fraction
        elif (
            hasattr(tracking, "ball_velocity")
            and tracking.ball_velocity is not None
            and hasattr(tracking, "ball_deflection_angle")
            and tracking.ball_deflection_angle is not None
        ):
            # Fallback when Stage 3 control_evidence is absent:
            has_significant_deflection = (
                (tracking.ball_deflection_angle > FALLBACK_DEFLECTION_MIN_ANGLE)
                & (tracking.ball_velocity > FALLBACK_DEFLECTION_MIN_VELOCITY)
            )
            fallback_deflect_home = (
                has_significant_deflection
                & ~np.isnan(tracking.d_H)
                & (tracking.d_H < FALLBACK_DEFLECTION_MAX_DISTANCE)
                & (tracking.d_H < tracking.d_A - FALLBACK_DEFLECTION_ADVANTAGE_MARGIN)
            )
            fallback_deflect_away = (
                has_significant_deflection
                & ~np.isnan(tracking.d_A)
                & (tracking.d_A < FALLBACK_DEFLECTION_MAX_DISTANCE)
                & (tracking.d_A < tracking.d_H - FALLBACK_DEFLECTION_ADVANTAGE_MARGIN)
            )

            def _apply_fallback_event_kernel(event_mask: np.ndarray, state_idx: int) -> None:
                if not np.any(event_mask):
                    return
                diff = np.diff(np.concatenate(([0], event_mask.astype(np.int8), [0])))
                starts = np.flatnonzero(diff == 1)
                ends = np.flatnonzero(diff == -1)
                for start_f, end_f in zip(starts, ends):
                    cluster_peak = start_f + int(np.argmax(tracking.ball_deflection_angle[start_f:end_f]))
                    budget = DEFLECTION_ACTIVE_TOUCH_BONUS if tracking.ball_observed[cluster_peak] else DEFLECTION_IMPUTED_TOUCH_BONUS
                    for offset, profile_fraction in enumerate(CONTACT_KERNEL_PROFILE, start=-2):
                        target_frame = cluster_peak + offset
                        if 0 <= target_frame < n_frames:
                            log_B[target_frame, state_idx] += budget * profile_fraction

            _apply_fallback_event_kernel(fallback_deflect_home, STATE_H)
            _apply_fallback_event_kernel(fallback_deflect_away, STATE_A)

        # F. Positive Controlled Possession Boost (Harmonized with Stage 3):
        if control_evidence is not None:
            log_B[:, STATE_H] += CONTROLLED_POSSESSION_BONUS * control_evidence.control_confidence_H
            log_B[:, STATE_A] += CONTROLLED_POSSESSION_BONUS * control_evidence.control_confidence_A

        # G. Contested 50-50 Challenge Dampening:
        if hasattr(episode_evidence, "p_contested") and episode_evidence.p_contested is not None:
            is_contested_duel = episode_evidence.p_contested > CONTESTED_DUEL_PROB_THRESHOLD
            if np.any(is_contested_duel):
                contest_damping = 1.0 - CONTESTED_DUEL_DAMPING_FACTOR * episode_evidence.p_contested[is_contested_duel]
                log_B[is_contested_duel, STATE_H] *= contest_damping
                log_B[is_contested_duel, STATE_A] *= contest_damping

        # H. Set-Piece Restart Kick Impulse (Strike Frame Snap):
        confirmed_restart_frames = np.flatnonzero(episode_evidence.p_restart_candidate > RESTART_CONFIRMED_PROB_THRESHOLD)
        for restart_frame_idx in confirmed_restart_frames:
            kicking_team = episode_evidence.restart_team[restart_frame_idx]
            if kicking_team == 0:  # Home (FC Barcelona)
                log_B[restart_frame_idx, STATE_H] = RESTART_KICK_TEAM_BOOST
                log_B[restart_frame_idx, STATE_A] = RESTART_OPPONENT_AND_DEAD_PENALTY
                log_B[restart_frame_idx, STATE_D] = RESTART_OPPONENT_AND_DEAD_PENALTY
            elif kicking_team == 1:  # Away (Athletic Club)
                log_B[restart_frame_idx, STATE_A] = RESTART_KICK_TEAM_BOOST
                log_B[restart_frame_idx, STATE_H] = RESTART_OPPONENT_AND_DEAD_PENALTY
                log_B[restart_frame_idx, STATE_D] = RESTART_OPPONENT_AND_DEAD_PENALTY
    else:
        # Fallback: In-pitch active play strictly bars Dead Ball
        log_B[ball_observed_inside_pitch, STATE_D] = NEG_INF

    # =========================================================================
    # STEP 4.5: OUT-OF-BOUNDS GEOMETRIC GATE & METADATA BOUNDARIES
    # =========================================================================
    # A. Out-of-Bounds (OOB) Geometric Gate (IFAB Law 9):
    is_ball_out_of_bounds = (
        (tracking.ball_x_norm < 0.0)
        | (tracking.ball_x_norm > 1.0)
        | (tracking.ball_y_norm < 0.0)
        | (tracking.ball_y_norm > 1.0)
    ) & tracking.ball_observed

    log_B[is_ball_out_of_bounds, STATE_D] = CONFIRMED_STOPPAGE_DEAD_BONUS
    log_B[is_ball_out_of_bounds, STATE_H] = CONFIRMED_STOPPAGE_LIVE_PENALTY
    log_B[is_ball_out_of_bounds, STATE_A] = CONFIRMED_STOPPAGE_LIVE_PENALTY

    # B. Missing Ball Gap Categorization (M_t = 0):
    prob_dead = episode_evidence.p_dead if episode_evidence is not None else np.zeros(n_frames)
    is_restart_kick = (
        (episode_evidence.p_restart_candidate > RESTART_CONFIRMED_PROB_THRESHOLD)
        if episode_evidence is not None
        else np.zeros(n_frames, dtype=bool)
    )

    anchor_restarts = None
    anchor_crossings = None
    if dead_ball_anchors and episode_evidence is not None:
        anchor_restarts = episode_evidence.set_piece_restarts or {}
        anchor_crossings = episode_evidence.line_crossings
        if anchor_crossings is None:
            anchor_crossings = np.zeros(0, dtype=np.int64)

    for gap in tracking.missing_gaps:
        gap_start, gap_end = gap.start_frame, gap.end_frame
        non_restart_mask = ~is_restart_kick[gap_start : gap_end + 1]
        if not np.any(non_restart_mask):
            continue
        gap_frame_indices = np.arange(gap_start, gap_end + 1)[non_restart_mask]

        if gap.category == "stoppage":
            if anchor_crossings is not None:
                inside = anchor_crossings[(anchor_crossings > gap_start) & (anchor_crossings <= gap_end + 1)]
                if len(inside) and inside[0] - gap_start <= GAP_CROSSING_MAX_LIVE_FRAMES:
                    # The ball was lost on the pitch and crossed the line later: live until then.
                    live_part = gap_frame_indices[gap_frame_indices < inside[0]]
                    log_B[live_part, STATE_H] = GAP_ACTIVE_LIVE_EMISSION
                    log_B[live_part, STATE_A] = GAP_ACTIVE_LIVE_EMISSION
                    log_B[live_part, STATE_D] = NEG_INF
                    gap_frame_indices = gap_frame_indices[gap_frame_indices >= inside[0]]
            log_B[gap_frame_indices, STATE_D] = HALFTIME_DEAD_BASELINE
            log_B[gap_frame_indices, STATE_H] = HALFTIME_LIVE_PENALTY
            log_B[gap_frame_indices, STATE_A] = HALFTIME_LIVE_PENALTY

        elif gap.category == "active":
            live_active_sub_indices = gap_frame_indices[prob_dead[gap_frame_indices] < GAP_ACTIVE_MAX_DEAD_PROBABILITY]
            if len(live_active_sub_indices) > 0:
                log_B[live_active_sub_indices, STATE_H] = GAP_ACTIVE_LIVE_EMISSION
                log_B[live_active_sub_indices, STATE_A] = GAP_ACTIVE_LIVE_EMISSION
                log_B[live_active_sub_indices, STATE_D] = NEG_INF

        elif gap.category == "ambiguous":
            log_B[gap_frame_indices, STATE_D] = UNCLEAR_GAP_DEAD_BONUS
            log_B[gap_frame_indices, STATE_H] = UNCLEAR_GAP_LIVE_PENALTY
            log_B[gap_frame_indices, STATE_A] = UNCLEAR_GAP_LIVE_PENALTY

    # C. Smart Tagging Structural Priors (Set-Pieces & Goal Windows):
    if smart_tags:
        # 1. Set-Piece Stationary Preparation:
        for set_piece in smart_tags.get("set_pieces", []):
            sp_start = max(0, set_piece["start"])
            sp_end = min(n_frames - 1, set_piece["end"])
            prep_limit = min(sp_end, sp_start + SET_PIECE_PREP_MAX_FRAMES)
            if anchor_restarts is not None and anchor_restarts.get(set_piece["start"]) is not None:
                # Preparation ends at the restart; after it the ball is live even if it rolls slowly.
                prep_limit = min(prep_limit, anchor_restarts[set_piece["start"]] - 1)
            if prep_limit < sp_start:
                continue
            is_ball_stationary = (
                tracking.ball_velocity[sp_start : prep_limit + 1] < SET_PIECE_BALL_STATIONARY_VELOCITY
            ) & tracking.ball_observed[sp_start : prep_limit + 1]
            min_opp_dist = np.minimum(
                tracking.d_H[sp_start : prep_limit + 1], tracking.d_A[sp_start : prep_limit + 1]
            )
            is_uncontested = np.isnan(min_opp_dist) | (min_opp_dist > SET_PIECE_UNCONTESTED_DISTANCE)
            is_prep_dead = is_ball_stationary & is_uncontested

            prep_indices = np.arange(sp_start, prep_limit + 1)[is_prep_dead]
            if len(prep_indices) > 0:
                log_B[prep_indices, STATE_D] = CONFIRMED_STOPPAGE_DEAD_BONUS
                log_B[prep_indices, STATE_H] = CONFIRMED_STOPPAGE_LIVE_PENALTY
                log_B[prep_indices, STATE_A] = CONFIRMED_STOPPAGE_LIVE_PENALTY

        # 2. Goal Celebration & Kickoff Coordination:
        for goal_event in smart_tags.get("goals", []):
            goal_start = goal_event["start"]
            goal_end = min(n_frames - 1, goal_event["end"] + GOAL_CELEBRATION_MAX_FRAMES)

            # Truncate at subsequent kickoff set-piece:
            for sp in smart_tags.get("set_pieces", []):
                if sp["start"] >= goal_start and "kick off" in sp.get("type", "").lower():
                    kick = anchor_restarts.get(sp["start"]) if anchor_restarts is not None else None
                    # With anchors the goal window runs to the detected kick-off, not the clip start.
                    goal_end = min(goal_end, kick - 1 if kick is not None else sp["start"])
                    break

            # Locate exact goal line crossing:
            goal_crossing_frame = goal_start + int(0.25 * (goal_end - goal_start))
            for cand_frame in range(goal_start, min(n_frames - 1, goal_start + GOAL_CROSSING_SEARCH_FRAMES)):
                if tracking.ball_observed[cand_frame] and (
                    tracking.ball_x_metric[cand_frame] < GOAL_LINE_X_HOME
                    or tracking.ball_x_metric[cand_frame] > GOAL_LINE_X_AWAY
                ):
                    goal_crossing_frame = cand_frame
                    break

            # Buildup is live for attacking team:
            attacking_team = goal_event.get("team")
            if attacking_team in ("H", "A"):
                attacking_state = STATE_H if attacking_team == "H" else STATE_A
                log_B[goal_start:goal_crossing_frame, attacking_state] += GOAL_BUILDUP_LIVE_BONUS

            # Dead ball strictly applies post-crossing:
            log_B[goal_crossing_frame : goal_end + 1, STATE_D] = GOAL_CELEBRATION_DEAD_BONUS
            log_B[goal_crossing_frame : goal_end + 1, STATE_H] = HALFTIME_LIVE_PENALTY
            log_B[goal_crossing_frame : goal_end + 1, STATE_A] = HALFTIME_LIVE_PENALTY

        # 3. Capped Active Event Priors:
        for phase in smart_tags.get("active_phases", []):
            target_team_code = phase.get("team")
            if target_team_code not in ("H", "A"):
                continue
            target_phase_state = STATE_H if target_team_code == "H" else STATE_A
            phase_start = max(0, phase["start"])
            phase_end = min(n_frames - 1, phase["end"])
            event_length = phase_end - phase_start + 1
            if event_length <= 0:
                continue

            per_frame_prior_bonus = min(
                REFEREE_EVENT_PER_FRAME_BONUS, event_prior_cap / float(event_length)
            )
            log_B[phase_start : phase_end + 1, target_phase_state] += per_frame_prior_bonus

    # D. Static Non-Gameplay Dead Blocks:
    if n_frames >= H1_END:
        static_dead_intervals = [
            (0, H1_START - 1),
            (H1_END + 1, H2_START - 1),
            (H2_START, (h2_kickoff if h2_kickoff is not None else H2_KICKOFF) - 1),
            (H2_END + 1, n_frames - 1),
        ]
        for block_start, block_end in static_dead_intervals:
            if block_start <= block_end and block_start < n_frames:
                block_end = min(n_frames - 1, block_end)
                log_B[block_start : block_end + 1, STATE_D] = HALFTIME_DEAD_BASELINE
                log_B[block_start : block_end + 1, STATE_H] = HALFTIME_LIVE_PENALTY
                log_B[block_start : block_end + 1, STATE_A] = HALFTIME_LIVE_PENALTY

    # =========================================================================
    # STEP 4.6: OPTIONAL AUGMENTED 5-STATE EXPANSION
    # =========================================================================
    if augmented_5state:
        log_B5 = np.zeros((n_frames, 5), dtype=np.float32)
        # 0: H_ctrl, 1: H_unctrl, 2: A_ctrl, 3: A_unctrl, 4: Dead
        log_B5[:, 0] = log_B[:, STATE_H]
        log_B5[:, 2] = log_B[:, STATE_A]
        log_B5[:, 4] = log_B[:, STATE_D]

        # Baseline for uncontrolled flight / loose ball:
        prob_transit = (
            episode_evidence.p_fast_transit if episode_evidence is not None else np.zeros(n_frames, dtype=np.float32)
        )
        loose_ball_conf = (
            control_evidence.loose_ball_confidence if control_evidence is not None else np.zeros(n_frames, dtype=np.float32)
        )
        contest_conf = (
            control_evidence.contest_confidence if control_evidence is not None else np.zeros(n_frames, dtype=np.float32)
        )

        uncontrolled_baseline = (
            AUGMENTED_UNCONTROLLED_BASE_OFFSET
            + AUGMENTED_UNCONTROLLED_LOOSE_BONUS * loose_ball_conf
            + AUGMENTED_UNCONTROLLED_TRANSIT_BONUS * prob_transit
            + AUGMENTED_UNCONTROLLED_CONTEST_BONUS * contest_conf
        )
        log_B5[:, 1] = uncontrolled_baseline
        log_B5[:, 3] = uncontrolled_baseline

        # When dead/stoppage applies, suppress uncontrolled states as well:
        dead_dominant_mask = log_B[:, STATE_D] > log_B[:, STATE_H]
        log_B5[dead_dominant_mask, 1] = log_B[dead_dominant_mask, STATE_H]
        log_B5[dead_dominant_mask, 3] = log_B[dead_dominant_mask, STATE_A]

        return log_B5

    return log_B
