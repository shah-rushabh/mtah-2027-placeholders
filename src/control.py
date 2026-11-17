"""Phase 3: Player Interaction & Controlled Possession Engine.

Decouples candidate physical interaction from controlled possession and intentional action:
  - Distinguishes intentional actions (first-time pass, flick, clearance, shot) from passive deflections/rebounds.
  - Preserves genuine short controlled spells (< 1.0s) through intentional action tagging.
  - Computes radial closing and separating velocities to track incoming approach vs outgoing release.
  - Allocates a single bounded event budget with a strictly symmetric kernel centered on peak contact.
  - Formulates two-tiered candidate turnovers vs confirmed turnovers.
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.loader import TrackingData, FPS, PITCH_LENGTH, PITCH_WIDTH
from src.episodes import EpisodeEvidence


# ==============================================================================
# SPATIAL, KINEMATIC & TEMPORAL CONSTANTS
# ==============================================================================
# Spatial Radii & Distances (meters)
PHYSICAL_CONTACT_RADIUS_METERS = 1.8       # 1.80m: Maximum human limb reach (lunging/sliding tackle)
CONTROLLED_CARRY_RADIUS_METERS = 1.6       # 1.60m: Spatial boundary for settled ball carrying / dribbling
LOOSE_BALL_DISTANCE_THRESHOLD = 2.8        # 2.80m: 1.8m reach + 1.0m stride; beyond this, ball is free in grass
LOOSE_BALL_RAMP_SPAN = 2.0                 # 2.00m: Scaling denominator, reaches loose_ball=1.0 at 4.80m
CONTESTED_DUEL_RADIUS_METERS = 2.5         # 2.50m: Duel boundary where opposing players actively contest
CONTESTED_EQUIDISTANCE_TOLERANCE = 0.8     # 0.80m: Proximity difference below which duel is 50-50
CONVERGENCE_PROXIMITY_CEILING = 3.0        # 3.00m: Spatial ceiling for dynamic approach convergence
PROXIMITY_DOMINANCE_MARGIN = 0.30          # 0.30m (~boot length): Clear body positioning advantage
SPATIAL_CONTROL_DECAY_SCALE = 2.0          # 2.00m: Exponential decay scale parameter for control score
CONTROL_CONFIRMED_TURNOVER_FLOOR = 0.35    # 0.35: Control floor corresponding to 2.10m Spearman pitch control
TURNOVER_DOMINANCE_DIFFERENCE = 0.10       # 0.10: Control margin required to confirm possession change
UNOBSERVED_DISTANCE_FALLBACK = 99.0        # 99.0m: Fallback distance when coordinates are unobserved (NaN)

# Temporal Quanta & Windows (frames @ 25 Hz)
DELTA_TIME_FRAME_SPAN = 2                  # 2 frames: Centered finite difference span (80ms @ 25Hz)
DEFLECTION_GROUPING_WINDOW_FRAMES = 5      # 5 frames (0.20s): Universal neuromuscular reaction window
MIN_CONTROL_SETTLED_FRAMES = 6             # 6 frames (0.24s): Single running stride smoothing window
ACTION_ENVELOPE_PRE_FRAMES = 1             # 1 frame (0.04s): Pre-contact wind-up window
ACTION_ENVELOPE_POST_FRAMES = 4            # 4 frames (0.16s): Post-contact launch/follow-through window

# Kinematic Velocities & Accelerations
DEFLECTION_RAW_ANGLE_THRESHOLD_DEG = 25.0  # 25.0 deg: Minimum angular trajectory kink for deflection
BALL_MOTION_NOISE_FLOOR = 2.0              # 2.00 m/s (7.2 km/h, 8:20 min/km): Filters stationary coordinate noise
KICK_IMPULSE_MIN_ACCELERATION = 15.0       # 15.0 m/s^2: Minimum acceleration spike of a boot strike
KICK_DRIVEN_LAUNCH_VELOCITY = 4.5          # 4.50 m/s (16.2 km/h, 3:42 min/km): Driven kick launch speed
INTENTIONAL_REDIRECT_MIN_VELOCITY = 5.0    # 5.00 m/s (18.0 km/h, 3:20 min/km): Volley/header flick exit speed
SEPARATING_PASS_MIN_VELOCITY = 2.0         # 2.00 m/s (7.2 km/h, 8:20 min/km): Separation rate (FIFA Zone 1/2)
RADIAL_CONVERGENCE_MIN_VELOCITY = 1.0      # 1.00 m/s (3.6 km/h, 16:40 min/km): Minimum radial approach rate
FAST_TRANSIT_BALLISTIC_PROBABILITY = 0.50  # 0.50: Transit probability threshold confirming aerial ballistics

# Probability Levels & Scaling Factors
BALLISTIC_CONTROL_SUPPRESSION = 0.85       # 85%: Suppression of control evidence during fast ball flight
INTENTIONAL_ACTION_CONTROL_BOOST = 0.85    # 0.85: Elevated control certainty granted to confirmed 1-touch actions
MIN_TRACKING_CONFIDENCE_THRESHOLD = 0.30   # 0.30: Minimum reliability required for candidate turnovers


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class ControlEvidence:
    """Factorized continuous interaction, intentional action, and control evidence."""
    control_confidence_H: np.ndarray        # shape (N,), float32 in [0, 1]: Home sustained carry/dribble
    control_confidence_A: np.ndarray        # shape (N,), float32 in [0, 1]: Away sustained carry/dribble
    contact_confidence_H: np.ndarray        # shape (N,), float32 in [0, 1]: Home physical contact/touch
    contact_confidence_A: np.ndarray        # shape (N,), float32 in [0, 1]: Away physical contact/touch
    loose_ball_confidence: np.ndarray       # shape (N,), float32 in [0, 1]: Ball uncontrolled in open grass
    contest_confidence: np.ndarray          # shape (N,), float32 in [0, 1]: 50-50 physical duel
    is_deflection_event: np.ndarray         # shape (N,), bool: True at single peak frame of grouped deflection
    deflection_team: np.ndarray             # shape (N,), int8: 0 for H, 1 for A, -1 for none
    turnover_candidate_H_to_A: np.ndarray   # shape (N,), bool: Candidate possession change H -> A
    turnover_candidate_A_to_H: np.ndarray   # shape (N,), bool: Candidate possession change A -> H
    intentional_action_H: Optional[np.ndarray] = None        # shape (N,), bool: True on frames of intentional pass/shot/clearance
    intentional_action_A: Optional[np.ndarray] = None        # shape (N,), bool: True on frames of intentional pass/shot/clearance
    is_passive_deflection_H: Optional[np.ndarray] = None     # shape (N,), bool: Glancing touch/rebound without control/intention
    is_passive_deflection_A: Optional[np.ndarray] = None     # shape (N,), bool: Glancing touch/rebound without control/intention
    turnover_confirmed_H_to_A: Optional[np.ndarray] = None   # shape (N,), bool: Confirmed possession change H -> A
    turnover_confirmed_A_to_H: Optional[np.ndarray] = None   # shape (N,), bool: Confirmed possession change A -> H

    def __post_init__(self):
        total_frames = len(self.control_confidence_H)
        if self.turnover_confirmed_H_to_A is None:
            self.turnover_confirmed_H_to_A = self.turnover_candidate_H_to_A
        if self.turnover_confirmed_A_to_H is None:
            self.turnover_confirmed_A_to_H = self.turnover_candidate_A_to_H
        if self.intentional_action_H is None:
            self.intentional_action_H = np.zeros(total_frames, dtype=bool)
        if self.intentional_action_A is None:
            self.intentional_action_A = np.zeros(total_frames, dtype=bool)
        if self.is_passive_deflection_H is None:
            self.is_passive_deflection_H = np.zeros(total_frames, dtype=bool)
        if self.is_passive_deflection_A is None:
            self.is_passive_deflection_A = np.zeros(total_frames, dtype=bool)


# ==============================================================================
# MAIN CONTROL EVIDENCE ENGINE
# ==============================================================================
def compute_control_evidence(
    tracking: TrackingData,
    episodes: Optional[EpisodeEvidence] = None,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    contact_radius: float = PHYSICAL_CONTACT_RADIUS_METERS,
    control_radius: float = CONTROLLED_CARRY_RADIUS_METERS,
    loose_ball_threshold: float = LOOSE_BALL_DISTANCE_THRESHOLD,
    contest_radius: float = CONTESTED_DUEL_RADIUS_METERS,
    contest_diff: float = CONTESTED_EQUIDISTANCE_TOLERANCE,
    min_control_settled_frames: int = MIN_CONTROL_SETTLED_FRAMES,
    deflection_grouping_window: int = DEFLECTION_GROUPING_WINDOW_FRAMES,
) -> ControlEvidence:
    """Computes factorized interaction, intentional action, and control evidence."""
    n_frames = len(tracking.ball_observed)
    pitch_length, pitch_width = pitch

    control_confidence_home = np.zeros(n_frames, dtype=np.float32)
    control_confidence_away = np.zeros(n_frames, dtype=np.float32)
    intentional_action_home = np.zeros(n_frames, dtype=bool)
    intentional_action_away = np.zeros(n_frames, dtype=bool)
    contact_confidence_home = np.zeros(n_frames, dtype=np.float32)
    contact_confidence_away = np.zeros(n_frames, dtype=np.float32)
    is_passive_deflection_home = np.zeros(n_frames, dtype=bool)
    is_passive_deflection_away = np.zeros(n_frames, dtype=bool)
    loose_ball_confidence = np.zeros(n_frames, dtype=np.float32)
    contest_confidence = np.zeros(n_frames, dtype=np.float32)
    is_deflection_event = np.zeros(n_frames, dtype=bool)
    deflection_team = np.full(n_frames, -1, dtype=np.int8)
    cand_turnover_home_to_away = np.zeros(n_frames, dtype=bool)
    cand_turnover_away_to_home = np.zeros(n_frames, dtype=bool)
    conf_turnover_home_to_away = np.zeros(n_frames, dtype=bool)
    conf_turnover_away_to_home = np.zeros(n_frames, dtype=bool)

    # Resolve player-to-ball distances, protecting against NaN during missing gaps
    dist_home_outfield = np.nan_to_num(tracking.d_H, nan=UNOBSERVED_DISTANCE_FALLBACK)
    dist_away_outfield = np.nan_to_num(tracking.d_A, nan=UNOBSERVED_DISTANCE_FALLBACK)
    dist_home_gk = np.nan_to_num(tracking.d_H_gk, nan=UNOBSERVED_DISTANCE_FALLBACK) if tracking.d_H_gk is not None else np.full(n_frames, UNOBSERVED_DISTANCE_FALLBACK)
    dist_away_gk = np.nan_to_num(tracking.d_A_gk, nan=UNOBSERVED_DISTANCE_FALLBACK) if tracking.d_A_gk is not None else np.full(n_frames, UNOBSERVED_DISTANCE_FALLBACK)

    # Effective team distance: closest team member including goalkeeper
    dist_home_effective = np.minimum(dist_home_outfield, dist_home_gk)
    dist_away_effective = np.minimum(dist_away_outfield, dist_away_gk)

    # Ball kinematics & sensor reliability
    ball_speed = np.nan_to_num(tracking.ball_velocity, nan=0.0)
    ball_deflection_angle = np.nan_to_num(tracking.ball_deflection_angle, nan=0.0) if tracking.ball_deflection_angle is not None else np.zeros(n_frames)
    tracking_confidence = tracking.tracking_confidence if tracking.tracking_confidence is not None else np.ones(n_frames, dtype=np.float32)

    fast_transit_prob = episodes.p_fast_transit if episodes is not None else np.zeros(n_frames, dtype=np.float32)
    is_ballistic = fast_transit_prob > FAST_TRANSIT_BALLISTIC_PROBABILITY

    # -------------------------------------------------------------------------
    # 1. Radial Closing and Separating Velocity Estimation
    # -------------------------------------------------------------------------
    closing_vel_home = np.zeros(n_frames, dtype=np.float32)
    closing_vel_away = np.zeros(n_frames, dtype=np.float32)
    sep_vel_home = np.zeros(n_frames, dtype=np.float32)
    sep_vel_away = np.zeros(n_frames, dtype=np.float32)

    if n_frames > DELTA_TIME_FRAME_SPAN:
        delta_time_two_frames = float(DELTA_TIME_FRAME_SPAN) / FPS  # 0.080s centered window
        # Positive closing velocity: distance is decreasing (approaching)
        closing_vel_home[1:-1] = -(dist_home_effective[2:] - dist_home_effective[:-2]) / delta_time_two_frames
        closing_vel_away[1:-1] = -(dist_away_effective[2:] - dist_away_effective[:-2]) / delta_time_two_frames
        # Positive separating velocity: distance is increasing (launching outward)
        sep_vel_home[1:-1] = (dist_home_effective[2:] - dist_home_effective[:-2]) / delta_time_two_frames
        sep_vel_away[1:-1] = (dist_away_effective[2:] - dist_away_effective[:-2]) / delta_time_two_frames

    # Ball acceleration: rate of change of ball speed via centered difference
    ball_accel = np.zeros(n_frames, dtype=np.float32)
    if n_frames > DELTA_TIME_FRAME_SPAN:
        delta_time_two_frames = float(DELTA_TIME_FRAME_SPAN) / FPS
        ball_accel[1:-1] = (ball_speed[2:] - ball_speed[:-2]) / delta_time_two_frames

    # -------------------------------------------------------------------------
    # 2. Deflection Event Grouping with True Symmetric Centering
    # -------------------------------------------------------------------------
    is_deflection_raw = (ball_deflection_angle > DEFLECTION_RAW_ANGLE_THRESHOLD_DEG) & (ball_speed > BALL_MOTION_NOISE_FLOOR)
    if episodes is not None and getattr(episodes, "is_deflection", None) is not None:
        is_deflection_raw = is_deflection_raw | episodes.is_deflection

    deflection_indices = np.flatnonzero(is_deflection_raw)
    if len(deflection_indices) > 0:
        event_clusters = []
        curr_cluster = [deflection_indices[0]]
        for deflection_frame_idx in deflection_indices[1:]:
            if deflection_frame_idx - curr_cluster[-1] <= deflection_grouping_window:
                curr_cluster.append(deflection_frame_idx)
            else:
                event_clusters.append(curr_cluster)
                curr_cluster = [deflection_frame_idx]
        event_clusters.append(curr_cluster)

        for cluster in event_clusters:
            # Pick representative event frame: minimum player-ball distance during impact
            best_contact_frame = cluster[0]
            min_contact_dist = min(dist_home_effective[best_contact_frame], dist_away_effective[best_contact_frame])
            for cluster_frame in cluster[1:]:
                cluster_frame_dist = min(dist_home_effective[cluster_frame], dist_away_effective[cluster_frame])
                if (
                    cluster_frame_dist < min_contact_dist
                    or (cluster_frame_dist == min_contact_dist and ball_deflection_angle[cluster_frame] > ball_deflection_angle[best_contact_frame])
                ):
                    min_contact_dist = cluster_frame_dist
                    best_contact_frame = cluster_frame

            is_deflection_event[best_contact_frame] = True
            if dist_home_effective[best_contact_frame] < dist_away_effective[best_contact_frame]:
                deflection_team[best_contact_frame] = 0
            elif dist_away_effective[best_contact_frame] < dist_home_effective[best_contact_frame]:
                deflection_team[best_contact_frame] = 1

    # -------------------------------------------------------------------------
    # 3. Intentional Action Detection (First-time Pass, Flick, Clearance, Shot)
    # -------------------------------------------------------------------------
    for frame_idx in range(1, n_frames - 2):
        # Home Intentional Action
        if dist_home_effective[frame_idx] <= contact_radius and dist_home_effective[frame_idx] < dist_away_effective[frame_idx] - PROXIMITY_DOMINANCE_MARGIN:
            has_kick_impulse = (ball_accel[frame_idx] > KICK_IMPULSE_MIN_ACCELERATION) and (ball_speed[frame_idx + 1] > KICK_DRIVEN_LAUNCH_VELOCITY)
            has_separating_pass = (sep_vel_home[frame_idx] > SEPARATING_PASS_MIN_VELOCITY) and (ball_speed[frame_idx] > KICK_DRIVEN_LAUNCH_VELOCITY)
            next_frame_idx = min(n_frames - 1, frame_idx + 1)
            has_intentional_redirect = is_deflection_event[frame_idx] and (sep_vel_home[next_frame_idx] > SEPARATING_PASS_MIN_VELOCITY) and (ball_speed[next_frame_idx] > INTENTIONAL_REDIRECT_MIN_VELOCITY)

            if has_kick_impulse or has_separating_pass or has_intentional_redirect:
                # Mark 5-frame intentional action envelope [t-1 .. t+3]
                action_window_start = max(0, frame_idx - ACTION_ENVELOPE_PRE_FRAMES)
                action_window_end = min(n_frames, frame_idx + ACTION_ENVELOPE_POST_FRAMES)
                intentional_action_home[action_window_start:action_window_end] = True
            elif is_deflection_event[frame_idx] and not has_kick_impulse and not has_separating_pass:
                is_passive_deflection_home[frame_idx] = True

        # Away Intentional Action
        if dist_away_effective[frame_idx] <= contact_radius and dist_away_effective[frame_idx] < dist_home_effective[frame_idx] - PROXIMITY_DOMINANCE_MARGIN:
            has_kick_impulse = (ball_accel[frame_idx] > KICK_IMPULSE_MIN_ACCELERATION) and (ball_speed[frame_idx + 1] > KICK_DRIVEN_LAUNCH_VELOCITY)
            has_separating_pass = (sep_vel_away[frame_idx] > SEPARATING_PASS_MIN_VELOCITY) and (ball_speed[frame_idx] > KICK_DRIVEN_LAUNCH_VELOCITY)
            next_frame_idx = min(n_frames - 1, frame_idx + 1)
            has_intentional_redirect = is_deflection_event[frame_idx] and (sep_vel_away[next_frame_idx] > SEPARATING_PASS_MIN_VELOCITY) and (ball_speed[next_frame_idx] > INTENTIONAL_REDIRECT_MIN_VELOCITY)

            if has_kick_impulse or has_separating_pass or has_intentional_redirect:
                action_window_start = max(0, frame_idx - ACTION_ENVELOPE_PRE_FRAMES)
                action_window_end = min(n_frames, frame_idx + ACTION_ENVELOPE_POST_FRAMES)
                intentional_action_away[action_window_start:action_window_end] = True
            elif is_deflection_event[frame_idx] and not has_kick_impulse and not has_separating_pass:
                is_passive_deflection_away[frame_idx] = True

    # -------------------------------------------------------------------------
    # 4. Sustained Controlled Possession Evidence (Dribbling, Carrying, Settling)
    # -------------------------------------------------------------------------
    raw_control_home = (
        np.exp(-np.maximum(0.0, dist_home_effective) / SPATIAL_CONTROL_DECAY_SCALE)
        * (1.0 - BALLISTIC_CONTROL_SUPPRESSION * is_ballistic)
        * tracking_confidence
    )
    raw_control_away = (
        np.exp(-np.maximum(0.0, dist_away_effective) / SPATIAL_CONTROL_DECAY_SCALE)
        * (1.0 - BALLISTIC_CONTROL_SUPPRESSION * is_ballistic)
        * tracking_confidence
    )

    # Temporal smoothing over running stride window (6 frames = 0.24s)
    stride_smoothing_kernel = np.ones(min_control_settled_frames, dtype=np.float32) / float(min_control_settled_frames)
    padding_length = min_control_settled_frames // 2
    padded_control_home = np.pad(raw_control_home, (padding_length, padding_length), mode="edge")
    padded_control_away = np.pad(raw_control_away, (padding_length, padding_length), mode="edge")
    control_confidence_home = np.convolve(padded_control_home, stride_smoothing_kernel, mode="same")[padding_length:padding_length + n_frames].astype(np.float32)
    control_confidence_away = np.convolve(padded_control_away, stride_smoothing_kernel, mode="same")[padding_length:padding_length + n_frames].astype(np.float32)

    # Elevate control confidence when an intentional 1-touch action is executed
    control_confidence_home = np.maximum(control_confidence_home, np.where(intentional_action_home, INTENTIONAL_ACTION_CONTROL_BOOST, 0.0).astype(np.float32))
    control_confidence_away = np.maximum(control_confidence_away, np.where(intentional_action_away, INTENTIONAL_ACTION_CONTROL_BOOST, 0.0).astype(np.float32))

    # -------------------------------------------------------------------------
    # 5. Continuous Contact Evidence (Physical Touches, Blocks, Duels)
    # -------------------------------------------------------------------------
    contact_confidence_home = np.clip((contact_radius - dist_home_effective) / contact_radius, 0.0, 1.0) * tracking_confidence
    contact_confidence_away = np.clip((contact_radius - dist_away_effective) / contact_radius, 0.0, 1.0) * tracking_confidence

    # -------------------------------------------------------------------------
    # 6. Loose-Ball & Contested 50-50 Duel Evidence
    # -------------------------------------------------------------------------
    min_dist_both_teams = np.minimum(dist_home_effective, dist_away_effective)
    loose_ball_confidence = np.clip((min_dist_both_teams - loose_ball_threshold) / LOOSE_BALL_RAMP_SPAN, 0.0, 1.0)

    # Contested duel: spatial proximity equidistance OR simultaneous dynamic approach
    is_duel_equidistant = (min_dist_both_teams <= contest_radius) & (np.abs(dist_home_effective - dist_away_effective) < contest_diff)
    is_duel_converging = (
        (dist_home_effective < CONVERGENCE_PROXIMITY_CEILING)
        & (dist_away_effective < CONVERGENCE_PROXIMITY_CEILING)
        & (closing_vel_home > RADIAL_CONVERGENCE_MIN_VELOCITY)
        & (closing_vel_away > RADIAL_CONVERGENCE_MIN_VELOCITY)
    )
    contest_confidence = np.where(is_duel_equidistant | is_duel_converging, 1.0, 0.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # 7. Candidate vs. Confirmed Turnovers
    # -------------------------------------------------------------------------
    cand_turnover_home_to_away = (dist_away_effective <= contact_radius) & (dist_away_effective < dist_home_effective) & (tracking_confidence > MIN_TRACKING_CONFIDENCE_THRESHOLD)
    cand_turnover_away_to_home = (dist_home_effective <= contact_radius) & (dist_home_effective < dist_away_effective) & (tracking_confidence > MIN_TRACKING_CONFIDENCE_THRESHOLD)

    # Confirmed turnover requires intentional action OR settled control exceeding opponent
    conf_turnover_home_to_away = (
        intentional_action_away | (
            (control_confidence_away > CONTROL_CONFIRMED_TURNOVER_FLOOR)
            & (control_confidence_away > control_confidence_home + TURNOVER_DOMINANCE_DIFFERENCE)
        )
    ) & ~is_ballistic

    conf_turnover_away_to_home = (
        intentional_action_home | (
            (control_confidence_home > CONTROL_CONFIRMED_TURNOVER_FLOOR)
            & (control_confidence_home > control_confidence_away + TURNOVER_DOMINANCE_DIFFERENCE)
        )
    ) & ~is_ballistic

    return ControlEvidence(
        control_confidence_H=control_confidence_home,
        control_confidence_A=control_confidence_away,
        intentional_action_H=intentional_action_home,
        intentional_action_A=intentional_action_away,
        contact_confidence_H=contact_confidence_home,
        contact_confidence_A=contact_confidence_away,
        is_passive_deflection_H=is_passive_deflection_home,
        is_passive_deflection_A=is_passive_deflection_away,
        loose_ball_confidence=loose_ball_confidence,
        contest_confidence=contest_confidence,
        is_deflection_event=is_deflection_event,
        deflection_team=deflection_team,
        turnover_candidate_H_to_A=cand_turnover_home_to_away,
        turnover_candidate_A_to_H=cand_turnover_away_to_home,
        turnover_confirmed_H_to_A=conf_turnover_home_to_away,
        turnover_confirmed_A_to_H=conf_turnover_away_to_home,
    )
