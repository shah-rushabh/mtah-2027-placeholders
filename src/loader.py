"""Stage 1: Data Ingestion, Velocity Engine & Missingness Validity.

Parses EPTS tracking data (BAR-ATH_tactical_atd.txt and .xml) and Smart Tagging XML.
Calculates velocities, outfield distances (excluding goalkeepers), and manages
missingness observation validity (M_t in {0, 1}).
"""

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ==============================================================================
# MATCH CONSTANTS & PHYSICAL DIMENSIONS
# ==============================================================================
HOME_TEAM_CODE = 0  # FC Barcelona (Slot indices 0..23)
AWAY_TEAM_CODE = 1  # Athletic Club (Slot indices 24..45)

# Goalkeeper Track Slots in metadata squad list (STARTERS + BENCH)
# Slot 23 = Marc-André ter Stegen (FC Barcelona starter GK)
# Slot 45 = Álex Padilla (Athletic Club starter GK)
HOME_GK_SLOT = 23
AWAY_GK_SLOT = 45

# Certified Match Pitch Dimensions (meters), parsed from BAR-ATH_tactical_atd.xml:
# Field of play surveyed dimensions: 105.3m length by 68.0m width.
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0

# Sampling rate: EPTS optical tracking runs at 25.0 Hz (25 frames = 1.0 second).
FPS = 25.0

# Standardized Physical Time Scales (Occam's Razor Consolidation):
# 1. Neuromuscular reaction window (200 ms = 5 frames @ 25 Hz)
# Universally applied across instantaneous velocity steps, deflection continuity, and smoothing.
REACTION_WINDOW_FRAMES = 5

# 2. Ballistic micro-gap ceiling (480 ms = 12 frames @ 25 Hz)
# Maximum occlusion duration where ball flight is strictly ballistic (below human kick reaction time).
MICRO_GAP_MAX_FRAMES = 12

# Total match video timeline length across both halves and halftime break.
TOTAL_FRAMES = 199319

# Kinematic Speed Ceilings, Activity Boundaries & Percentiles
MAX_PLAUSIBLE_HUMAN_SPRINT_SPEED = 14.0   # 14.0 m/s (~50.4 km/h): Absolute human sprint ceiling (Usain Bolt limit)
ACTIVE_RUNNER_SPEED_THRESHOLD = 2.5       # 2.50 m/s (9.0 km/h, 6:40 min/km): Active tempo running vs walking/jogging
GAP_STOPPAGE_SPEED_CEILING = 1.6          # 1.60 m/s (5.76 km/h): Mean player speed below which occlusion is a stoppage
GAP_ACTIVE_SPEED_FLOOR = 2.0              # 2.00 m/s (7.20 km/h, FIFA Zone 1/2 boundary): Active open-play occlusion
PLAYER_SPEED_PERCENTILE_RATIO = 0.75      # 75th percentile: captures active outfield pacing while ignoring walking outliers
BALL_STATIONARY_NOISE_FLOOR = 2.0         # 2.00 m/s (7.2 km/h): Filters coordinate jitter where deflection angles are numerical noise
OBSERVED_TRACKING_CONFIDENCE = 1.00       # 1.00: Full reliability for sensor-detected ball coordinates
INTERPOLATED_TRACKING_CONFIDENCE = 0.70   # 0.70: Reliability assigned to ballistically bridged coordinates
UNOBSERVED_TRACKING_CONFIDENCE = 0.00     # 0.00: Zero reliability when coordinates are unobserved and unbridged

# EPTS ASCII delimiter format constants
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================
@dataclass
class MissingGap:
    """Represents a contiguous interval of missing ball observations."""
    start_frame: int
    end_frame: int
    length: int
    mean_player_speed: float
    max_player_speed: float
    category: str  # 'stoppage', 'active', 'ambiguous'
    overlaps_xml_stoppage: bool
    entry_possession: Optional[str] = None
    exit_possession: Optional[str] = None


@dataclass
class TrackingData:
    """Pre-conditioned multimodal tracking arrays for the entire match."""
    ball_x_metric: np.ndarray        # shape (N,), meters, NaN when unobserved
    ball_y_metric: np.ndarray        # shape (N,), meters, NaN when unobserved
    ball_x_norm: np.ndarray          # shape (N,), normalized 0..1
    ball_y_norm: np.ndarray          # shape (N,), normalized 0..1
    ball_observed: np.ndarray        # shape (N,), bool: True if raw sensor saw ball
    ball_velocity: np.ndarray        # shape (N,), m/s
    d_H: np.ndarray                  # shape (N,), min outfield dist Home, meters
    d_A: np.ndarray                  # shape (N,), min outfield dist Away, meters
    d_H_gk: np.ndarray               # shape (N,), dist Home GK, meters
    d_A_gk: np.ndarray               # shape (N,), dist Away GK, meters
    player_mean_speed: np.ndarray    # shape (N,), mean speed of tracked players, m/s
    missing_gaps: List[MissingGap]
    d_H_top3: Optional[np.ndarray] = None  # shape (N, 3), top 3 outfield distances Home, meters
    d_A_top3: Optional[np.ndarray] = None  # shape (N, 3), top 3 outfield distances Away, meters
    player_p75_speed: Optional[np.ndarray] = None    # shape (N,), 75th percentile player speed, m/s
    player_active_frac: Optional[np.ndarray] = None  # shape (N,), fraction of players moving > 2.5 m/s
    ball_vx: Optional[np.ndarray] = None             # shape (N,), m/s X component
    ball_vy: Optional[np.ndarray] = None             # shape (N,), m/s Y component
    ball_deflection_angle: Optional[np.ndarray] = None  # shape (N,), angular shift in degrees
    ball_valid: Optional[np.ndarray] = None          # shape (N,), bool: True if raw or bridged coords exist
    ball_interpolated: Optional[np.ndarray] = None   # shape (N,), bool: True if filled by micro-gap interpolation
    ball_missing: Optional[np.ndarray] = None        # shape (N,), bool: True if unobserved and not interpolated
    tracking_confidence: Optional[np.ndarray] = None # shape (N,), float 0.0..1.0 reliability indicator
    gk_slots: Optional[Dict[str, int]] = None        # {'H': slot_idx, 'A': slot_idx}


# ==============================================================================
# METADATA & SMART TAGGING PARSERS
# ==============================================================================
def parse_metadata_roles(meta_path: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Parses player track slots and goalkeeper positions from EPTS metadata XML.
    
    Returns:
        team_of_track: np.ndarray of shape (n_slots,) containing 0 for Home, 1 for Away.
        gk_slots: Dict with 'H' and 'A' mapping to their respective goalkeeper slot index.
    """
    metadata_tree_root = ET.parse(meta_path).getroot()
    team_elements = [team for team in metadata_tree_root.iter("Team")]
    if len(team_elements) != 2:
        raise ValueError(f"{meta_path}: expected 2 teams, found {len(team_elements)}")
    home_team_id = team_elements[0].get("id")
    away_team_id = team_elements[1].get("id")

    track_slot_teams = []
    gk_slots = {}

    # Map first goalkeeper found for each team (starter).
    # Assumption Note: Assumes standard match without in-game goalkeeper substitutions
    # (empirically verified: Slot 23 and Slot 45 are active 100% of match frames).
    # If a goalkeeper substitution occurred, gk_slots could be expanded to a set or
    # time-varying index mapped to the substitute keeper's slot.
    for slot_index, player in enumerate(metadata_tree_root.iter("Player")):
        player_team_id = player.get("teamId")
        team_code = HOME_TEAM_CODE if player_team_id == home_team_id else AWAY_TEAM_CODE
        track_slot_teams.append(team_code)

        # Check for goalkeeper position type in ProviderParameter metadata
        is_goalkeeper = False
        for param in player.iter("ProviderParameter"):
            param_name = param.findtext("Name", "")
            param_value = param.findtext("Value", "")
            if param_name == "position_type" and "goalkeeper" in param_value.lower():
                is_goalkeeper = True
                break

        if is_goalkeeper:
            if team_code == HOME_TEAM_CODE and "H" not in gk_slots:
                gk_slots["H"] = slot_index
            elif team_code == AWAY_TEAM_CODE and "A" not in gk_slots:
                gk_slots["A"] = slot_index

    # Fallback to defaults if metadata position_type is missing
    if "H" not in gk_slots:
        gk_slots["H"] = HOME_GK_SLOT
    if "A" not in gk_slots:
        gk_slots["A"] = AWAY_GK_SLOT

    return np.array(track_slot_teams, dtype=np.int8), gk_slots


def parse_team_slots(meta_path: str) -> np.ndarray:
    """Parses player track slots from EPTS metadata XML.
    
    Returns an array over 46 track slots: 0 for Home, 1 for Away.
    """
    team_of_track, _ = parse_metadata_roles(meta_path)
    return team_of_track


def parse_smart_tagging(xml_path: str, fps: float = FPS) -> Dict:
    """Parses Smart Tagging XML for set pieces, goals, and tactical phases.
    
    Corrects XML spelling ('Atletic Club' without 'h') to map both teams reliably.
    """
    smart_tag_root = ET.parse(xml_path).getroot()
    set_pieces = []
    goals = []
    active_phases = []

    possession_phase_tags = {
        "PROGRESSION", "LONG BALL", "ATTACKING TRANSITION", "BUILD UP",
        "CREATING CHANCES", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"
    }
    defending_phase_tags = {
        "DEFENSIVE TRANSITION", "DEFENDING IN ATTACKING THIRD",
        "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD"
    }

    for instance in smart_tag_root.iter("instance"):
        instance_code = instance.findtext("code", "").strip().upper()
        start_seconds = float(instance.findtext("start", "0.0"))
        end_seconds = float(instance.findtext("end", "0.0"))
        start_frame = int(round(start_seconds * fps))
        end_frame = int(round(end_seconds * fps))

        instance_labels = {label.findtext("group", ""): label.findtext("text", "") for label in instance.findall("label")}
        tagged_team_name = instance_labels.get("Team", "")

        # Robust team normalization matching XML 'Atletic Club' and 'FC Barcelona'
        assigned_team_side = None
        if "BARCELONA" in tagged_team_name.upper():
            assigned_team_side = "H"
        elif "ATLETIC" in tagged_team_name.upper() or "ATHLETIC" in tagged_team_name.upper():
            assigned_team_side = "A"

        if instance_code == "SET PIECES":
            set_pieces.append({
                "start": start_frame,
                "end": end_frame,
                "type": instance_labels.get("Type", ""),
                "team": tagged_team_name,
                "team_code": assigned_team_side,
            })
        elif instance_code == "GOALS":
            goals.append({
                "start": start_frame,
                "end": end_frame,
                "team": tagged_team_name,
                "team_code": assigned_team_side,
            })
        elif instance_code in possession_phase_tags and assigned_team_side:
            active_phases.append({
                "start": start_frame,
                "end": end_frame,
                "code": instance_code,
                "team": assigned_team_side,
                "phase_type": "attacking",
            })
        elif instance_code in defending_phase_tags and assigned_team_side:
            # Defending team means opponent has possession
            opponent_side = "A" if assigned_team_side == "H" else "H"
            active_phases.append({
                "start": start_frame,
                "end": end_frame,
                "code": instance_code,
                "team": opponent_side,
                "phase_type": "defending",
            })

    # Sort events chronologically
    set_pieces.sort(key=lambda x: x["start"])
    goals.sort(key=lambda x: x["start"])
    active_phases.sort(key=lambda x: x["start"])

    return {
        "set_pieces": set_pieces,
        "goals": goals,
        "active_phases": active_phases,
    }


# ==============================================================================
# MAIN TRACKING INGESTION & CONDITIONING ENGINE
# ==============================================================================
def load_and_condition_tracking(
    raw_path: str,
    meta_path: str,
    smart_tagging_path: Optional[str] = None,
    n_frames: int = TOTAL_FRAMES,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
    micro_gap_max: int = MICRO_GAP_MAX_FRAMES,
    trajectory_strategy: str = "neutral_gaps",  # 'neutral_gaps' or 'active_bridging'
) -> TrackingData:
    """Streams EPTS raw data, scales coordinates, detects missing gaps, and computes kinematics.
    
    Biomechanical & Kinematic Parameter Justifications:
    ---------------------------------------------------
    - micro_gap_max = 12 frames: 12 frames @ 25 fps = 0.48 seconds. Across < 0.5s,
      ball flight is ballistic without unobserved bounces; linear interpolation is robust,
      fast, and mathematically exact across micro-steps.
    - spd < 14.0 m/s: Usain Bolt's peak sprint speed is ~12.4 m/s (44.6 km/h). 14.0 m/s
      (50.4 km/h) serves as an absolute human sprint ceiling to filter optical coordinate glitches.
    - s > 2.5 m/s: Standard sports science threshold (9.0 km/h) separating walking/jogging
      from active running.
    - mean_spd < 1.6 m/s: Typical human walking speed (5.7 km/h). When average player speed
      in a gap is < 1.6 m/s, players have stopped running, signifying a referee stoppage.
    - mean_spd >= 2.0 m/s: Active running speed (7.2 km/h), denoting open-play camera occlusions.
    """
    length, width = pitch
    team_of_track, gk_slots = parse_metadata_roles(meta_path)
    n_slots = len(team_of_track)
    home_gk_slot = gk_slots.get("H", HOME_GK_SLOT)
    away_gk_slot = gk_slots.get("A", AWAY_GK_SLOT)

    smart_tags = parse_smart_tagging(smart_tagging_path) if smart_tagging_path and os.path.exists(smart_tagging_path) else None

    # Pre-allocate output arrays
    ball_x_metric = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_metric = np.full(n_frames, np.nan, dtype=np.float32)
    ball_x_norm = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_norm = np.full(n_frames, np.nan, dtype=np.float32)
    ball_observed = np.zeros(n_frames, dtype=bool)
    ball_interpolated = np.zeros(n_frames, dtype=bool)

    d_H = np.full(n_frames, np.nan, dtype=np.float32)
    d_A = np.full(n_frames, np.nan, dtype=np.float32)
    d_H_top3 = np.full((n_frames, 3), np.nan, dtype=np.float32)
    d_A_top3 = np.full((n_frames, 3), np.nan, dtype=np.float32)
    d_H_gk = np.full(n_frames, np.nan, dtype=np.float32)
    d_A_gk = np.full(n_frames, np.nan, dtype=np.float32)

    player_mean_speed = np.zeros(n_frames, dtype=np.float32)
    player_p75_speed = np.zeros(n_frames, dtype=np.float32)
    player_active_frac = np.zeros(n_frames, dtype=np.float32)

    # To compute player speeds frame-to-frame with elapsed dt
    prev_player_coords = np.full((n_slots, 2), np.nan, dtype=np.float32)
    prev_player_frames = np.full(n_slots, -1, dtype=np.int32)

    # Stream line by line (O(1) memory footprint for raw parsing)
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0]) - 1  # 0-indexed video frame
            if frame_id < 0 or frame_id >= n_frames:
                continue

            # Parse player track slots
            curr_player_coords = np.full((n_slots, 2), np.nan, dtype=np.float32)
            player_deltas = []

            for slot_idx, group in enumerate(parts[1].split(GROUP_SEP)):
                if slot_idx >= n_slots:
                    break
                if group[0] == MISSING_CHAR:
                    continue
                first_comma_index = group.find(",")
                if first_comma_index == -1:
                    continue
                second_comma_index = group.find(",", first_comma_index + 1)
                player_x_metric = float(group[:first_comma_index]) * length
                player_y_metric = float(group[first_comma_index + 1:second_comma_index if second_comma_index != -1 else len(group)]) * width
                curr_player_coords[slot_idx, 0] = player_x_metric
                curr_player_coords[slot_idx, 1] = player_y_metric

                # Compute player speed using elapsed delta time
                if prev_player_frames[slot_idx] >= 0 and not np.isnan(prev_player_coords[slot_idx, 0]):
                    step_frames = frame_id - prev_player_frames[slot_idx]
                    if 0 < step_frames <= REACTION_WINDOW_FRAMES:  # <= 0.20s neuromuscular window for valid velocity
                        elapsed_time_seconds = step_frames / FPS
                        delta_x_meters = player_x_metric - prev_player_coords[slot_idx, 0]
                        delta_y_meters = player_y_metric - prev_player_coords[slot_idx, 1]
                        player_speed = math.sqrt(delta_x_meters * delta_x_meters + delta_y_meters * delta_y_meters) / elapsed_time_seconds
                        if player_speed < MAX_PLAUSIBLE_HUMAN_SPRINT_SPEED:  # Physically plausible human sprint bound (< 14 m/s)
                            player_deltas.append(player_speed)

                prev_player_coords[slot_idx, 0] = player_x_metric
                prev_player_coords[slot_idx, 1] = player_y_metric
                prev_player_frames[slot_idx] = frame_id

            if player_deltas:
                player_mean_speed[frame_id] = float(np.mean(player_deltas))
                player_deltas.sort()
                percentile_75_index = int(PLAYER_SPEED_PERCENTILE_RATIO * (len(player_deltas) - 1))
                player_p75_speed[frame_id] = float(player_deltas[percentile_75_index])
                active_runner_count = sum(1 for s in player_deltas if s > ACTIVE_RUNNER_SPEED_THRESHOLD)  # Active runners (> 2.5 m/s)
                player_active_frac[frame_id] = float(active_runner_count / len(player_deltas))
            elif frame_id > 0:
                player_mean_speed[frame_id] = player_mean_speed[frame_id - 1]
                player_p75_speed[frame_id] = player_p75_speed[frame_id - 1]
                player_active_frac[frame_id] = player_active_frac[frame_id - 1]

            # Parse ball coordinate
            ball_token = parts[2]
            if ball_token[0] != MISSING_CHAR:
                ball_comma_idx_1 = ball_token.find(",")
                if ball_comma_idx_1 != -1:
                    ball_comma_idx_2 = ball_token.find(",", ball_comma_idx_1 + 1)
                    ball_x_normalized = float(ball_token[:ball_comma_idx_1])
                    ball_y_normalized = float(ball_token[ball_comma_idx_1 + 1:ball_comma_idx_2 if ball_comma_idx_2 != -1 else len(ball_token)])
                    ball_x_metric_val = ball_x_normalized * length
                    ball_y_metric_val = ball_y_normalized * width

                    ball_x_norm[frame_id] = ball_x_normalized
                    ball_y_norm[frame_id] = ball_y_normalized
                    ball_x_metric[frame_id] = ball_x_metric_val
                    ball_y_metric[frame_id] = ball_y_metric_val
                    ball_observed[frame_id] = True

                    # Compute outfield distances (excluding GKs to avoid set-piece distortion)
                    home_player_distances = []
                    away_player_distances = []

                    for track_slot in range(n_slots):
                        player_x_pos = curr_player_coords[track_slot, 0]
                        player_y_pos = curr_player_coords[track_slot, 1]
                        if np.isnan(player_x_pos):
                            continue
                        distance_to_ball = math.sqrt((player_x_pos - ball_x_metric_val) ** 2 + (player_y_pos - ball_y_metric_val) ** 2)

                        if track_slot == home_gk_slot:
                            d_H_gk[frame_id] = distance_to_ball
                        elif track_slot == away_gk_slot:
                            d_A_gk[frame_id] = distance_to_ball
                        elif team_of_track[track_slot] == HOME_TEAM_CODE:
                            home_player_distances.append(distance_to_ball)
                        elif team_of_track[track_slot] == AWAY_TEAM_CODE:
                            away_player_distances.append(distance_to_ball)

                    if home_player_distances:
                        home_player_distances.sort()
                        d_H[frame_id] = home_player_distances[0]
                        for rank_idx in range(min(3, len(home_player_distances))):
                            d_H_top3[frame_id, rank_idx] = home_player_distances[rank_idx]
                    if away_player_distances:
                        away_player_distances.sort()
                        d_A[frame_id] = away_player_distances[0]
                        for rank_idx in range(min(3, len(away_player_distances))):
                            d_A_top3[frame_id, rank_idx] = away_player_distances[rank_idx]

    # Zero-phase rolling filter (0.20s window = REACTION_WINDOW_FRAMES) for player kinematics
    kernel = np.ones(REACTION_WINDOW_FRAMES, dtype=np.float32) / float(REACTION_WINDOW_FRAMES)
    player_p75_speed = np.convolve(player_p75_speed, kernel, mode="same")
    player_active_frac = np.convolve(player_active_frac, kernel, mode="same")

    # Detect all contiguous missing gaps
    missing_mask = ~ball_observed
    diffs = np.diff(np.concatenate(([0], missing_mask.astype(np.int8), [0])))
    gap_starts = np.flatnonzero(diffs == 1)
    gap_ends = np.flatnonzero(diffs == -1) - 1

    missing_gaps: List[MissingGap] = []

    # Process gaps
    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start + 1

        # Check if gap overlaps any Smart Tagging stoppage
        overlaps_stoppage = False
        if smart_tags:
            for sp in smart_tags["set_pieces"]:
                if max(start, sp["start"]) <= min(end, sp["end"]):
                    overlaps_stoppage = True
                    break
            if not overlaps_stoppage:
                for goal in smart_tags["goals"]:
                    # XML goal instances already span from goal-line crossing until restart kickoff
                    if max(start, goal["start"]) <= min(end, goal["end"]):
                        overlaps_stoppage = True
                        break

        # Calculate player kinematic stats during the gap
        speeds_in_gap = player_mean_speed[start:end + 1]
        mean_player_speed_in_gap = float(np.mean(speeds_in_gap)) if len(speeds_in_gap) > 0 else 0.0
        max_player_speed_in_gap = float(np.max(speeds_in_gap)) if len(speeds_in_gap) > 0 else 0.0

        # Classify gap
        if gap_len <= micro_gap_max:
            category = "micro"
            # Micro-gap: interpolate linearly if valid bounding endpoints exist (<= 12 frames = 0.48s)
            if start > 0 and end < n_frames - 1:
                prev_valid_ball_x = ball_x_metric[start - 1]
                next_valid_ball_x = ball_x_metric[end + 1]
                prev_valid_ball_y = ball_y_metric[start - 1]
                next_valid_ball_y = ball_y_metric[end + 1]
                if not np.isnan(prev_valid_ball_x) and not np.isnan(next_valid_ball_x):
                    interpolation_ratios = np.linspace(0, 1, gap_len + 2)[1:-1]
                    ball_x_metric[start:end + 1] = prev_valid_ball_x + interpolation_ratios * (next_valid_ball_x - prev_valid_ball_x)
                    ball_y_metric[start:end + 1] = prev_valid_ball_y + interpolation_ratios * (next_valid_ball_y - prev_valid_ball_y)
                    ball_x_norm[start:end + 1] = ball_x_metric[start:end + 1] / length
                    ball_y_norm[start:end + 1] = ball_y_metric[start:end + 1] / width
                    ball_interpolated[start:end + 1] = True
                    # Compute linearly interpolated distances (outfield min, top3, and GK)
                    if not np.isnan(d_H[start - 1]) and not np.isnan(d_H[end + 1]):
                        d_H[start:end + 1] = np.linspace(d_H[start - 1], d_H[end + 1], gap_len + 2)[1:-1]
                    if not np.isnan(d_A[start - 1]) and not np.isnan(d_A[end + 1]):
                        d_A[start:end + 1] = np.linspace(d_A[start - 1], d_A[end + 1], gap_len + 2)[1:-1]
                    for rank_idx in range(3):
                        if not np.isnan(d_H_top3[start - 1, rank_idx]) and not np.isnan(d_H_top3[end + 1, rank_idx]):
                            d_H_top3[start:end + 1, rank_idx] = np.linspace(d_H_top3[start - 1, rank_idx], d_H_top3[end + 1, rank_idx], gap_len + 2)[1:-1]
                        if not np.isnan(d_A_top3[start - 1, rank_idx]) and not np.isnan(d_A_top3[end + 1, rank_idx]):
                            d_A_top3[start:end + 1, rank_idx] = np.linspace(d_A_top3[start - 1, rank_idx], d_A_top3[end + 1, rank_idx], gap_len + 2)[1:-1]
                    if not np.isnan(d_H_gk[start - 1]) and not np.isnan(d_H_gk[end + 1]):
                        d_H_gk[start:end + 1] = np.linspace(d_H_gk[start - 1], d_H_gk[end + 1], gap_len + 2)[1:-1]
                    if not np.isnan(d_A_gk[start - 1]) and not np.isnan(d_A_gk[end + 1]):
                        d_A_gk[start:end + 1] = np.linspace(d_A_gk[start - 1], d_A_gk[end + 1], gap_len + 2)[1:-1]
        elif overlaps_stoppage or mean_player_speed_in_gap < GAP_STOPPAGE_SPEED_CEILING:
            category = "stoppage"
        elif mean_player_speed_in_gap >= GAP_ACTIVE_SPEED_FLOOR:
            category = "active"
            # Conservative interpolation policy: active gaps exceeding micro_gap_max (12 frames = 0.48s)
            # are NOT linearly bridged, preventing artificial straight-line flights across 10 seconds.
            # Sequence inertia in Viterbi and player movement naturally carry state through occlusions.
        else:
            category = "ambiguous"

        missing_gaps.append(MissingGap(
            start_frame=int(start),
            end_frame=int(end),
            length=int(gap_len),
            mean_player_speed=mean_player_speed_in_gap,
            max_player_speed=max_player_speed_in_gap,
            category=category,
            overlaps_xml_stoppage=overlaps_stoppage,
        ))

    # Compute continuous 2D ball kinematics AFTER bridging gaps to ensure no 0.0 m/s velocity holes
    ball_valid = ~np.isnan(ball_x_metric)
    ball_velocity = np.zeros(n_frames, dtype=np.float32)
    ball_vx = np.zeros(n_frames, dtype=np.float32)
    ball_vy = np.zeros(n_frames, dtype=np.float32)
    ball_deflection_angle = np.zeros(n_frames, dtype=np.float32)

    valid_ball_indices = np.flatnonzero(ball_valid)
    if len(valid_ball_indices) > 1:
        step_target_indices = valid_ball_indices[1:]
        step_source_indices = valid_ball_indices[:-1]
        step_frame_intervals = step_target_indices - step_source_indices

        # Compute velocity across valid micro-steps (<= REACTION_WINDOW_FRAMES = 0.20s)
        valid_micro_step_mask = step_frame_intervals <= REACTION_WINDOW_FRAMES
        target_frames = step_target_indices[valid_micro_step_mask]
        source_frames = step_source_indices[valid_micro_step_mask]
        
        delta_x_meters = ball_x_metric[target_frames] - ball_x_metric[source_frames]
        delta_y_meters = ball_y_metric[target_frames] - ball_y_metric[source_frames]
        delta_time_seconds = step_frame_intervals[valid_micro_step_mask] / FPS
        ball_vx_instantaneous = delta_x_meters / delta_time_seconds
        ball_vy_instantaneous = delta_y_meters / delta_time_seconds
        ball_speeds_instantaneous = np.sqrt(ball_vx_instantaneous * ball_vx_instantaneous + ball_vy_instantaneous * ball_vy_instantaneous)

        ball_vx[target_frames] = ball_vx_instantaneous
        ball_vy[target_frames] = ball_vy_instantaneous
        ball_velocity[target_frames] = ball_speeds_instantaneous

        # Calculate angular directional deflections on frames with meaningful speed (> 2.0 m/s = 7.2 km/h)
        # Filters out stationary ball coordinate dithering where angles are numerical noise.
        active_speed_indices = np.flatnonzero(ball_velocity > BALL_STATIONARY_NOISE_FLOOR)
        if len(active_speed_indices) > 1:
            curr_deflection_indices = active_speed_indices[1:]
            prev_deflection_indices = active_speed_indices[:-1]
            consecutive_deflection_mask = (curr_deflection_indices - prev_deflection_indices) <= REACTION_WINDOW_FRAMES
            active_curr_indices = curr_deflection_indices[consecutive_deflection_mask]
            active_prev_indices = prev_deflection_indices[consecutive_deflection_mask]

            velocity_dot_products = (
                ball_vx[active_curr_indices] * ball_vx[active_prev_indices] +
                ball_vy[active_curr_indices] * ball_vy[active_prev_indices]
            )
            velocity_norm_products = ball_velocity[active_curr_indices] * ball_velocity[active_prev_indices]
            cosine_deflection_angles = np.clip(velocity_dot_products / (velocity_norm_products + 1e-6), -1.0, 1.0)
            ball_deflection_angle[active_curr_indices] = np.degrees(np.arccos(cosine_deflection_angles))

    # Explicit observation classification & tracking quality indicator
    ball_missing = ~ball_observed & ~ball_interpolated
    tracking_confidence = np.zeros(n_frames, dtype=np.float32)
    tracking_confidence[ball_observed] = OBSERVED_TRACKING_CONFIDENCE
    tracking_confidence[ball_interpolated] = INTERPOLATED_TRACKING_CONFIDENCE

    return TrackingData(
        ball_x_metric=ball_x_metric,
        ball_y_metric=ball_y_metric,
        ball_x_norm=ball_x_norm,
        ball_y_norm=ball_y_norm,
        ball_observed=ball_observed,
        ball_velocity=ball_velocity,
        d_H=d_H,
        d_A=d_A,
        d_H_top3=d_H_top3,
        d_A_top3=d_A_top3,
        d_H_gk=d_H_gk,
        d_A_gk=d_A_gk,
        player_mean_speed=player_mean_speed,
        player_p75_speed=player_p75_speed,
        player_active_frac=player_active_frac,
        missing_gaps=missing_gaps,
        ball_vx=ball_vx,
        ball_vy=ball_vy,
        ball_deflection_angle=ball_deflection_angle,
        ball_valid=ball_valid,
        ball_interpolated=ball_interpolated,
        ball_missing=ball_missing,
        tracking_confidence=tracking_confidence,
        gk_slots={"H": home_gk_slot, "A": away_gk_slot},
    )
