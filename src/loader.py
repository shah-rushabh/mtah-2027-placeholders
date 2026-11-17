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

# Broadcast ball fill (opt-in, improvement #5)
# The broadcast recording has its own timeline (cuts, replays, no mapping table), so each stretch
# of broadcast frames with a usable homography gets its own frame offset to the tactical video,
# found by matching the ball positions that both cameras see. Both ATD files use the same 0..1
# pitch coordinates, so a matched broadcast ball position goes straight into a tactical frame.
BROADCAST_MAX_REPROJECTION_ERROR = 2.0     # px: trust a broadcast frame below this (95% of valid frames are under 0.74 px)
BROADCAST_MIN_SEGMENT_FRAMES = 50          # 2.00s: shorter usable stretches are not aligned
BROADCAST_ALIGN_STRIDE = 5                 # frames: coarse offset grid, refined to 1 frame afterwards
BROADCAST_ALIGN_REFINE_FRAMES = 5          # frames: refinement half-window around the coarse offset
BROADCAST_ALIGN_MAX_SAMPLES = 150          # frames per segment used for the coarse search
BROADCAST_ALIGN_MIN_MATCHES = 15           # co-observed ball frames needed to trust an offset
BROADCAST_ALIGN_MAX_ERROR_METERS = 1.5     # median ball distance at an accepted offset
BROADCAST_ALIGN_COST_CAP_METERS = 5.0      # distance cap in the robust offset cost
BROADCAST_ALIGN_UNIQUENESS_RATIO = 0.5     # best cost must be under half the best cost >= 2 s away
BROADCAST_ALIGN_EXCLUSION_FRAMES = 50      # 2.00s: "away" for the uniqueness test
BROADCAST_FILLED_TRACKING_CONFIDENCE = 0.85  # reliability of a ball position taken from the broadcast camera
BROADCAST_LIVE_OFFSET_MIN_SEGMENTS = 3     # an offset shared by this many stretches (+-REFINE frames) is a live-feed offset


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
    ball_from_broadcast: Optional[np.ndarray] = None # shape (N,), bool: ball position taken from the broadcast tracking


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
    ball_fill: Optional[Tuple[np.ndarray, np.ndarray]] = None,
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

    ball_fill (opt-in, improvement #5): (x_norm, y_norm) arrays of length n_frames, NaN where
    there is no fill. Where the tactical ball is missing and a fill exists, the fill is used as
    an observed ball position (see broadcast_ball_fill). None keeps the submitted behaviour.
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
    ball_from_broadcast = np.zeros(n_frames, dtype=bool)
    fill_x, fill_y = ball_fill if ball_fill is not None else (None, None)

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
            if ball_token[0] == MISSING_CHAR and fill_x is not None and not math.isnan(fill_x[frame_id]):
                # repr() round-trips a float exactly, so the fill goes through the same parser.
                ball_token = repr(float(fill_x[frame_id])) + "," + repr(float(fill_y[frame_id]))
                ball_from_broadcast[frame_id] = True
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
    if ball_fill is not None:
        tracking_confidence[ball_from_broadcast] = BROADCAST_FILLED_TRACKING_CONFIDENCE

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
        ball_from_broadcast=ball_from_broadcast if ball_fill is not None else None,
    )


# ==============================================================================
# BROADCAST BALL FILL (opt-in, improvement #5)
# ==============================================================================
@dataclass
class BroadcastSegment:
    """One stretch of usable broadcast frames and its frame offset to the tactical video."""
    start: int           # first broadcast frame (inclusive)
    stop: int            # last broadcast frame + 1
    offset: Optional[int]  # tactical_frame = broadcast_frame + offset; None if not aligned
    error_m: float       # median ball distance (metres) at the offset, over co-observed frames
    matches: int         # co-observed ball frames at the offset


def read_ball_track(raw_path: str, id_base: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Ball (x, y) in 0..1 pitch units from an EPTS raw file, NaN where missing.

    Row i is frame id `id_base + i`. With id_base=None the smallest frame id in the file is
    used. The tactical file uses id_base=1 (video_frame = frame_id - 1).
    """
    ids, xs, ys = [], [], []
    with open(raw_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            token = parts[2]
            ids.append(int(parts[0]))
            if token[0] == MISSING_CHAR or token.find(",") == -1:
                xs.append(np.nan)
                ys.append(np.nan)
                continue
            c1 = token.find(",")
            c2 = token.find(",", c1 + 1)
            xs.append(float(token[:c1]))
            ys.append(float(token[c1 + 1:c2 if c2 != -1 else len(token)]))
    if not ids:
        return np.zeros(0), np.zeros(0)
    ids = np.asarray(ids)
    base = int(ids.min()) if id_base is None else id_base
    n = int(ids.max()) - base + 1
    bx = np.full(n, np.nan)
    by = np.full(n, np.nan)
    keep = ids >= base
    bx[ids[keep] - base] = np.asarray(xs)[keep]
    by[ids[keep] - base] = np.asarray(ys)[keep]
    return bx, by


def read_homography_usable(csv_path: str, n_frames: int, max_error: float = BROADCAST_MAX_REPROJECTION_ERROR) -> np.ndarray:
    """True on frames whose homography is valid and has a reprojection error <= max_error px."""
    import pandas as pd

    table = pd.read_csv(csv_path, usecols=["frame", "reprojection_error", "valid"])
    usable = np.zeros(n_frames, dtype=bool)
    good = (table["valid"].to_numpy() == 1) & (table["reprojection_error"].to_numpy() <= max_error)
    frames = table["frame"].to_numpy()
    inside = (frames >= 0) & (frames < n_frames)
    usable[frames[inside & good]] = True
    return usable


def _segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b - a >= min_len]


def _offset_costs(samples, offsets, b_x, b_y, t_x, t_y, cap):
    """Robust cost and match count of each offset: mean of min(distance, cap) over co-observed samples."""
    costs = np.full(len(offsets), np.inf)
    counts = np.zeros(len(offsets), dtype=np.int64)
    chunk = max(1, 2_000_000 // max(1, len(samples)))
    for i in range(0, len(offsets), chunk):
        off = offsets[i:i + chunk]
        idx = samples[None, :] + off[:, None]
        dx = t_x[idx] - b_x[samples][None, :]
        dy = t_y[idx] - b_y[samples][None, :]
        dist = np.minimum(np.hypot(dx, dy), cap)
        seen = ~np.isnan(dist)
        counts[i:i + chunk] = seen.sum(axis=1)
        with np.errstate(invalid="ignore"):
            costs[i:i + chunk] = np.where(seen, dist, 0.0).sum(axis=1) / np.maximum(1, counts[i:i + chunk])
    return costs, counts


def _best_offset(frames, candidates, b_x, b_y, t_x, t_y, lo_off, hi_off):
    """Best offset (refined to one frame around each candidate), its median ball error and matches."""
    best = (None, float("inf"), 0, float("inf"))
    for centre in candidates:
        fine = np.arange(max(lo_off, centre - BROADCAST_ALIGN_REFINE_FRAMES), min(hi_off, centre + BROADCAST_ALIGN_REFINE_FRAMES) + 1)
        if len(fine) == 0:
            continue
        costs, counts = _offset_costs(frames, fine, b_x, b_y, t_x, t_y, BROADCAST_ALIGN_COST_CAP_METERS)
        costs[counts < BROADCAST_ALIGN_MIN_MATCHES] = np.inf
        k = int(np.argmin(costs))
        if costs[k] < best[3]:
            offset = int(fine[k])
            dist = np.hypot(t_x[frames + offset] - b_x[frames], t_y[frames + offset] - b_y[frames])
            dist = dist[~np.isnan(dist)]
            best = (offset, float(np.median(dist)), int(len(dist)), float(costs[k]))
    return best[0], best[1], best[2]


def align_broadcast(
    t_x_norm: np.ndarray,
    t_y_norm: np.ndarray,
    b_x_norm: np.ndarray,
    b_y_norm: np.ndarray,
    b_usable: np.ndarray,
    pitch: Tuple[float, float] = (PITCH_LENGTH, PITCH_WIDTH),
) -> List[BroadcastSegment]:
    """Frame offset of each usable broadcast stretch to the tactical video.

    Pass 1, per stretch of usable broadcast frames (valid homography, low reprojection error) at
    least BROADCAST_MIN_SEGMENT_FRAMES long:
      a. Coarse search over every offset that keeps the stretch inside the tactical video, on a
         BROADCAST_ALIGN_STRIDE grid, with up to BROADCAST_ALIGN_MAX_SAMPLES broadcast ball
         frames. Cost: mean of min(ball distance, cap) over frames where the tactical camera
         also sees the ball (at least BROADCAST_ALIGN_MIN_MATCHES of them).
      b. Refine to one frame around the best coarse offset with every broadcast ball frame.
      c. Keep it if the median ball distance is <= BROADCAST_ALIGN_MAX_ERROR_METERS and the
         match is unique: best coarse cost < BROADCAST_ALIGN_UNIQUENESS_RATIO x the best cost
         more than BROADCAST_ALIGN_EXCLUSION_FRAMES away.
    Live-feed offsets: offsets kept for at least BROADCAST_LIVE_OFFSET_MIN_SEGMENTS stretches
    (within +-BROADCAST_ALIGN_REFINE_FRAMES). A live TV feed keeps one offset until time is cut
    out of the recording (for example halftime).
    Pass 2:
      - A kept offset that is not a live-feed offset is a replay. A replay shows the past, so it
        is kept only if it maps at least BROADCAST_ALIGN_EXCLUSION_FRAMES before the live-feed
        offset of the nearest live stretch.
      - A stretch with no unique match (for example a long stoppage with the ball at rest) is
        tried at each live-feed offset, and kept if the median ball distance is small enough.
    """
    length, width = pitch
    t_x = np.concatenate((np.asarray(t_x_norm, dtype=np.float64) * length, [np.nan]))
    t_y = np.concatenate((np.asarray(t_y_norm, dtype=np.float64) * width, [np.nan]))
    b_x = np.asarray(b_x_norm, dtype=np.float64) * length
    b_y = np.asarray(b_y_norm, dtype=np.float64) * width
    n_t = len(t_x) - 1
    b_seen = ~np.isnan(b_x) & np.asarray(b_usable, dtype=bool)

    # Pass 1.
    stretches = []  # (start, stop, frames, lo_off, hi_off, offset, error, matches, unique)
    for start, stop in _segments(np.asarray(b_usable, dtype=bool), BROADCAST_MIN_SEGMENT_FRAMES):
        frames = start + np.flatnonzero(b_seen[start:stop])
        entry = [start, stop, frames, 0, -1, None, float("nan"), 0, False]
        stretches.append(entry)
        if len(frames) < BROADCAST_ALIGN_MIN_MATCHES:
            continue
        lo_off, hi_off = -int(frames.min()), n_t - 1 - int(frames.max())
        entry[3], entry[4] = lo_off, hi_off
        if hi_off < lo_off:
            continue
        samples = frames[np.linspace(0, len(frames) - 1, min(len(frames), BROADCAST_ALIGN_MAX_SAMPLES)).astype(int)]
        coarse = np.arange(lo_off, hi_off + 1, BROADCAST_ALIGN_STRIDE)
        costs, counts = _offset_costs(samples, coarse, b_x, b_y, t_x, t_y, BROADCAST_ALIGN_COST_CAP_METERS)
        costs[counts < min(BROADCAST_ALIGN_MIN_MATCHES, len(samples))] = np.inf
        if not np.isfinite(costs).any():
            continue
        best = int(np.argmin(costs))
        far = np.abs(coarse - coarse[best]) > BROADCAST_ALIGN_EXCLUSION_FRAMES
        runner_up = costs[far].min() if far.any() else np.inf
        offset, error, matches = _best_offset(frames, [int(coarse[best])], b_x, b_y, t_x, t_y, lo_off, hi_off)
        if offset is None:
            continue
        entry[5], entry[6], entry[7] = offset, error, matches
        entry[8] = error <= BROADCAST_ALIGN_MAX_ERROR_METERS and costs[best] < BROADCAST_ALIGN_UNIQUENESS_RATIO * runner_up

    # Live-feed offsets: clusters of kept offsets.
    kept = sorted(e[5] for e in stretches if e[8])
    live_offsets = []
    i = 0
    while i < len(kept):
        j = i
        while j + 1 < len(kept) and kept[j + 1] - kept[i] <= 2 * BROADCAST_ALIGN_REFINE_FRAMES:
            j += 1
        if j - i + 1 >= BROADCAST_LIVE_OFFSET_MIN_SEGMENTS:
            live_offsets.append(int(np.median(kept[i:j + 1])))
        i = j + 1

    def is_live(offset):
        return any(abs(offset - live) <= BROADCAST_ALIGN_REFINE_FRAMES for live in live_offsets)

    live_stretches = [(e[0], e[5]) for e in stretches if e[8] and is_live(e[5])]

    # Pass 2.
    result: List[BroadcastSegment] = []
    for start, stop, frames, lo_off, hi_off, offset, error, matches, unique in stretches:
        accepted = None
        if unique:
            if not live_stretches or is_live(offset):
                accepted = offset
            else:
                nearest_live = min(live_stretches, key=lambda item: abs(item[0] - start))[1]
                if offset < nearest_live - BROADCAST_ALIGN_EXCLUSION_FRAMES:
                    accepted = offset  # a replay of earlier play
        elif live_offsets and len(frames) >= BROADCAST_ALIGN_MIN_MATCHES and hi_off >= lo_off:
            prior, prior_error, prior_matches = _best_offset(frames, live_offsets, b_x, b_y, t_x, t_y, lo_off, hi_off)
            if prior is not None and prior_error <= BROADCAST_ALIGN_MAX_ERROR_METERS:
                accepted, error, matches = prior, prior_error, prior_matches
        result.append(BroadcastSegment(start, stop, accepted, error, matches))
    return result


def broadcast_ball_fill(
    t_x_norm: np.ndarray,
    t_y_norm: np.ndarray,
    b_x_norm: np.ndarray,
    b_y_norm: np.ndarray,
    b_usable: np.ndarray,
    segments: List[BroadcastSegment],
) -> Tuple[np.ndarray, np.ndarray]:
    """Ball positions for tactical frames where the tactical ball is missing, NaN elsewhere.

    Only aligned segments and usable broadcast frames with a ball are used. If two segments map
    to the same tactical frame (a replay), the one with the smaller alignment error wins.
    """
    n_t = len(t_x_norm)
    fill_x = np.full(n_t, np.nan)
    fill_y = np.full(n_t, np.nan)
    best_error = np.full(n_t, np.inf)
    t_missing = np.isnan(np.asarray(t_x_norm, dtype=np.float64))
    for seg in segments:
        if seg.offset is None:
            continue
        frames = np.arange(seg.start, seg.stop)
        frames = frames[np.asarray(b_usable[seg.start:seg.stop], dtype=bool) & ~np.isnan(b_x_norm[seg.start:seg.stop])]
        target = frames + seg.offset
        ok = (target >= 0) & (target < n_t)
        frames, target = frames[ok], target[ok]
        ok = t_missing[target] & (seg.error_m < best_error[target])
        fill_x[target[ok]] = b_x_norm[frames[ok]]
        fill_y[target[ok]] = b_y_norm[frames[ok]]
        best_error[target[ok]] = seg.error_m
    return fill_x, fill_y
