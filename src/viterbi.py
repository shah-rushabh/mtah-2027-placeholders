"""Stage 5: Context-Aware Log-Viterbi HMM Decoder.

Implements optimal sequence decoding over the match timeline (199,319 frames @ 25 Hz)
using dynamic programming MAP inference in log-probability space:
    v_t(j) = max_i [ v_{t-1}(i) + psi_t(i, j) ] + ln B_j(t)
    backpointers[t, j] = argmax_i [ v_{t-1}(i) + psi_t(i, j) ]

Supports both the primary 3-state competition space {H, A, D} and the augmented
5-state physics space {H_ctrl, H_unctrl, A_ctrl, A_unctrl, Dead} with deterministic
output mapping to {H, A, D}.

All transition parameters are strictly derived from IFAB Laws of the Game,
empirical sports science dwell-time distributions (25 Hz frame clock),
and confirmed turnover physical inception gating.
"""

from typing import Optional, Tuple
import numpy as np

from .episodes import EpisodeEvidence
from .control import ControlEvidence

# =============================================================================
# 1. HMM STATE IDENTIFIERS & NUMERICAL SAFEGUARDS
# =============================================================================
# Primary 3-State Competition Alphabet:
STATE_H = 0  # Home Possession (FC Barcelona)
STATE_A = 1  # Away Possession (Athletic Club)
STATE_D = 2  # Dead Ball / Stoppage (Out of bounds, foul, set-piece setup)

# Augmented 5-State Physical Model:
STATE_5_H_CTRL = 0    # Home has active controlled possession (dribble, carry, settled control)
STATE_5_H_UNCTRL = 1  # Home held prior control; ball is in pass, flight, or loose/deflected
STATE_5_A_CTRL = 2    # Away has active controlled possession (dribble, carry, settled control)
STATE_5_A_UNCTRL = 3  # Away held prior control; ball is in pass, flight, or loose/deflected
STATE_5_DEAD = 4      # Match stoppage / Dead Ball

# Deterministic projection from 5 internal physical states to 3 competition labels:
MAP_5_TO_3 = np.array([STATE_H, STATE_H, STATE_A, STATE_A, STATE_D], dtype=np.int8)

# Strict penalty preventing physically impossible transitions:
FORBIDDEN_TRANSITION_PENALTY = -1000.0

# =============================================================================
# 2. MATCH CONTEXT PENALTIES & BONUSES (Dynamic Transition Adjustments)
# =============================================================================
# Premature stoppage exit penalty:
# When play is stopped (p_dead > 0.70), casual setup touches receive an exit penalty of -8.0
# to prevent leaving the Dead Ball state before the legal restart kick is actually taken.
STOPPAGE_CONFIRMED_DEAD_PROBABILITY = 0.70
PREMATURE_STOPPAGE_EXIT_PENALTY = -8.0

# Confirmed restart transition penalty:
# Once a legal restart kick is detected (p_restart > 0.50), leaving Dead Ball has virtually zero penalty (-0.1):
RESTART_ACTIVE_PROBABILITY_THRESHOLD = 0.50
CONFIRMED_RESTART_TRANSITION_PENALTY = -0.1

# Unconfirmed turnover penalty:
# An opponent touch without sustained control or an intentional pass/clearance receives a -6.0 penalty:
UNCONFIRMED_TURNOVER_PENALTY = -6.0

# =============================================================================
# 3. 5-STATE ROW-STOCHASTIC BASE TRANSITION MATRIX
# =============================================================================
# Every row sums to exactly 1.000000.
# Transitions T_{0,3}, T_{1,3}, T_{2,1}, T_{3,1}, T_{4,1}, T_{4,3} are strictly 0.0 (IFAB laws).
# Expected dwell times at 25 Hz:
# - H_ctrl / A_ctrl (p = 0.9850): 1 / (25 * 0.0150) = 2.67s (elite dribble/carry duration).
# - H_unctrl / A_unctrl (p = 0.9570): 1 / (25 * 0.0430) = 0.93s (pass/cross flight duration).
# - Dead (p = 0.9960): 1 / (25 * 0.0040) = 10.0s (set-piece setup delay duration).
DEFAULT_5STATE_TRANSITION_MATRIX = np.array([
    # H_ctrl  H_unctrl  A_ctrl   A_unctrl  Dead
    [0.9850,  0.0120,   0.0008,  0.0000,   0.0022],  # From H_ctrl   (sum = 1.000000)
    [0.0380,  0.9570,   0.0030,  0.0000,   0.0020],  # From H_unctrl (sum = 1.000000)
    [0.0008,  0.0000,   0.9850,  0.0120,   0.0022],  # From A_ctrl   (sum = 1.000000)
    [0.0030,  0.0000,   0.0380,  0.9570,   0.0020],  # From A_unctrl (sum = 1.000000)
    [0.0020,  0.0000,   0.0020,  0.0000,   0.9960],  # From Dead     (sum = 1.000000)
], dtype=np.float64)

assert np.allclose(DEFAULT_5STATE_TRANSITION_MATRIX.sum(axis=1), 1.0), "5-state transition matrix must be strictly row-stochastic"

DEFAULT_5STATE_INITIAL_PROBS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)

# =============================================================================
# 4. 3-STATE BASELINE TRANSITION MATRICES (Rank 1 Engine)
# =============================================================================
# Default symmetric high-persistence 3x3 matrix:
# Dwell times: Live possession p = 0.9980 -> 20.0s expected spell duration.
DEFAULT_TRANSITION_MATRIX = np.array([
    [0.9980, 0.0010, 0.0010],  # From H (sum = 1.0000)
    [0.0010, 0.9980, 0.0010],  # From A (sum = 1.0000)
    [0.0020, 0.0020, 0.9960],  # From D (sum = 1.0000)
], dtype=np.float64)

DEFAULT_INITIAL_PROBS = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float64)


def get_transition_matrix(strategy: str = "asymmetric_turnover") -> np.ndarray:
    """Returns row-stochastic 3x3 transition matrix based on requested strategy."""
    if strategy == "high_persistence":
        return DEFAULT_TRANSITION_MATRIX.copy()
    elif strategy == "natural_dwell":
        return np.array([
            [0.9950, 0.0025, 0.0025],
            [0.0025, 0.9950, 0.0025],
            [0.0040, 0.0040, 0.9920],
        ], dtype=np.float64)
    elif strategy == "asymmetric_turnover":
        return np.array([
            [0.9980, 0.0008, 0.0012],
            [0.0008, 0.9980, 0.0012],
            [0.0020, 0.0020, 0.9960],
        ], dtype=np.float64)
    else:
        raise ValueError(f"Unknown transition strategy: {strategy}")


def viterbi_decode(
    log_emissions: np.ndarray,
    transition_matrix: Optional[np.ndarray] = None,
    transition_strategy: Optional[str] = "asymmetric_turnover",
    initial_probs: Optional[np.ndarray] = None,
    episode_evidence: Optional[EpisodeEvidence] = None,
    is_deflection: Optional[np.ndarray] = None,
    control_evidence: Optional[ControlEvidence] = None,
) -> np.ndarray:
    """Decodes globally optimal state sequence using dynamic programming MAP inference.

    Mathematical Formalism (Linear-Chain CRF Trellis):
    --------------------------------------------------
    The base transition matrix A defines the Markov prior. During sequence decoding,
    local kinematic episode and control evidence dynamically modulates transition costs psi_t(i, j):
        S* = argmax_S sum_{t=1}^T [ psi_t(S_{t-1}, S_t) + ln B(t, S_t) ]

    Recursion (Forward Bellman Update):
        v_t(j) = max_i [ v_{t-1}(i) + psi_t(i, j) ] + ln B_j(t)
        backpointers[t, j] = argmax_i [ v_{t-1}(i) + psi_t(i, j) ]

    Backtracking Pass:
        Walks backward along backpointers from argmax(v_T) to extract optimal path S*.

    Args:
        log_emissions: np.ndarray of shape (n_frames, 3) or (n_frames, 5) containing ln B(t, k).
        transition_matrix: Optional explicit transition matrix of shape (3, 3) or (5, 5).
        transition_strategy: Pre-configured 3x3 strategy ('asymmetric_turnover', 'high_persistence').
        initial_probs: Optional prior state distribution vector of shape (3,) or (5,).
        episode_evidence: Optional macro game phase kinematic evidence from Stage 2.
        is_deflection: Optional boolean mask of deflection events.
        control_evidence: Optional micro interaction and turnover evidence from Stage 3.

    Returns:
        optimal_state_sequence: np.ndarray of shape (n_frames,) containing decoded labels in {0, 1, 2}.
    """
    total_frames, n_states = log_emissions.shape
    assert n_states in (3, 5), f"Expected 3 or 5 states, got {n_states}"

    # Select default transition matrix based on state dimension:
    if transition_matrix is None:
        if n_states == 5:
            transition_matrix = DEFAULT_5STATE_TRANSITION_MATRIX
        elif transition_strategy is not None:
            transition_matrix = get_transition_matrix(transition_strategy)
        else:
            transition_matrix = DEFAULT_TRANSITION_MATRIX

    if initial_probs is None:
        initial_probs = DEFAULT_5STATE_INITIAL_PROBS if n_states == 5 else DEFAULT_INITIAL_PROBS

    # Convert transition and prior probabilities to log-space with strict -1000.0 barrier for zeros:
    with np.errstate(divide="ignore"):
        base_log_A = np.where(transition_matrix > 0, np.log(transition_matrix), FORBIDDEN_TRANSITION_PENALTY)
        log_initial_prior = np.where(initial_probs > 0, np.log(initial_probs), FORBIDDEN_TRANSITION_PENALTY)

    # Pre-allocate backpointer table and initialize forward trellis accumulator:
    optimal_predecessors = np.zeros((total_frames, n_states), dtype=np.int8)
    running_best_scores = log_initial_prior + log_emissions[0]

    # Unpack contextual evidence masks:
    has_episodes = episode_evidence is not None
    prob_dead = episode_evidence.p_dead if has_episodes else None
    prob_restart = episode_evidence.p_restart_candidate if has_episodes else None
    restart_team = episode_evidence.restart_team if has_episodes else None
    has_control = control_evidence is not None

    # =========================================================================
    # FORWARD DYNAMIC PROGRAMMING PASS (Bellman Recursion)
    # =========================================================================
    for frame_idx in range(1, total_frames):
        current_transition_log_matrix = base_log_A
        matrix_needs_copy = False

        if n_states == 5:
            # -----------------------------------------------------------------
            # 5-State Dynamic Trellis Modulations
            # -----------------------------------------------------------------
            # 1. Stoppage Inertia Barrier (bar casual touches from exiting Dead Ball):
            if (
                has_episodes
                and prob_dead is not None
                and prob_dead[frame_idx] > STOPPAGE_CONFIRMED_DEAD_PROBABILITY
                and (prob_restart is None or prob_restart[frame_idx] < RESTART_ACTIVE_PROBABILITY_THRESHOLD)
            ):
                if not matrix_needs_copy:
                    current_transition_log_matrix = base_log_A.copy()
                    matrix_needs_copy = True
                current_transition_log_matrix[STATE_5_DEAD, STATE_5_H_CTRL] += PREMATURE_STOPPAGE_EXIT_PENALTY
                current_transition_log_matrix[STATE_5_DEAD, STATE_5_A_CTRL] += PREMATURE_STOPPAGE_EXIT_PENALTY

            # 2. Restart Snap Potential (free transition on confirmed restart kicks):
            if (
                has_episodes
                and prob_restart is not None
                and prob_restart[frame_idx] > RESTART_ACTIVE_PROBABILITY_THRESHOLD
                and restart_team is not None
            ):
                kicking_team = restart_team[frame_idx]
                if kicking_team in (0, 1):
                    if not matrix_needs_copy:
                        current_transition_log_matrix = base_log_A.copy()
                        matrix_needs_copy = True
                    target_ctrl_state = STATE_5_H_CTRL if kicking_team == 0 else STATE_5_A_CTRL
                    current_transition_log_matrix[STATE_5_DEAD, target_ctrl_state] = CONFIRMED_RESTART_TRANSITION_PENALTY

            # 3. Confirmed Turnover Gating:
            if has_control:
                if not control_evidence.turnover_confirmed_H_to_A[frame_idx]:
                    if not matrix_needs_copy:
                        current_transition_log_matrix = base_log_A.copy()
                        matrix_needs_copy = True
                    current_transition_log_matrix[STATE_5_H_UNCTRL, STATE_5_A_CTRL] += UNCONFIRMED_TURNOVER_PENALTY
                    current_transition_log_matrix[STATE_5_H_CTRL, STATE_5_A_CTRL] += UNCONFIRMED_TURNOVER_PENALTY

                if not control_evidence.turnover_confirmed_A_to_H[frame_idx]:
                    if not matrix_needs_copy:
                        current_transition_log_matrix = base_log_A.copy()
                        matrix_needs_copy = True
                    current_transition_log_matrix[STATE_5_A_UNCTRL, STATE_5_H_CTRL] += UNCONFIRMED_TURNOVER_PENALTY
                    current_transition_log_matrix[STATE_5_A_CTRL, STATE_5_H_CTRL] += UNCONFIRMED_TURNOVER_PENALTY

        else:
            # -----------------------------------------------------------------
            # 3-State Dynamic Trellis Modulations (Rank 1 Engine)
            # -----------------------------------------------------------------
            if has_episodes:
                # 1. Stoppage Inertia Barrier:
                if (
                    prob_dead is not None
                    and prob_dead[frame_idx] > STOPPAGE_CONFIRMED_DEAD_PROBABILITY
                    and (prob_restart is None or prob_restart[frame_idx] < RESTART_ACTIVE_PROBABILITY_THRESHOLD)
                ):
                    if not matrix_needs_copy:
                        current_transition_log_matrix = base_log_A.copy()
                        matrix_needs_copy = True
                    current_transition_log_matrix[STATE_D, STATE_H] += PREMATURE_STOPPAGE_EXIT_PENALTY
                    current_transition_log_matrix[STATE_D, STATE_A] += PREMATURE_STOPPAGE_EXIT_PENALTY

                # 2. Restart Snap Potential:
                if (
                    prob_restart is not None
                    and prob_restart[frame_idx] > RESTART_ACTIVE_PROBABILITY_THRESHOLD
                    and restart_team is not None
                ):
                    kicking_team = restart_team[frame_idx]
                    if kicking_team in (0, 1):
                        if not matrix_needs_copy:
                            current_transition_log_matrix = base_log_A.copy()
                            matrix_needs_copy = True
                        current_transition_log_matrix[STATE_D, kicking_team] = CONFIRMED_RESTART_TRANSITION_PENALTY

        # Bellman broadcast update:
        # running_best_scores[:, np.newaxis] broadcasts over destination states j (shape K, K)
        path_transition_scores = running_best_scores[:, np.newaxis] + current_transition_log_matrix
        optimal_predecessors[frame_idx] = np.argmax(path_transition_scores, axis=0)
        running_best_scores = (
            path_transition_scores[optimal_predecessors[frame_idx], np.arange(n_states)]
            + log_emissions[frame_idx]
        )

    # =========================================================================
    # BACKTRACKING PASS (Optimal Sequence Recovery)
    # =========================================================================
    optimal_path = np.zeros(total_frames, dtype=np.int8)
    optimal_path[-1] = np.argmax(running_best_scores)

    for frame_idx in range(total_frames - 2, -1, -1):
        optimal_path[frame_idx] = optimal_predecessors[frame_idx + 1, optimal_path[frame_idx + 1]]

    # Map 5-state decoded path to 3 competition labels {0: H, 1: A, 2: D} if applicable:
    if n_states == 5:
        return MAP_5_TO_3[optimal_path]

    return optimal_path
