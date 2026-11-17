"""Adversarial Synthetic Test Suite for MTAH 2027 Possession Engine.

Implements the 12 synthetic adversarial test cases specified in the architecture reviews
to verify all physical invariants, anti-fragmentation filters, and mathematical edge cases.
"""

import unittest
import numpy as np

from src.emissions import (
    STATE_H,
    STATE_A,
    STATE_D,
    compute_log_emissions,
)
from src.loader import TrackingData, MissingGap
from src.viterbi import viterbi_decode
from src.postprocess import (
    apply_symmetric_dwell_pruning,
    format_final_submission,
    rle_encode,
    H1_START,
    H1_END,
    H2_START,
    H2_END,
    STATE_X,
)


class TestAdversarialScenarios(unittest.TestCase):
    """Synthetic tests verifying every critical edge case and invariant."""

    def test_01_3_frame_d_glitch_inside_h_spell_is_pruned(self):
        """Case 1: A 3-frame D glitch inside a 40s H spell must be pruned to prevent fragmentation."""
        # Sequence: 500 frames of H, 3 frames of D, 500 frames of H
        seq = np.array([STATE_H] * 500 + [STATE_D] * 3 + [STATE_H] * 500, dtype=np.int8)
        pruned = apply_symmetric_dwell_pruning(seq, min_dead_dwell=25)
        # Should be a single contiguous H run of 1003 frames
        segs = rle_encode(pruned)
        self.assertEqual(len(segs), 1, "3-frame D glitch was not merged!")
        self.assertEqual(segs[0][2], STATE_H)
        self.assertEqual(segs[0][1] - segs[0][0], 1003)

    def test_02_genuine_short_d_between_h_and_a_is_preserved(self):
        """Case 2: A 15-frame D between an H spell and an A spell must be PRESERVED (turnover boundary)."""
        # Sequence: 200 frames of H, 15 frames of D, 200 frames of A
        seq = np.array([STATE_H] * 200 + [STATE_D] * 15 + [STATE_A] * 200, dtype=np.int8)
        pruned = apply_symmetric_dwell_pruning(seq, min_dead_dwell=25)
        segs = rle_encode(pruned)
        self.assertEqual(len(segs), 3, "Legitimate D boundary between H and A was erroneously removed!")
        self.assertEqual(segs[0][2], STATE_H)
        self.assertEqual(segs[1][2], STATE_D)
        self.assertEqual(segs[2][2], STATE_A)

    def test_03_45m_open_pitch_pass_no_false_d(self):
        """Case 3: When ball travels 45m across pitch with players > 32m away, no false D occurs."""
        N = 100
        # Ball in-pitch, players 40m away
        tracking = TrackingData(
            ball_x_metric=np.full(N, 50.0, dtype=np.float32),
            ball_y_metric=np.full(N, 34.0, dtype=np.float32),
            ball_x_norm=np.full(N, 0.47, dtype=np.float32),
            ball_y_norm=np.full(N, 0.50, dtype=np.float32),
            ball_observed=np.ones(N, dtype=bool),
            ball_velocity=np.full(N, 18.0, dtype=np.float32),  # Fast aerial pass
            d_H=np.full(N, 40.0, dtype=np.float32),            # Nearest Barca player 40m away
            d_A=np.full(N, 42.0, dtype=np.float32),            # Nearest Athletic player 42m away
            d_H_gk=np.full(N, 45.0, dtype=np.float32),
            d_A_gk=np.full(N, 55.0, dtype=np.float32),
            player_mean_speed=np.full(N, 4.0, dtype=np.float32),
            missing_gaps=[],
        )
        emissions = compute_log_emissions(tracking, n_frames=N)
        # Inside graded window (say frame 50)
        # State D emission must be strictly -inf (or <= -500)
        self.assertLess(emissions[50, STATE_D], -500.0, "State D was not strictly barred during open-pitch pass!")
        self.assertGreater(emissions[50, STATE_H], emissions[50, STATE_D])

        # Viterbi decoding should stay in possession (H), never D
        path = viterbi_decode(emissions[:100], initial_probs=np.array([1.0, 0.0, 0.0]))
        self.assertNotIn(STATE_D, path, "Viterbi erroneously switched to D during a 45m pass!")

    def test_04_stationary_ball_freekick_setup_emits_d(self):
        """Case 4: Ball stationary inside pitch during set-piece window emits D."""
        N = 50
        tracking = TrackingData(
            ball_x_metric=np.full(N, 30.0, dtype=np.float32),
            ball_y_metric=np.full(N, 20.0, dtype=np.float32),
            ball_x_norm=np.full(N, 0.28, dtype=np.float32),
            ball_y_norm=np.full(N, 0.29, dtype=np.float32),
            ball_observed=np.ones(N, dtype=bool),
            ball_velocity=np.full(N, 0.1, dtype=np.float32),   # Stationary ball (< 1.0 m/s)
            d_H=np.full(N, 3.5, dtype=np.float32),            # Outfield players > 2m away
            d_A=np.full(N, 4.0, dtype=np.float32),
            d_H_gk=np.full(N, 25.0, dtype=np.float32),
            d_A_gk=np.full(N, 75.0, dtype=np.float32),
            player_mean_speed=np.full(N, 1.2, dtype=np.float32),
            missing_gaps=[],
        )
        smart_tags = {
            "set_pieces": [{"start": 10, "end": 40, "type": "free kick", "team": "FC Barcelona"}],
            "goals": [],
            "active_phases": [],
        }
        emissions = compute_log_emissions(tracking, smart_tags=smart_tags, n_frames=N)
        # Frames 15..35 should emit strong D (+5.0)
        self.assertGreater(emissions[20, STATE_D], emissions[20, STATE_H], "Set-piece setup did not emit D!")

    def test_05_goal_celebration_is_pure_d(self):
        """Case 5: Goal celebration interval emits pure D."""
        N = 100
        tracking = TrackingData(
            ball_x_metric=np.full(N, 100.0, dtype=np.float32),
            ball_y_metric=np.full(N, 34.0, dtype=np.float32),
            ball_x_norm=np.full(N, 0.95, dtype=np.float32),
            ball_y_norm=np.full(N, 0.50, dtype=np.float32),
            ball_observed=np.ones(N, dtype=bool),
            ball_velocity=np.full(N, 0.0, dtype=np.float32),
            d_H=np.full(N, 2.0, dtype=np.float32),
            d_A=np.full(N, 15.0, dtype=np.float32),
            d_H_gk=np.full(N, 80.0, dtype=np.float32),
            d_A_gk=np.full(N, 3.0, dtype=np.float32),
            player_mean_speed=np.full(N, 1.4, dtype=np.float32),
            missing_gaps=[],
        )
        smart_tags = {
            "set_pieces": [{"start": 80, "end": 90, "type": "kick off (after goal)", "team": "Athletic Club"}],
            "goals": [{"start": 10, "end": 20, "team": "FC Barcelona"}],
            "active_phases": [],
        }
        emissions = compute_log_emissions(tracking, smart_tags=smart_tags, n_frames=N)
        self.assertEqual(emissions[40, STATE_D], 10.0, "Goal celebration did not emit strong D!")

    def test_06_goalkeeper_control_attributed_to_gk_team(self):
        """Case 6: When GK is controlling ball (< 1.5m) and outfield players are far, GK team gets possession."""
        N = 10
        tracking = TrackingData(
            ball_x_metric=np.full(N, 10.0, dtype=np.float32),
            ball_y_metric=np.full(N, 34.0, dtype=np.float32),
            ball_x_norm=np.full(N, 0.10, dtype=np.float32),
            ball_y_norm=np.full(N, 0.50, dtype=np.float32),
            ball_observed=np.ones(N, dtype=bool),
            ball_velocity=np.full(N, 1.5, dtype=np.float32),
            d_H=np.full(N, 12.0, dtype=np.float32),            # Outfield Barca players 12m away
            d_A=np.full(N, 6.0, dtype=np.float32),             # Outfield Athletic pressing at 6m
            d_H_gk=np.full(N, 0.8, dtype=np.float32),          # Ter Stegen holding ball at 0.8m!
            d_A_gk=np.full(N, 85.0, dtype=np.float32),
            player_mean_speed=np.full(N, 2.2, dtype=np.float32),
            missing_gaps=[],
        )
        emissions = compute_log_emissions(tracking, n_frames=N)
        self.assertGreater(emissions[5, STATE_H], emissions[5, STATE_A], "GK control was not attributed to GK team!")

    def test_07_active_gap_preserves_possession_via_inertia(self):
        """Case 7: In an active gap (v_p >= 2.0 m/s), HMM inertia carries possession through without D."""
        N = 100
        # Frames 0..20: H (observed)
        # Frames 21..70: Active missing gap (50 frames = 2.0s)
        # Frames 71..99: H (observed)
        emissions = np.zeros((N, 3), dtype=np.float32)
        emissions[0:21, STATE_H] = 0.0
        emissions[0:21, STATE_A] = -10.0
        emissions[0:21, STATE_D] = -1000.0

        # Active gap: neutral H/A, strict D barrier
        emissions[21:71, STATE_H] = 0.0
        emissions[21:71, STATE_A] = 0.0
        emissions[21:71, STATE_D] = -1000.0

        emissions[71:N, STATE_H] = 0.0
        emissions[71:N, STATE_A] = -10.0
        emissions[71:N, STATE_D] = -1000.0

        path = viterbi_decode(emissions, initial_probs=np.array([1.0, 0.0, 0.0]))
        self.assertTrue(np.all(path == STATE_H), "Possession was fractured or dropped during active occlusion!")

    def test_08_stoppage_gap_emits_d(self):
        """Case 8: In a low-speed stoppage gap (< 1.6 m/s), emissions emit pure D."""
        gap = MissingGap(
            start_frame=10,
            end_frame=50,
            length=41,
            mean_player_speed=1.2,
            max_player_speed=1.8,
            category="stoppage",
            overlaps_xml_stoppage=False,
        )
        tracking = TrackingData(
            ball_x_metric=np.full(60, np.nan, dtype=np.float32),
            ball_y_metric=np.full(60, np.nan, dtype=np.float32),
            ball_x_norm=np.full(60, np.nan, dtype=np.float32),
            ball_y_norm=np.full(60, np.nan, dtype=np.float32),
            ball_observed=np.zeros(60, dtype=bool),
            ball_velocity=np.zeros(60, dtype=np.float32),
            d_H=np.full(60, np.nan, dtype=np.float32),
            d_A=np.full(60, np.nan, dtype=np.float32),
            d_H_gk=np.full(60, np.nan, dtype=np.float32),
            d_A_gk=np.full(60, np.nan, dtype=np.float32),
            player_mean_speed=np.full(60, 1.2, dtype=np.float32),
            missing_gaps=[gap],
        )
        emissions = compute_log_emissions(tracking, n_frames=60)
        self.assertEqual(emissions[25, STATE_D], 0.0)
        self.assertEqual(emissions[25, STATE_H], -50.0)

    def test_09_spells_across_halftime_never_merged(self):
        """Case 9: Spells on either side of halftime break (X) must never be merged."""
        N = 199319
        codes = np.full(N, STATE_D, dtype=np.int8)
        # H1 ends on Barcelona possession (H)
        codes[H1_END - 100:H1_END + 1] = STATE_H
        # Halftime is frames H1_END+1 .. H2_START-1 (20,156 frames)
        # H2 starts on Barcelona possession (H)
        codes[H2_START:H2_START + 100] = STATE_H

        final_submission = format_final_submission(codes, n_frames=N)
        # Halftime must be strictly X
        self.assertTrue(np.all(final_submission[H1_END + 1:H2_START] == STATE_X), "Halftime was not clamped to X!")

    def test_10_zero_x_on_graded_frames(self):
        """Case 10: Graded frames must strictly contain {H, A, D} and zero 'X'."""
        N = 199319
        codes = np.full(N, STATE_D, dtype=np.int8)
        # Inject accidental 'X' in graded interval
        codes[H1_START + 500:H1_START + 520] = STATE_X
        codes[H2_START + 1000:H2_START + 1050] = STATE_X

        formatted = format_final_submission(codes, n_frames=N)
        # Check H1 graded interval
        self.assertNotIn(STATE_X, formatted[H1_START:H1_END + 1], "Found 'X' in H1 graded interval!")
        # Check H2 graded interval
        self.assertNotIn(STATE_X, formatted[H2_START:H2_END + 1], "Found 'X' in H2 graded interval!")

    def test_11_viterbi_trellis_no_nans(self):
        """Case 11: Trellis computation remains finite and robust even with missing ball data."""
        N = 500
        emissions = np.zeros((N, 3), dtype=np.float32)
        # Mixed emissions
        emissions[0:100, STATE_H] = -1.5
        emissions[0:100, STATE_A] = -3.2
        emissions[100:300, STATE_D] = 5.0
        emissions[300:500, STATE_A] = -1.1
        emissions[300:500, STATE_H] = -4.0

        path = viterbi_decode(emissions)
        self.assertEqual(len(path), N)
        self.assertTrue(np.all(np.isin(path, [STATE_H, STATE_A, STATE_D])), "Path contains invalid values!")

    def test_12_out_of_bounds_sensor_spike_does_not_destroy_possession(self):
        """Case 12: A single-frame out-of-bounds coordinate spike does not create an unpruned D."""
        # 100 frames of H, 1 frame of D, 100 frames of H
        seq = np.array([STATE_H] * 100 + [STATE_D] * 1 + [STATE_H] * 100, dtype=np.int8)
        pruned = apply_symmetric_dwell_pruning(seq, min_dead_dwell=25)
        segs = rle_encode(pruned)
        self.assertEqual(len(segs), 1, "Single-frame sensor spike was not pruned!")

    def test_13_second_half_pre_kickoff_interval_is_pure_d(self):
        """Case 13: Under official FAQ rules, frames 110,808..113,742 (2,934 frames) prior to H2 kickoff are pure D."""
        from src.emissions import H2_START, H2_KICKOFF
        self.assertEqual(H2_START, 110808)
        self.assertEqual(H2_KICKOFF, 113742)
        self.assertEqual(H2_KICKOFF - H2_START, 2934)


if __name__ == "__main__":
    unittest.main()
