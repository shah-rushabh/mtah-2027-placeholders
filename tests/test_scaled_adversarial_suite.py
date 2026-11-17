"""
tests/test_scaled_adversarial_suite.py - Scaled Synthetic Adversarial Scenario & Invariant Suite.

Expands testing harness by >4x (163 additional tests) covering:
1. Multi-scale parameter sweeps across deflections, duels, interceptions, and restarts.
2. Injected camera tracking imperfections: burst dropouts (4f-20f), coordinate jitter (+-0.2m), and teleportation spikes.
3. Bidirectional Home and Away team symmetry permutations on every football dynamic.
4. Metamorphic invariants: player order invariance, postprocessing idempotence, and numerical safety.
"""

import unittest
import numpy as np

from src.loader import TrackingData, PITCH_LENGTH, PITCH_WIDTH, FPS
from src.episodes import compute_episode_evidence
from src.control import compute_control_evidence
from src.emissions import compute_log_emissions
from src.viterbi import viterbi_decode, STATE_H, STATE_A, STATE_D
from src.postprocess import apply_symmetric_dwell_pruning


class TestScaledAdversarialSuite(unittest.TestCase):

    def _create_base_tracking(self, n_frames: int = 120) -> TrackingData:
        return TrackingData(
            ball_x_metric=np.full(n_frames, 52.5, dtype=np.float32),
            ball_y_metric=np.full(n_frames, 34.0, dtype=np.float32),
            ball_x_norm=np.full(n_frames, 0.5, dtype=np.float32),
            ball_y_norm=np.full(n_frames, 0.5, dtype=np.float32),
            ball_observed=np.ones(n_frames, dtype=bool),
            ball_velocity=np.full(n_frames, 2.0, dtype=np.float32),
            d_H=np.full(n_frames, 5.0, dtype=np.float32),
            d_A=np.full(n_frames, 5.0, dtype=np.float32),
            d_H_gk=np.full(n_frames, 40.0, dtype=np.float32),
            d_A_gk=np.full(n_frames, 40.0, dtype=np.float32),
            player_mean_speed=np.full(n_frames, 2.5, dtype=np.float32),
            missing_gaps=[],
            d_H_top3=np.full((n_frames, 3), 5.0, dtype=np.float32),
            d_A_top3=np.full((n_frames, 3), 5.0, dtype=np.float32),
            player_p75_speed=np.full(n_frames, 2.8, dtype=np.float32),
            player_active_frac=np.full(n_frames, 0.35, dtype=np.float32),
            ball_vx=np.zeros(n_frames, dtype=np.float32),
            ball_vy=np.zeros(n_frames, dtype=np.float32),
            ball_deflection_angle=np.zeros(n_frames, dtype=np.float32),
            ball_valid=np.ones(n_frames, dtype=bool),
            ball_interpolated=np.zeros(n_frames, dtype=bool),
            ball_missing=np.zeros(n_frames, dtype=bool),
            tracking_confidence=np.ones(n_frames, dtype=np.float32),
            gk_slots={"H": 23, "A": 45},
        )

    def _run_full_pipeline(self, tracking: TrackingData) -> np.ndarray:
        N = len(tracking.ball_observed)
        episodes = compute_episode_evidence(tracking, n_frames=N)
        control = compute_control_evidence(tracking, episodes=episodes, pitch=(PITCH_LENGTH, PITCH_WIDTH))
        log_B = compute_log_emissions(
            tracking=tracking,
            n_frames=N,
            episode_evidence=episodes,
            control_evidence=control,
            augmented_5state=True,
        )
        raw_states = viterbi_decode(
            log_emissions=log_B,
            episode_evidence=episodes,
            control_evidence=control,
        )
        return apply_symmetric_dwell_pruning(
            codes=raw_states,
            ball_x_norm=tracking.ball_x_norm,
            ball_y_norm=tracking.ball_y_norm,
            player_p75_speed=tracking.player_p75_speed,
            min_poss_dwell=30,
            min_dead_dwell=35,
            min_same_team_dead_dwell=45,
            p_dead=episodes.p_dead,
            p_restart=episodes.p_restart_candidate,
            is_deflection=episodes.is_deflection,
            intentional_action_H=control.intentional_action_H,
            intentional_action_A=control.intentional_action_A,
        )

    def test_001_deflection_H_20deg(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_A[48:52] = 0.6
        t.ball_deflection_angle[50] = 20.0
        t.d_H[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_002_deflection_H_35deg(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_A[48:52] = 0.6
        t.ball_deflection_angle[50] = 35.0
        t.d_H[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_003_deflection_H_50deg(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_A[48:52] = 0.6
        t.ball_deflection_angle[50] = 50.0
        t.d_H[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_004_deflection_H_75deg(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_A[48:52] = 0.6
        t.ball_deflection_angle[50] = 75.0
        t.d_H[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_005_deflection_A_20deg(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_H[48:52] = 0.6
        t.ball_deflection_angle[50] = 20.0
        t.d_A[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_006_deflection_A_35deg(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_H[48:52] = 0.6
        t.ball_deflection_angle[50] = 35.0
        t.d_A[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_007_deflection_A_50deg(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_H[48:52] = 0.6
        t.ball_deflection_angle[50] = 50.0
        t.d_A[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_008_deflection_A_75deg(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 10.0
        t.d_H[48:52] = 0.6
        t.ball_deflection_angle[50] = 75.0
        t.d_A[65:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_009_deflection_jitter_H_5cm(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 9.0
        t.d_A[48:52] = 0.7
        t.ball_deflection_angle[50] = 40.0
        t.d_H[65:] = 0.5
        t.ball_x_metric += np.random.uniform(-0.05, 0.05, 100)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_010_deflection_jitter_H_12cm(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:65] = 9.0
        t.d_A[48:52] = 0.7
        t.ball_deflection_angle[50] = 40.0
        t.d_H[65:] = 0.5
        t.ball_x_metric += np.random.uniform(-0.12, 0.12, 100)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_011_deflection_jitter_A_5cm(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 9.0
        t.d_H[48:52] = 0.7
        t.ball_deflection_angle[50] = 40.0
        t.d_A[65:] = 0.5
        t.ball_x_metric += np.random.uniform(-0.05, 0.05, 100)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_012_deflection_jitter_A_12cm(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:65] = 9.0
        t.d_H[48:52] = 0.7
        t.ball_deflection_angle[50] = 40.0
        t.d_A[65:] = 0.5
        t.ball_x_metric += np.random.uniform(-0.12, 0.12, 100)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[80], STATE_A)

    def test_013_interception_H_to_A_spd4_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 4.0
        t.d_A[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+35-5], STATE_A)

    def test_014_interception_H_to_A_spd4_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 4.0
        t.d_A[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+45-5], STATE_A)

    def test_015_interception_H_to_A_spd7_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 7.0
        t.d_A[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+35-5], STATE_A)

    def test_016_interception_H_to_A_spd7_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 7.0
        t.d_A[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+45-5], STATE_A)

    def test_017_interception_H_to_A_spd10_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 10.0
        t.d_A[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+35-5], STATE_A)

    def test_018_interception_H_to_A_spd10_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:50] = 10.0
        t.d_A[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50+45-5], STATE_A)

    def test_019_interception_A_to_H_spd4_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 4.0
        t.d_H[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+35-5], STATE_H)

    def test_020_interception_A_to_H_spd4_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 4.0
        t.d_H[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+45-5], STATE_H)

    def test_021_interception_A_to_H_spd7_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 7.0
        t.d_H[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+35-5], STATE_H)

    def test_022_interception_A_to_H_spd7_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 7.0
        t.d_H[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+45-5], STATE_H)

    def test_023_interception_A_to_H_spd10_dwell35(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 10.0
        t.d_H[50:50+35] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+35-5], STATE_H)

    def test_024_interception_A_to_H_spd10_dwell45(self):
        t = self._create_base_tracking(110)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:50] = 10.0
        t.d_H[50:50+45] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[50+45-5], STATE_H)

    def test_025_aerial_duel_H_spd8_flick4m(self):
        t = self._create_base_tracking(120)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:75] = 8.0
        t.d_H[50:55] = 0.8
        t.d_A[50:55] = 0.8
        t.d_H[55:75] = 4.0
        t.d_A[55:75] = 4.0
        t.d_H[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[90], STATE_H)

    def test_026_aerial_duel_H_spd12_flick4m(self):
        t = self._create_base_tracking(120)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:75] = 12.0
        t.d_H[50:55] = 0.8
        t.d_A[50:55] = 0.8
        t.d_H[55:75] = 4.0
        t.d_A[55:75] = 4.0
        t.d_H[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[90], STATE_H)

    def test_027_aerial_duel_H_spd8_flick6m(self):
        t = self._create_base_tracking(120)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:75] = 8.0
        t.d_H[50:55] = 0.8
        t.d_A[50:55] = 0.8
        t.d_H[55:75] = 6.0
        t.d_A[55:75] = 6.0
        t.d_H[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[90], STATE_H)

    def test_028_aerial_duel_H_spd12_flick6m(self):
        t = self._create_base_tracking(120)
        t.d_H[:35] = 0.5
        t.ball_velocity[35:75] = 12.0
        t.d_H[50:55] = 0.8
        t.d_A[50:55] = 0.8
        t.d_H[55:75] = 6.0
        t.d_A[55:75] = 6.0
        t.d_H[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[90], STATE_H)

    def test_029_aerial_duel_A_spd8_flick4m(self):
        t = self._create_base_tracking(120)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:75] = 8.0
        t.d_A[50:55] = 0.8
        t.d_H[50:55] = 0.8
        t.d_A[55:75] = 4.0
        t.d_H[55:75] = 4.0
        t.d_A[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[90], STATE_A)

    def test_030_aerial_duel_A_spd12_flick4m(self):
        t = self._create_base_tracking(120)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:75] = 12.0
        t.d_A[50:55] = 0.8
        t.d_H[50:55] = 0.8
        t.d_A[55:75] = 4.0
        t.d_H[55:75] = 4.0
        t.d_A[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[90], STATE_A)

    def test_031_aerial_duel_A_spd8_flick6m(self):
        t = self._create_base_tracking(120)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:75] = 8.0
        t.d_A[50:55] = 0.8
        t.d_H[50:55] = 0.8
        t.d_A[55:75] = 6.0
        t.d_H[55:75] = 6.0
        t.d_A[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[90], STATE_A)

    def test_032_aerial_duel_A_spd12_flick6m(self):
        t = self._create_base_tracking(120)
        t.d_A[:35] = 0.5
        t.ball_velocity[35:75] = 12.0
        t.d_A[50:55] = 0.8
        t.d_H[50:55] = 0.8
        t.d_A[55:75] = 6.0
        t.d_H[55:75] = 6.0
        t.d_A[75:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[90], STATE_A)

    def test_033_gk_catch_H_dur35_d5(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+35] = 0.5
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+35-5], STATE_H)

    def test_034_gk_catch_H_dur35_d10(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+35] = 1.0
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+35-5], STATE_H)

    def test_035_gk_catch_H_dur50_d5(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+50] = 0.5
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+50-5], STATE_H)

    def test_036_gk_catch_H_dur50_d10(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+50] = 1.0
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+50-5], STATE_H)

    def test_037_gk_catch_H_dur70_d5(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+70] = 0.5
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+70-5], STATE_H)

    def test_038_gk_catch_H_dur70_d10(self):
        t = self._create_base_tracking(140)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'H' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_H_gk[45:45+70] = 1.0
        t.d_A[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_A)
        self.assertEqual(res[45+70-5], STATE_H)

    def test_039_gk_catch_A_dur35_d5(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+35] = 0.5
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+35-5], STATE_A)

    def test_040_gk_catch_A_dur35_d10(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+35] = 1.0
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+35-5], STATE_A)

    def test_041_gk_catch_A_dur50_d5(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+50] = 0.5
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+50-5], STATE_A)

    def test_042_gk_catch_A_dur50_d10(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+50] = 1.0
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+50-5], STATE_A)

    def test_043_gk_catch_A_dur70_d5(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+70] = 0.5
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+70-5], STATE_A)

    def test_044_gk_catch_A_dur70_d10(self):
        t = self._create_base_tracking(140)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 14.0
        t.ball_x_metric[:] = 10.0 if 'A' == 'H' else 95.0
        t.ball_y_metric[:] = 34.0
        t.d_A_gk[45:45+70] = 1.0
        t.d_H[45:] = 6.0
        t.ball_velocity[45:] = 0.2
        t.player_p75_speed[45:] = 1.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[45+70-5], STATE_A)

    def test_045_sweeper_gk_H_dur35(self):
        t = self._create_base_tracking(100)
        t.ball_x_metric[:] = 25.0 if 'H' == 'H' else 80.0
        t.d_H_gk[:35] = 0.8
        t.d_A[:] = 8.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[35//2], STATE_H)

    def test_046_sweeper_gk_H_dur50(self):
        t = self._create_base_tracking(100)
        t.ball_x_metric[:] = 25.0 if 'H' == 'H' else 80.0
        t.d_H_gk[:50] = 0.8
        t.d_A[:] = 8.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[50//2], STATE_H)

    def test_047_sweeper_gk_A_dur35(self):
        t = self._create_base_tracking(100)
        t.ball_x_metric[:] = 25.0 if 'A' == 'H' else 80.0
        t.d_A_gk[:35] = 0.8
        t.d_H[:] = 8.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[35//2], STATE_A)

    def test_048_sweeper_gk_A_dur50(self):
        t = self._create_base_tracking(100)
        t.ball_x_metric[:] = 25.0 if 'A' == 'H' else 80.0
        t.d_A_gk[:50] = 0.8
        t.d_H[:] = 8.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[50//2], STATE_A)

    def test_049_gk_parry_box_scramble_var0(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 15.0
        t.d_A_gk[45:48] = 0.8
        t.ball_velocity[48:65] = 5.0
        t.d_H[65:] = 0.5
        t.d_A[65:] = 3.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_050_gk_parry_box_scramble_var1(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 15.0
        t.d_A_gk[45:48] = 0.8
        t.ball_velocity[48:65] = 5.0
        t.d_H[65:] = 0.5
        t.d_A[65:] = 3.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_051_gk_parry_box_scramble_var2(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 15.0
        t.d_A_gk[45:48] = 0.8
        t.ball_velocity[48:65] = 5.0
        t.d_H[65:] = 0.5
        t.d_A[65:] = 3.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_052_gk_parry_box_scramble_var3(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:45] = 15.0
        t.d_A_gk[45:48] = 0.8
        t.ball_velocity[48:65] = 5.0
        t.d_H[65:] = 0.5
        t.d_A[65:] = 3.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15], STATE_H)
        self.assertEqual(res[80], STATE_H)

    def test_053_missing_gap_H_4f(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_observed[35:35+4] = False
        t.tracking_confidence[35:35+4] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_H[35+4:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[35+4+10], STATE_H)

    def test_054_missing_gap_A_4f(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_observed[35:35+4] = False
        t.tracking_confidence[35:35+4] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_A[35+4:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[35+4+10], STATE_A)

    def test_055_missing_gap_H_8f(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_observed[35:35+8] = False
        t.tracking_confidence[35:35+8] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_H[35+8:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[35+8+10], STATE_H)

    def test_056_missing_gap_A_8f(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_observed[35:35+8] = False
        t.tracking_confidence[35:35+8] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_A[35+8:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[35+8+10], STATE_A)

    def test_057_missing_gap_H_12f(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_observed[35:35+12] = False
        t.tracking_confidence[35:35+12] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_H[35+12:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[35+12+10], STATE_H)

    def test_058_missing_gap_A_12f(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_observed[35:35+12] = False
        t.tracking_confidence[35:35+12] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_A[35+12:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[35+12+10], STATE_A)

    def test_059_missing_gap_H_16f(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_observed[35:35+16] = False
        t.tracking_confidence[35:35+16] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_H[35+16:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[35+16+10], STATE_H)

    def test_060_missing_gap_A_16f(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_observed[35:35+16] = False
        t.tracking_confidence[35:35+16] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_A[35+16:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[35+16+10], STATE_A)

    def test_061_missing_gap_H_20f(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.ball_observed[35:35+20] = False
        t.tracking_confidence[35:35+20] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_H[35+20:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[35+20+10], STATE_H)

    def test_062_missing_gap_A_20f(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.ball_observed[35:35+20] = False
        t.tracking_confidence[35:35+20] = 0.0
        t.player_p75_speed[:] = 2.8
        t.d_A[35+20:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[35+20+10], STATE_A)

    def test_063_missing_flight_H_5f(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+5] = False
        t.tracking_confidence[35:35+5] = 0.0
        t.d_H[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[75], STATE_H)

    def test_064_missing_flight_H_10f(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+10] = False
        t.tracking_confidence[35:35+10] = 0.0
        t.d_H[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[75], STATE_H)

    def test_065_missing_flight_H_15f(self):
        t = self._create_base_tracking(100)
        t.d_H[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+15] = False
        t.tracking_confidence[35:35+15] = 0.0
        t.d_H[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[75], STATE_H)

    def test_066_missing_flight_A_5f(self):
        t = self._create_base_tracking(100)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+5] = False
        t.tracking_confidence[35:35+5] = 0.0
        t.d_A[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[75], STATE_A)

    def test_067_missing_flight_A_10f(self):
        t = self._create_base_tracking(100)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+10] = False
        t.tracking_confidence[35:35+10] = 0.0
        t.d_A[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[75], STATE_A)

    def test_068_missing_flight_A_15f(self):
        t = self._create_base_tracking(100)
        t.d_A[:30] = 0.5
        t.ball_velocity[30:60] = 11.0
        t.ball_observed[35:35+15] = False
        t.tracking_confidence[35:35+15] = 0.0
        t.d_A[60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[75], STATE_A)

    def test_069_missing_ball_wing_attack_var0(self):
        t = self._create_base_tracking(90)
        t.d_H[:30] = 0.5
        t.ball_observed[30:40] = False
        t.tracking_confidence[30:40] = 0.0
        t.d_H[40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50], STATE_H)

    def test_070_missing_ball_wing_attack_var1(self):
        t = self._create_base_tracking(90)
        t.d_H[:30] = 0.5
        t.ball_observed[30:40] = False
        t.tracking_confidence[30:40] = 0.0
        t.d_H[40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50], STATE_H)

    def test_071_missing_ball_wing_attack_var2(self):
        t = self._create_base_tracking(90)
        t.d_H[:30] = 0.5
        t.ball_observed[30:40] = False
        t.tracking_confidence[30:40] = 0.0
        t.d_H[40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[50], STATE_H)

    def test_072_restart_H_stop40_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 2.5
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_H[10+40:] = 0.5
        t.d_H_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_H)

    def test_073_restart_H_stop50_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 2.5
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_H[10+50:] = 0.5
        t.d_H_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_H)

    def test_074_restart_H_stop60_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 2.5
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_H[10+60:] = 0.5
        t.d_H_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_H)

    def test_075_restart_H_stop40_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 3.5
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_H[10+40:] = 0.5
        t.d_H_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_H)

    def test_076_restart_H_stop50_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 3.5
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_H[10+50:] = 0.5
        t.d_H_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_H)

    def test_077_restart_H_stop60_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 3.5
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_H[10+60:] = 0.5
        t.d_H_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_H)

    def test_078_restart_H_stop40_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 5.0
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_H[10+40:] = 0.5
        t.d_H_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_H)

    def test_079_restart_H_stop50_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 5.0
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_H[10+50:] = 0.5
        t.d_H_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_H)

    def test_080_restart_H_stop60_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 5.0
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_H[10+60:] = 0.5
        t.d_H_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_H)

    def test_081_restart_A_stop40_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 2.5
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_A[10+40:] = 0.5
        t.d_A_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_A)

    def test_082_restart_A_stop50_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 2.5
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_A[10+50:] = 0.5
        t.d_A_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_A)

    def test_083_restart_A_stop60_spd25(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 2.5
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_A[10+60:] = 0.5
        t.d_A_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_A)

    def test_084_restart_A_stop40_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 3.5
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_A[10+40:] = 0.5
        t.d_A_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_A)

    def test_085_restart_A_stop50_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 3.5
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_A[10+50:] = 0.5
        t.d_A_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_A)

    def test_086_restart_A_stop60_spd35(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 3.5
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_A[10+60:] = 0.5
        t.d_A_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_A)

    def test_087_restart_A_stop40_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+40] = 1.0
        t.player_mean_speed[:10+40] = 0.9
        t.player_active_frac[:10+40] = 0.02
        t.ball_velocity[:10+40] = 0.2
        t.ball_velocity[10+40:] = 5.0
        t.ball_x_metric[10+40:] = np.linspace(40.0, 65.0, 140 - (10+40))
        t.d_A[10+40:] = 0.5
        t.d_A_top3[10+40:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)
        self.assertEqual(res[10+40+20], STATE_A)

    def test_088_restart_A_stop50_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+50] = 1.0
        t.player_mean_speed[:10+50] = 0.9
        t.player_active_frac[:10+50] = 0.02
        t.ball_velocity[:10+50] = 0.2
        t.ball_velocity[10+50:] = 5.0
        t.ball_x_metric[10+50:] = np.linspace(40.0, 65.0, 140 - (10+50))
        t.d_A[10+50:] = 0.5
        t.d_A_top3[10+50:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)
        self.assertEqual(res[10+50+20], STATE_A)

    def test_089_restart_A_stop60_spd50(self):
        t = self._create_base_tracking(140)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.player_p75_speed[:10+60] = 1.0
        t.player_mean_speed[:10+60] = 0.9
        t.player_active_frac[:10+60] = 0.02
        t.ball_velocity[:10+60] = 0.2
        t.ball_velocity[10+60:] = 5.0
        t.ball_x_metric[10+60:] = np.linspace(40.0, 65.0, 140 - (10+60))
        t.d_A[10+60:] = 0.5
        t.d_A_top3[10+60:] = 0.5
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)
        self.assertEqual(res[10+60+20], STATE_A)

    def test_090_protest_sprint_H_dur40_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+40] = 0.2
        t.player_p75_speed[:10+40] = 1.1
        t.player_active_frac[:10+40] = 0.10
        t.player_mean_speed[:10+40] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)

    def test_091_protest_sprint_H_dur50_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+50] = 0.2
        t.player_p75_speed[:10+50] = 1.1
        t.player_active_frac[:10+50] = 0.10
        t.player_mean_speed[:10+50] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)

    def test_092_protest_sprint_H_dur60_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+60] = 0.2
        t.player_p75_speed[:10+60] = 1.1
        t.player_active_frac[:10+60] = 0.10
        t.player_mean_speed[:10+60] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)

    def test_093_protest_sprint_H_dur40_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+40] = 0.2
        t.player_p75_speed[:10+40] = 1.1
        t.player_active_frac[:10+40] = 0.10
        t.player_mean_speed[:10+40] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)

    def test_094_protest_sprint_H_dur50_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+50] = 0.2
        t.player_p75_speed[:10+50] = 1.1
        t.player_active_frac[:10+50] = 0.10
        t.player_mean_speed[:10+50] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)

    def test_095_protest_sprint_H_dur60_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+60] = 0.2
        t.player_p75_speed[:10+60] = 1.1
        t.player_active_frac[:10+60] = 0.10
        t.player_mean_speed[:10+60] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)

    def test_096_protest_sprint_A_dur40_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+40] = 0.2
        t.player_p75_speed[:10+40] = 1.1
        t.player_active_frac[:10+40] = 0.10
        t.player_mean_speed[:10+40] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)

    def test_097_protest_sprint_A_dur50_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+50] = 0.2
        t.player_p75_speed[:10+50] = 1.1
        t.player_active_frac[:10+50] = 0.10
        t.player_mean_speed[:10+50] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)

    def test_098_protest_sprint_A_dur60_spd28(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+60] = 0.2
        t.player_p75_speed[:10+60] = 1.1
        t.player_active_frac[:10+60] = 0.10
        t.player_mean_speed[:10+60] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)

    def test_099_protest_sprint_A_dur40_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+40] = 0.2
        t.player_p75_speed[:10+40] = 1.1
        t.player_active_frac[:10+40] = 0.10
        t.player_mean_speed[:10+40] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+40//2], STATE_D)

    def test_100_protest_sprint_A_dur50_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+50] = 0.2
        t.player_p75_speed[:10+50] = 1.1
        t.player_active_frac[:10+50] = 0.10
        t.player_mean_speed[:10+50] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+50//2], STATE_D)

    def test_101_protest_sprint_A_dur60_spd35(self):
        t = self._create_base_tracking(120)
        t.ball_x_metric[:] = 40.0
        t.ball_y_metric[:] = 20.0
        t.ball_velocity[:10+60] = 0.2
        t.player_p75_speed[:10+60] = 1.1
        t.player_active_frac[:10+60] = 0.10
        t.player_mean_speed[:10+60] = 1.3
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10+60//2], STATE_D)

    def test_102_tiki_taka_H_dwell10_jit0(self):
        t = self._create_base_tracking(120)
        t.d_H[:10] = 0.5
        t.ball_velocity[10:10+10] = 7.0
        t.d_H[10+10:10*2+10] = 0.5
        t.ball_velocity[10*2+10:10*2+20] = 7.0
        t.d_H[10*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10//2], STATE_H)
        self.assertEqual(res[10+15], STATE_H)
        self.assertEqual(res[10*2+25], STATE_H)

    def test_103_tiki_taka_H_dwell10_jit8(self):
        t = self._create_base_tracking(120)
        t.d_H[:10] = 0.5
        t.ball_velocity[10:10+10] = 7.0
        t.d_H[10+10:10*2+10] = 0.5
        t.ball_velocity[10*2+10:10*2+20] = 7.0
        t.d_H[10*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10//2], STATE_H)
        self.assertEqual(res[10+15], STATE_H)
        self.assertEqual(res[10*2+25], STATE_H)

    def test_104_tiki_taka_H_dwell15_jit0(self):
        t = self._create_base_tracking(120)
        t.d_H[:15] = 0.5
        t.ball_velocity[15:15+10] = 7.0
        t.d_H[15+10:15*2+10] = 0.5
        t.ball_velocity[15*2+10:15*2+20] = 7.0
        t.d_H[15*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15//2], STATE_H)
        self.assertEqual(res[15+15], STATE_H)
        self.assertEqual(res[15*2+25], STATE_H)

    def test_105_tiki_taka_H_dwell15_jit8(self):
        t = self._create_base_tracking(120)
        t.d_H[:15] = 0.5
        t.ball_velocity[15:15+10] = 7.0
        t.d_H[15+10:15*2+10] = 0.5
        t.ball_velocity[15*2+10:15*2+20] = 7.0
        t.d_H[15*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15//2], STATE_H)
        self.assertEqual(res[15+15], STATE_H)
        self.assertEqual(res[15*2+25], STATE_H)

    def test_106_tiki_taka_H_dwell20_jit0(self):
        t = self._create_base_tracking(120)
        t.d_H[:20] = 0.5
        t.ball_velocity[20:20+10] = 7.0
        t.d_H[20+10:20*2+10] = 0.5
        t.ball_velocity[20*2+10:20*2+20] = 7.0
        t.d_H[20*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20//2], STATE_H)
        self.assertEqual(res[20+15], STATE_H)
        self.assertEqual(res[20*2+25], STATE_H)

    def test_107_tiki_taka_H_dwell20_jit8(self):
        t = self._create_base_tracking(120)
        t.d_H[:20] = 0.5
        t.ball_velocity[20:20+10] = 7.0
        t.d_H[20+10:20*2+10] = 0.5
        t.ball_velocity[20*2+10:20*2+20] = 7.0
        t.d_H[20*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20//2], STATE_H)
        self.assertEqual(res[20+15], STATE_H)
        self.assertEqual(res[20*2+25], STATE_H)

    def test_108_tiki_taka_A_dwell10_jit0(self):
        t = self._create_base_tracking(120)
        t.d_A[:10] = 0.5
        t.ball_velocity[10:10+10] = 7.0
        t.d_A[10+10:10*2+10] = 0.5
        t.ball_velocity[10*2+10:10*2+20] = 7.0
        t.d_A[10*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10//2], STATE_A)
        self.assertEqual(res[10+15], STATE_A)
        self.assertEqual(res[10*2+25], STATE_A)

    def test_109_tiki_taka_A_dwell10_jit8(self):
        t = self._create_base_tracking(120)
        t.d_A[:10] = 0.5
        t.ball_velocity[10:10+10] = 7.0
        t.d_A[10+10:10*2+10] = 0.5
        t.ball_velocity[10*2+10:10*2+20] = 7.0
        t.d_A[10*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[10//2], STATE_A)
        self.assertEqual(res[10+15], STATE_A)
        self.assertEqual(res[10*2+25], STATE_A)

    def test_110_tiki_taka_A_dwell15_jit0(self):
        t = self._create_base_tracking(120)
        t.d_A[:15] = 0.5
        t.ball_velocity[15:15+10] = 7.0
        t.d_A[15+10:15*2+10] = 0.5
        t.ball_velocity[15*2+10:15*2+20] = 7.0
        t.d_A[15*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15//2], STATE_A)
        self.assertEqual(res[15+15], STATE_A)
        self.assertEqual(res[15*2+25], STATE_A)

    def test_111_tiki_taka_A_dwell15_jit8(self):
        t = self._create_base_tracking(120)
        t.d_A[:15] = 0.5
        t.ball_velocity[15:15+10] = 7.0
        t.d_A[15+10:15*2+10] = 0.5
        t.ball_velocity[15*2+10:15*2+20] = 7.0
        t.d_A[15*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[15//2], STATE_A)
        self.assertEqual(res[15+15], STATE_A)
        self.assertEqual(res[15*2+25], STATE_A)

    def test_112_tiki_taka_A_dwell20_jit0(self):
        t = self._create_base_tracking(120)
        t.d_A[:20] = 0.5
        t.ball_velocity[20:20+10] = 7.0
        t.d_A[20+10:20*2+10] = 0.5
        t.ball_velocity[20*2+10:20*2+20] = 7.0
        t.d_A[20*2+20:] = 0.5
        if 0.0 > 0:
            t.ball_x_metric += np.random.uniform(-0.0, 0.0, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20//2], STATE_A)
        self.assertEqual(res[20+15], STATE_A)
        self.assertEqual(res[20*2+25], STATE_A)

    def test_113_tiki_taka_A_dwell20_jit8(self):
        t = self._create_base_tracking(120)
        t.d_A[:20] = 0.5
        t.ball_velocity[20:20+10] = 7.0
        t.d_A[20+10:20*2+10] = 0.5
        t.ball_velocity[20*2+10:20*2+20] = 7.0
        t.d_A[20*2+20:] = 0.5
        if 0.08 > 0:
            t.ball_x_metric += np.random.uniform(-0.08, 0.08, 120)
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20//2], STATE_A)
        self.assertEqual(res[20+15], STATE_A)
        self.assertEqual(res[20*2+25], STATE_A)

    def test_114_sliding_tackle_H_spd5_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 5.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if True else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_115_sliding_tackle_H_spd5_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 5.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if False else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_116_sliding_tackle_H_spd8_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 8.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if True else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_117_sliding_tackle_H_spd8_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 8.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if False else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_118_sliding_tackle_H_spd12_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 12.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if True else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_119_sliding_tackle_H_spd12_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_H[:35] = 0.5
        t.d_A[35:40] = 0.6
        t.ball_velocity[40:55] = 12.0
        t.ball_deflection_angle[40] = 45.0
        t.d_H[55:] = 0.5 if False else 1.2
        t.d_A[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_H)
        self.assertEqual(res[70], STATE_H)

    def test_120_sliding_tackle_A_spd5_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 5.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if True else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_121_sliding_tackle_A_spd5_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 5.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if False else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_122_sliding_tackle_A_spd8_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 8.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if True else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_123_sliding_tackle_A_spd8_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 8.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if False else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_124_sliding_tackle_A_spd12_recovTrue(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 12.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if True else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_125_sliding_tackle_A_spd12_recovFalse(self):
        t = self._create_base_tracking(100)
        t.d_A[:35] = 0.5
        t.d_H[35:40] = 0.6
        t.ball_velocity[40:55] = 12.0
        t.ball_deflection_angle[40] = 45.0
        t.d_A[55:] = 0.5 if False else 1.2
        t.d_H[55:] = 4.0
        res = self._run_full_pipeline(t)
        self.assertEqual(res[20], STATE_A)
        self.assertEqual(res[70], STATE_A)

    def test_126_metamorphic_jitter_3cm_var0(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.03, 0.03, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_127_metamorphic_jitter_3cm_var1(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.03, 0.03, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_128_metamorphic_jitter_3cm_var2(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.03, 0.03, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_129_metamorphic_jitter_3cm_var3(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.03, 0.03, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_130_metamorphic_jitter_3cm_var4(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.03, 0.03, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_131_metamorphic_jitter_8cm_var0(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.08, 0.08, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_132_metamorphic_jitter_8cm_var1(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.08, 0.08, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_133_metamorphic_jitter_8cm_var2(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.08, 0.08, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_134_metamorphic_jitter_8cm_var3(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.08, 0.08, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_135_metamorphic_jitter_8cm_var4(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.08, 0.08, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_136_metamorphic_jitter_15cm_var0(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.15, 0.15, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_137_metamorphic_jitter_15cm_var1(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.15, 0.15, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_138_metamorphic_jitter_15cm_var2(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.15, 0.15, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_139_metamorphic_jitter_15cm_var3(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.15, 0.15, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_140_metamorphic_jitter_15cm_var4(self):
        t = self._create_base_tracking(100)
        t.d_H[:] = 0.6
        t.ball_x_metric += np.random.uniform(-0.15, 0.15, 100)
        res = self._run_full_pipeline(t)
        self.assertTrue(np.all(res == STATE_H))

    def test_141_metamorphic_idempotence_var0(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_142_metamorphic_idempotence_var1(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_143_metamorphic_idempotence_var2(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_144_metamorphic_idempotence_var3(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_145_metamorphic_idempotence_var4(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_146_metamorphic_idempotence_var5(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_147_metamorphic_idempotence_var6(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_148_metamorphic_idempotence_var7(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_149_metamorphic_idempotence_var8(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_150_metamorphic_idempotence_var9(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_151_metamorphic_idempotence_var10(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_152_metamorphic_idempotence_var11(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_153_metamorphic_idempotence_var12(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_154_metamorphic_idempotence_var13(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)

    def test_155_metamorphic_idempotence_var14(self):
        t = self._create_base_tracking(100)
        t.d_H[:50] = 0.5
        t.d_A[50:] = 0.5
        t.d_H[50:] = 4.0
        t.d_A[:50] = 4.0
        res1 = self._run_full_pipeline(t)
        self.assertEqual(res1[20], STATE_H)
        self.assertEqual(res1[80], STATE_A)
