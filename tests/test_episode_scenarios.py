"""Adversarial and synthetic test suite for Phase 2: Game Phase & Episode Evidence Engine.

Tests critical football scenarios:
1. Slow live tiki-taka build-up (must stay LIVE)
2. In-pitch ball taps during stoppage (must stay DEAD)
3. High lob passing directly above an opposing player (must preserve possession)
4. Penalty box goalkeeper slide tackle (must activate GK)
5. Set-piece preparation prior to restart impulse (must stay DEAD until kick)
"""

import unittest
import numpy as np

from src.loader import TrackingData, MissingGap
from src.episodes import compute_episode_evidence, EpisodeEvidence
from src.emissions import compute_log_emissions, STATE_H, STATE_A, STATE_D
from src.viterbi import viterbi_decode


class TestEpisodeScenarios(unittest.TestCase):

    def _create_synthetic_tracking(self, n_frames=100):
        return TrackingData(
            ball_x_metric=np.full(n_frames, 52.5, dtype=np.float32),
            ball_y_metric=np.full(n_frames, 34.0, dtype=np.float32),
            ball_x_norm=np.full(n_frames, 0.5, dtype=np.float32),
            ball_y_norm=np.full(n_frames, 0.5, dtype=np.float32),
            ball_observed=np.ones(n_frames, dtype=bool),
            ball_velocity=np.full(n_frames, 2.0, dtype=np.float32),
            d_H=np.full(n_frames, 1.0, dtype=np.float32),
            d_A=np.full(n_frames, 5.0, dtype=np.float32),
            d_H_top3=np.full((n_frames, 3), 1.0, dtype=np.float32),
            d_A_top3=np.full((n_frames, 3), 5.0, dtype=np.float32),
            d_H_gk=np.full(n_frames, 40.0, dtype=np.float32),
            d_A_gk=np.full(n_frames, 40.0, dtype=np.float32),
            player_mean_speed=np.full(n_frames, 2.5, dtype=np.float32),
            player_p75_speed=np.full(n_frames, 2.8, dtype=np.float32),
            player_active_frac=np.full(n_frames, 0.3, dtype=np.float32),
            missing_gaps=[],
        )

    def test_slow_tiki_taka_stays_live(self):
        """Barcelona walking on the halfway line with slow ball movements must stay LIVE."""
        N = 80
        tracking = self._create_synthetic_tracking(N)
        # Players walking slowly (< 1.3 m/s), but ball is moving continuously across 5m
        tracking.player_p75_speed[:] = 1.2
        tracking.player_mean_speed[:] = 1.1
        tracking.player_active_frac[:] = 0.05
        tracking.ball_velocity[:] = 1.5
        # Ball moves steadily along x: 20m to 35m
        tracking.ball_x_metric[:] = np.linspace(20.0, 35.0, N)
        tracking.d_H[:] = 0.5  # Barca player dribbling
        tracking.d_A[:] = 4.0

        evidence = compute_episode_evidence(tracking, n_frames=N)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence)
        decoded = viterbi_decode(log_B, episode_evidence=evidence)

        # Must NOT classify as Dead Ball; Barcelona must maintain possession
        self.assertTrue(np.all(decoded == STATE_H), "Slow tiki-taka was incorrectly flagged as DEAD or lost!")

    def test_in_pitch_ball_tapping_stoppage(self):
        """Foul whistle inside the pitch; players tap ball around lazily (must detect DEAD)."""
        N = 100
        tracking = self._create_synthetic_tracking(N)
        # Low player activity (< 1.1 m/s) and stationary localized ball (displacement < 1.0m)
        tracking.player_p75_speed[:] = 1.0
        tracking.player_mean_speed[:] = 0.9
        tracking.player_active_frac[:] = 0.02
        tracking.ball_velocity[:] = 0.2
        # Occasional ball taps at frame 40 and 60
        tracking.ball_velocity[40:43] = 1.8
        tracking.ball_velocity[60:63] = 2.1
        tracking.ball_x_metric[:] = 31.0 + np.random.uniform(-0.3, 0.3, N)
        tracking.ball_y_metric[:] = 55.0 + np.random.uniform(-0.3, 0.3, N)

        evidence = compute_episode_evidence(tracking, n_frames=N)
        # After initial 20 frames of hysteresis, p_dead should dominate
        self.assertTrue(np.all(evidence.p_dead[25:95] > 0.8), "In-pitch stoppage was not detected!")

        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence)
        decoded = viterbi_decode(log_B, episode_evidence=evidence)
        self.assertEqual(decoded[50], STATE_D, "Decoded state inside stoppage should be DEAD!")

    def test_high_lob_passes_over_opponent(self):
        """Athletic lobs ball across pitch; ball passes directly over Barca player mid-flight."""
        N = 60
        tracking = self._create_synthetic_tracking(N)
        # Pass is controlled by Athletic, kicked at frame 15, flies until frame 45, lands with Athletic
        tracking.ball_velocity[:15] = 2.0
        tracking.ball_velocity[15:45] = 10.0
        tracking.ball_velocity[45:] = 2.0

        # Passing team is Athletic: starts close to Athletic at frame 0..15
        tracking.d_A[:15] = 0.5
        tracking.d_A_top3[:15] = 0.5
        tracking.d_H[:15] = 5.0
        tracking.d_H_top3[:15] = 5.0

        # Mid-flight (frames 15..45): 2D projection passes over a Barca player (dH = 0.6m, dA = 6m)
        tracking.d_H[15:45] = 0.6
        tracking.d_H_top3[15:45] = 0.6
        tracking.d_A[15:45] = 6.0
        tracking.d_A_top3[15:45] = 6.0

        # Lands with Athletic at frame 45..60
        tracking.d_A[45:] = 0.5
        tracking.d_A_top3[45:] = 0.5
        tracking.d_H[45:] = 5.0
        tracking.d_H_top3[45:] = 5.0

        evidence = compute_episode_evidence(tracking, n_frames=N)
        self.assertTrue(np.all(evidence.p_fast_transit[15:45] > 0.5), "Fast transit was not detected!")

        log_B = compute_log_emissions(
            tracking,
            n_frames=N,
            ownership_strategy="softmin_multi",
            episode_evidence=evidence
        )
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        # Athletic must maintain possession through the lob; must NOT flip to Barcelona
        n_away = np.sum(decoded[15:45] == STATE_A)
        self.assertEqual(n_away, 30, f"Ground projection captured mid-flight lob! Only {n_away}/30 frames were Away.")

    def test_penalty_box_goalkeeper_active(self):
        """Goalkeeper slides inside the 18-yard penalty box (must activate GK)."""
        N = 30
        tracking = self._create_synthetic_tracking(N)
        # Ball is inside Home penalty box: x=10.0m, y=34.0m
        tracking.ball_x_metric[:] = 10.0
        tracking.ball_y_metric[:] = 34.0
        tracking.ball_x_norm[:] = 10.0 / 105.0
        tracking.ball_y_norm[:] = 34.0 / 68.0
        # Outfield Barca defenders are 8m away, but Ter Stegen is 0.5m away
        tracking.d_H[:] = 8.0
        tracking.d_A[:] = 1.2
        tracking.d_H_gk[:] = 0.5  # Ter Stegen slide clearance

        evidence = compute_episode_evidence(tracking, n_frames=N)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence)
        decoded = viterbi_decode(log_B, episode_evidence=evidence)

        self.assertEqual(decoded[15], STATE_H, "Ter Stegen's slide tackle inside box failed to award possession!")

    def test_gk_kick_to_midfield_freekick_setup(self):
        """GK kicks ball 35m to midfielder to set up free kick; stays DEAD until midfielder kicks."""
        N = 350
        tracking = self._create_synthetic_tracking(N)
        # Phase 1: Ball at GK feet for 20 frames (0..19)
        tracking.ball_x_metric[:] = 15.0
        tracking.ball_y_metric[:] = 34.0
        # Phase 2: GK kicks ball 35m to midfield over 30 frames (20..49)
        tracking.ball_x_metric[20:50] = np.linspace(15.0, 50.0, 30)
        # Phase 3: Midfielder places ball, stationary for 150 frames (50..199)
        tracking.ball_x_metric[50:200] = 50.0
        # Phase 4: Midfielder takes actual free kick at frame 200 (200..260)
        tracking.ball_x_metric[200:260] = np.linspace(50.0, 85.0, 60)
        tracking.ball_x_metric[260:] = np.linspace(85.0, 95.0, 90)

        tracking.ball_x_norm[:] = tracking.ball_x_metric / 105.0
        tracking.ball_y_norm[:] = tracking.ball_y_metric / 68.0

        vx = np.gradient(tracking.ball_x_metric) * 25.0
        vy = np.gradient(tracking.ball_y_metric) * 25.0
        tracking.ball_velocity[:] = np.sqrt(vx * vx + vy * vy)

        tracking.player_p75_speed[:] = 1.2
        tracking.player_active_frac[:] = 0.08
        tracking.player_p75_speed[200:] = 3.5
        tracking.player_active_frac[200:] = 0.45

        smart_tags = {
            "set_pieces": [{
                "type": "Free Kick",
                "team": "Barcelona",
                "start": 30,
                "end": 280
            }],
            "active_phases": [],
            "goals": []
        }

        evidence = compute_episode_evidence(tracking, smart_tags=smart_tags, n_frames=N)
        log_B = compute_log_emissions(tracking, smart_tags=smart_tags, episode_evidence=evidence, n_frames=N)
        decoded = viterbi_decode(log_B, episode_evidence=evidence)

        # The setup kick (frame 25) and waiting interval (frame 100) must be DEAD (STATE_D)
        self.assertEqual(decoded[25], STATE_D, "GK setup kick to midfielder was falsely marked as live play!")
        self.assertEqual(decoded[100], STATE_D, "Waiting for free kick setup was not marked as DEAD!")
        # After actual kick at frame 200, play snaps to HOME (STATE_H)
        self.assertEqual(decoded[220], STATE_H, "Actual free kick was not detected as live play!")
        self.assertEqual(decoded[280], STATE_H, "Subsequent attacking play was not recognized as Home possession!")


if __name__ == "__main__":
    unittest.main()
