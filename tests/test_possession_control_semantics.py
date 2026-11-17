"""Synthetic Test Suite: Official MTAH 2027 Possession Control Semantics.

Directly tests the official competition definition of possession:
"A change of possession occurs when the opposing team gains controlled possession.
During loose-ball or contested phases, possession stays with the team that last had it.
A simple touch does not constitute a possession change.
A deflection that does not intercept a pass does not constitute a change.
Winning an aerial duel without subsequent control does not constitute a change."

Scenarios tested:
1. Aerial duel deflection without opponent control (possession stays with kicking team).
2. Goalkeeper parry vs catch (parry = attacking possession persists; catch = goalkeeper control).
3. Shot rebound off post/defender into contested space (retains attacking possession).
4. Sliding tackle / toe-poke deflection without opponent control (possession persists).
5. One-touch rapid passing combination (continuous live possession maintained).
6. Failed reception / heavy miscontrol rolling into contested grass (retains passing team).
"""

import unittest
import numpy as np

from src.emissions import (
    STATE_H,
    STATE_A,
    STATE_D,
    compute_log_emissions,
)
from src.episodes import compute_episode_evidence
from src.loader import TrackingData
from src.viterbi import viterbi_decode


class TestPossessionControlSemantics(unittest.TestCase):
    """Verifies that ball contact is distinguished from controlled possession."""

    def _base_tracking(self, n_frames=60):
        return TrackingData(
            ball_x_metric=np.full(n_frames, 50.0, dtype=np.float32),
            ball_y_metric=np.full(n_frames, 34.0, dtype=np.float32),
            ball_x_norm=np.full(n_frames, 0.47, dtype=np.float32),
            ball_y_norm=np.full(n_frames, 0.50, dtype=np.float32),
            ball_observed=np.ones(n_frames, dtype=bool),
            ball_velocity=np.full(n_frames, 4.0, dtype=np.float32),
            ball_deflection_angle=np.zeros(n_frames, dtype=np.float32),
            d_H=np.full(n_frames, 1.0, dtype=np.float32),
            d_A=np.full(n_frames, 6.0, dtype=np.float32),
            d_H_top3=np.full((n_frames, 3), 1.0, dtype=np.float32),
            d_A_top3=np.full((n_frames, 3), 6.0, dtype=np.float32),
            d_H_gk=np.full(n_frames, 40.0, dtype=np.float32),
            d_A_gk=np.full(n_frames, 50.0, dtype=np.float32),
            player_mean_speed=np.full(n_frames, 3.0, dtype=np.float32),
            player_p75_speed=np.full(n_frames, 3.2, dtype=np.float32),
            player_active_frac=np.full(n_frames, 0.4, dtype=np.float32),
            missing_gaps=[],
        )

    def test_01_aerial_duel_deflection_without_opponent_control(self):
        """Scenario 1: Opponent heads the ball, but ball travels to open grass. Possession stays with kicking team."""
        N = 50
        tracking = self._base_tracking(N)
        # Frames 0..20: Barcelona passing/carrying (d_H = 0.8m)
        # Frame 20: High ball, Athletic player heads it (deflection = 45 deg, v = 8 m/s)
        # Frames 21..45: Ball travels into open space (d_min > 3.0m)
        tracking.ball_velocity[20:45] = 9.0
        tracking.ball_deflection_angle[20] = 45.0
        tracking.d_H[21:45] = 4.5
        tracking.d_H_top3[21:45] = 4.5
        tracking.d_A[21:45] = 3.5  # Athletic slightly closer, but in open grass (> 2.8m loose ball)
        tracking.d_A_top3[21:45] = 3.5

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        # Invariant: Since Athletic never established controlled possession (< 2.0m),
        # possession must remain with Barcelona throughout the loose ball phase.
        self.assertEqual(decoded[30], STATE_H, "Aerial deflection erroneously flipped possession without control!")
        self.assertEqual(decoded[40], STATE_H)

    def test_02_goalkeeper_parry_versus_catch(self):
        """Scenario 2: Goalkeeper parries shot into open box -> Attacking team retains possession."""
        N = 50
        tracking = self._base_tracking(N)
        # Barcelona attacks (x=95m, y=34m, in Away penalty box)
        tracking.ball_x_metric[:] = 95.0
        tracking.ball_x_norm[:] = 95.0 / 105.3
        # Shot at frame 15, Away GK parries at frame 20 (d_A_gk = 1.0m, v_ball = 7 m/s)
        # Ball rebounds to 4m away from GK, nearest Barca attacker at 2.5m, GK at 4.0m
        tracking.ball_velocity[20:40] = 6.0
        tracking.d_A_gk[20:23] = 1.0
        tracking.d_A_gk[23:45] = 4.0
        tracking.d_H[20:45] = 2.5
        tracking.d_H_top3[20:45] = 2.5
        tracking.d_A[20:45] = 5.0
        tracking.d_A_top3[20:45] = 5.0

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        # Attacking team (Barcelona) retains possession after parry
        self.assertEqual(decoded[30], STATE_H, "GK parry falsely awarded possession to defending team!")

    def test_03_rebound_off_post_or_defender(self):
        """Scenario 3: Shot rebounds off defender's body into contested space -> Retains attacking possession."""
        N = 40
        tracking = self._base_tracking(N)
        # Shot deflected by defender at frame 15
        tracking.ball_deflection_angle[15] = 55.0
        tracking.ball_velocity[15:30] = 8.0
        # Neither team controls immediately after block (contested space: d_H = 3.2m, d_A = 3.0m)
        tracking.d_H[15:30] = 3.2
        tracking.d_H_top3[15:30] = 3.2
        tracking.d_A[15:30] = 3.0  # Defender closer by 0.2m, but in loose ball (> 2.8m)
        tracking.d_A_top3[15:30] = 3.0

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        # Possession stays with Barcelona
        self.assertEqual(decoded[25], STATE_H, "Blocked shot rebound incorrectly transferred possession!")

    def test_04_sliding_tackle_deflection_without_control(self):
        """Scenario 4: Opponent slides and pokes ball, but fails to gain control -> Possession persists."""
        N = 50
        tracking = self._base_tracking(N)
        # Sliding tackle poke at frame 15 (v_ball = 5.0 m/s, deflection = 40 deg)
        tracking.ball_velocity[15:35] = 5.5
        tracking.ball_deflection_angle[15] = 40.0
        # Ball squirts 3.5m away; tackling opponent is on ground, Barca player pursues
        tracking.d_A[15:35] = 3.2
        tracking.d_A_top3[15:35] = 3.2
        tracking.d_H[15:35] = 3.4
        tracking.d_H_top3[15:35] = 3.4

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        self.assertEqual(decoded[25], STATE_H, "Sliding tackle deflection without control transferred possession!")

    def test_05_one_touch_rapid_passing_combination(self):
        """Scenario 5: Rapid one-touch sequence maintains continuous live possession."""
        N = 60
        tracking = self._base_tracking(N)
        # Player 1 touches at frame 10 (d_H=0.5m), passes at 8 m/s, Player 2 touches at frame 25 (d_H=0.6m),
        # passes at 9 m/s, Player 3 touches at frame 40 (d_H=0.4m)
        tracking.d_H[:] = 3.0
        tracking.d_H_top3[:] = 3.0
        tracking.d_A[:] = 6.0
        tracking.d_A_top3[:] = 6.0
        # One-touch touch moments
        tracking.d_H[10:13] = 0.5
        tracking.d_H_top3[10:13] = 0.5
        tracking.d_H[25:28] = 0.6
        tracking.d_H_top3[25:28] = 0.6
        tracking.d_H[40:43] = 0.4
        tracking.d_H_top3[40:43] = 0.4
        tracking.ball_velocity[:] = 7.5

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        # Must maintain continuous Barcelona possession throughout all 60 frames
        self.assertTrue(np.all(decoded == STATE_H), "One-touch passing combination was fragmented!")

    def test_06_failed_reception_miscontrol_in_contested_grass(self):
        """Scenario 6: Heavy touch / miscontrol rolls into contested space -> Retains original team until controlled."""
        N = 50
        tracking = self._base_tracking(N)
        # Heavy touch at frame 15: ball rolls 3.5m away into contested midfield
        tracking.d_H[15:35] = 3.5
        tracking.d_H_top3[15:35] = 3.5
        tracking.d_A[15:35] = 3.1  # Opponent is 3.1m away, neither controls
        tracking.d_A_top3[15:35] = 3.1
        tracking.ball_velocity[15:35] = 3.0

        evidence = compute_episode_evidence(tracking)
        log_B = compute_log_emissions(tracking, n_frames=N, episode_evidence=evidence, ownership_strategy="softmin_multi")
        decoded = viterbi_decode(log_B, transition_strategy="asymmetric_turnover", episode_evidence=evidence)

        self.assertEqual(decoded[25], STATE_H, "Miscontrol in contested grass falsely transferred possession!")


if __name__ == "__main__":
    unittest.main()
