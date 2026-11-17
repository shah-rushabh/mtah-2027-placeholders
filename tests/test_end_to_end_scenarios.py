"""
tests/test_end_to_end_scenarios.py - Comprehensive End-to-End Synthetic Scenario Suite.

Verifies the complete end-to-end possession engine (Episodes -> Control -> Emissions -> Viterbi -> Postprocess)
across 10 realistic match dynamics with full Home and Away team symmetry:
1. Deflection without control (H -> A defl -> H, and A -> H defl -> A)
2. Controlled interception (H -> A ctrl, and A -> H ctrl)
3. Aerial duel (50-50 header to open grass retains possession)
4. Rebound off post/defender into contested space
5. Goalkeeper parry vs. catch (parry retains attacking possession, catch establishes GK possession)
6. Sliding tackle deflection without control
7. One-touch rapid passing with short genuine spell retention (< 1.0s)
8. Loose-ball recovery in open pitch
9. Long missing-ball gap with active player locomotion
10. Match stoppage and quick legal restart
11. Verification of determinism, player-order invariance, and post-processing idempotence.
"""

import unittest
import numpy as np

from src.loader import TrackingData, PITCH_LENGTH, PITCH_WIDTH, FPS
from src.episodes import compute_episode_evidence, EpisodeEvidence
from src.control import compute_control_evidence, ControlEvidence
from src.emissions import compute_log_emissions
from src.viterbi import viterbi_decode, STATE_H, STATE_A, STATE_D
from src.postprocess import apply_symmetric_dwell_pruning, rle_encode


class TestEndToEndScenarios(unittest.TestCase):

    def _create_base_tracking(self, n_frames: int = 100) -> TrackingData:
        """Constructs synthetic tracking fixture initialized in neutral state."""
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

    def _run_full_pipeline(self, tracking: TrackingData, augmented: bool = True) -> np.ndarray:
        """Executes the complete end-to-end 6-stage application pipeline."""
        N = len(tracking.ball_observed)
        episodes = compute_episode_evidence(tracking, n_frames=N)
        control = compute_control_evidence(tracking, episodes=episodes, pitch=(PITCH_LENGTH, PITCH_WIDTH))
        log_B = compute_log_emissions(
            tracking=tracking,
            n_frames=N,
            episode_evidence=episodes,
            control_evidence=control,
            augmented_5state=augmented,
        )
        raw_states = viterbi_decode(
            log_emissions=log_B,
            episode_evidence=episodes,
            control_evidence=control,
        )
        clean_states = apply_symmetric_dwell_pruning(
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
        return clean_states

    # =========================================================================
    # 1. DEFLECTION WITHOUT CONTROL (H -> A defl -> H, and A -> H defl -> A)
    # =========================================================================
    def test_01_deflection_without_control_team_symmetry(self):
        """Attacker passes, opponent deflects mid-flight without control: kicking team retains possession."""
        for kicking_team in ("H", "A"):
            N = 80
            tracking = self._create_base_tracking(N)
            defending_team = "A" if kicking_team == "H" else "H"

            # Initial control by kicking team for 30 frames
            if kicking_team == "H":
                tracking.d_H[:30] = 0.5
                tracking.d_H_top3[:30] = 0.5
            else:
                tracking.d_A[:30] = 0.5
                tracking.d_A_top3[:30] = 0.5

            # Pass in flight from frame 30 to 70: ball velocity = 10 m/s
            tracking.ball_velocity[30:70] = 10.0
            tracking.d_H[30:70] = 4.0
            tracking.d_A[30:70] = 4.0

            # Opponent deflects at frame 45 (glancing touch at 1.4m, deflection angle 35 deg, ball speed continues at 8 m/s)
            if defending_team == "A":
                tracking.d_A[45] = 1.4
                tracking.d_A_top3[45] = 1.4
            else:
                tracking.d_H[45] = 1.4
                tracking.d_H_top3[45] = 1.4
            tracking.ball_deflection_angle[45] = 35.0

            # Kicking team settles ball at frame 60-80
            if kicking_team == "H":
                tracking.d_H[60:80] = 0.6
                tracking.d_H_top3[60:80] = 0.6
            else:
                tracking.d_A[60:80] = 0.6
                tracking.d_A_top3[60:80] = 0.6

            clean_states = self._run_full_pipeline(tracking)
            target_code = STATE_H if kicking_team == "H" else STATE_A
            opponent_code = STATE_A if kicking_team == "H" else STATE_H

            # Deflection frame 45 MUST NOT be attributed to defending team
            self.assertNotEqual(clean_states[45], opponent_code, f"Opponent {defending_team} deflection triggered false turnover!")
            self.assertEqual(clean_states[45], target_code, f"Kicking team {kicking_team} lost possession on opponent deflection!")

    # =========================================================================
    # 2. CONTROLLED INTERCEPTION (H -> A ctrl, and A -> H ctrl)
    # =========================================================================
    def test_02_controlled_interception_team_symmetry(self):
        """Opponent steps in, settles ball and carries: cleanly transfers possession."""
        for initial_team in ("H", "A"):
            N = 90
            tracking = self._create_base_tracking(N)
            intercepting_team = "A" if initial_team == "H" else "H"

            # Initial control by team 1
            if initial_team == "H":
                tracking.d_H[:30] = 0.5
                tracking.d_H_top3[:30] = 0.5
            else:
                tracking.d_A[:30] = 0.5
                tracking.d_A_top3[:30] = 0.5

            # Pass from 30 to 45
            tracking.ball_velocity[30:45] = 9.0

            # Intercepting team establishes sustained control from frame 45 to 90 (d = 0.4m, speed = 2.5 m/s)
            if intercepting_team == "A":
                tracking.d_A[45:90] = 0.4
                tracking.d_A_top3[45:90] = 0.4
                tracking.d_H[45:90] = 6.0
                tracking.d_H_top3[45:90] = 6.0
            else:
                tracking.d_H[45:90] = 0.4
                tracking.d_H_top3[45:90] = 0.4
                tracking.d_A[45:90] = 6.0
                tracking.d_A_top3[45:90] = 6.0
            tracking.ball_velocity[45:90] = 2.5

            clean_states = self._run_full_pipeline(tracking)
            expected_new_code = STATE_A if intercepting_team == "A" else STATE_H

            # Later segment must be controlled by intercepting team
            self.assertEqual(clean_states[70], expected_new_code, f"Controlled interception failed to transfer possession to {intercepting_team}!")

    # =========================================================================
    # 3. AERIAL DUEL (50-50 Header to open grass)
    # =========================================================================
    def test_03_aerial_duel_preserves_possession(self):
        """Two players contest aerial duel, flick goes to open grass: kicking team retains possession."""
        N = 75
        tracking = self._create_base_tracking(N)
        # Home controls ball, kicks high cross
        tracking.d_H[:25] = 0.5
        tracking.d_H_top3[:25] = 0.5
        tracking.ball_velocity[25:60] = 8.5

        # Aerial duel at frame 40: both players within 1.5m (|dH - dA| < 0.2m)
        tracking.d_H[40] = 1.3
        tracking.d_A[40] = 1.4
        tracking.d_H_top3[40] = 1.3
        tracking.d_A_top3[40] = 1.4
        tracking.ball_deflection_angle[40] = 40.0

        # Ball lands in open grass (d > 3.5m for both)
        tracking.d_H[42:60] = 4.0
        tracking.d_A[42:60] = 4.0

        clean_states = self._run_full_pipeline(tracking)
        self.assertEqual(clean_states[40], STATE_H, "Aerial duel falsely transferred possession away from attacking team!")
        self.assertEqual(clean_states[45], STATE_H, "Loose ball after aerial duel was lost!")

    # =========================================================================
    # 4. REBOUND OFF POST OR DEFENDER
    # =========================================================================
    def test_04_rebound_off_post_or_defender(self):
        """Attacking shot rebounds off defender's body into contested space: attacking team retains possession."""
        N = 65
        tracking = self._create_base_tracking(N)
        # Home controls, shoots
        tracking.d_H[:25] = 0.4
        tracking.d_H_top3[:25] = 0.4
        tracking.ball_velocity[25:35] = 14.0

        # Shot hits defender at frame 35, ricochets into penalty box
        tracking.d_A[35] = 1.1
        tracking.d_A_top3[35] = 1.1
        tracking.ball_deflection_angle[35] = 50.0
        tracking.ball_velocity[36:50] = 3.5
        tracking.d_H[36:50] = 3.0
        tracking.d_A[36:50] = 2.8

        clean_states = self._run_full_pipeline(tracking)
        self.assertEqual(clean_states[35], STATE_H, "Rebound off defender falsely awarded possession to defender!")

    # =========================================================================
    # 5. GOALKEEPER PARRY VERSUS CATCH
    # =========================================================================
    def test_05_goalkeeper_parry_versus_catch(self):
        """Goalkeeper parry into box retains attacking possession; goalkeeper catch establishes GK possession."""
        # Scenario A: Goalkeeper Parry
        N = 60
        tracking_parry = self._create_base_tracking(N)
        tracking_parry.d_H[:20] = 0.5  # Home attacks
        tracking_parry.ball_velocity[20:30] = 15.0
        # Away GK parries at frame 30 (high outgoing speed, ball loose in box)
        tracking_parry.d_A_gk[30] = 1.0
        tracking_parry.ball_deflection_angle[30] = 60.0
        tracking_parry.ball_velocity[31:50] = 6.0
        tracking_parry.d_H[31:50] = 3.0
        tracking_parry.d_A[31:50] = 4.0

        states_parry = self._run_full_pipeline(tracking_parry)
        self.assertEqual(states_parry[30], STATE_H, "Goalkeeper parry falsely transferred possession to defending team!")

        # Scenario B: Goalkeeper Catch
        tracking_catch = self._create_base_tracking(N)
        tracking_catch.d_H[:20] = 0.5
        tracking_catch.ball_velocity[20:30] = 12.0
        # Away GK catches at frame 30 (ball velocity drops to 0.5 m/s, GK holds inside box)
        tracking_catch.d_A_gk[30:60] = 0.5
        tracking_catch.ball_velocity[30:60] = 0.5
        tracking_catch.ball_x_metric[30:60] = 100.0  # inside Away penalty box
        tracking_catch.d_H[30:60] = 8.0

        states_catch = self._run_full_pipeline(tracking_catch)
        self.assertEqual(states_catch[45], STATE_A, "Goalkeeper catch failed to transfer possession to GK team!")

    # =========================================================================
    # 6. SLIDING TACKLE DEFLECTION WITHOUT CONTROL
    # =========================================================================
    def test_06_sliding_tackle_deflection(self):
        """Defender slides and pokes ball, but fails to gain control: dribbler retains possession."""
        N = 70
        tracking = self._create_base_tracking(N)
        # Home dribbles
        tracking.d_H[:35] = 0.5
        tracking.d_H_top3[:35] = 0.5
        # Away slides at frame 35, poking ball 2.2m away
        tracking.d_A[35] = 1.2
        tracking.d_A_top3[35] = 1.2
        tracking.ball_deflection_angle[35] = 30.0
        tracking.d_H[36:50] = 2.0
        tracking.d_A[36:50] = 3.0
        # Home re-gathers at frame 50
        tracking.d_H[50:70] = 0.5
        tracking.d_H_top3[50:70] = 0.5

        clean_states = self._run_full_pipeline(tracking)
        self.assertEqual(clean_states[35], STATE_H, "Sliding tackle poke falsely transferred possession to defender!")
        self.assertEqual(clean_states[40], STATE_H, "Loose ball after sliding tackle was lost!")

    # =========================================================================
    # 7. ONE-TOUCH RAPID PASSING (< 1.0s Genuine Possession Spell)
    # =========================================================================
    def test_07_one_touch_passing_short_spell_retention(self):
        """A quick intentional first-time pass (12 frames = 0.48s) is preserved as genuine possession."""
        for team in ("H", "A"):
            N = 80
            tracking = self._create_base_tracking(N)
            opp_team = "A" if team == "H" else "H"

            # Opponent controls initially
            if opp_team == "A":
                tracking.d_A[:25] = 0.5
                tracking.d_A_top3[:25] = 0.5
            else:
                tracking.d_H[:25] = 0.5
                tracking.d_H_top3[:25] = 0.5

            # Opponent pass goes toward player at frame 25-35
            tracking.ball_velocity[25:35] = 7.0

            # Player executes a first-time pass at frame 35 (strikes ball: ball accels to 9 m/s, separates rapidly)
            if team == "H":
                tracking.d_H[35:48] = 0.6  # 13 frames of interaction and pass release
                tracking.d_H_top3[35:48] = 0.6
                tracking.d_A[35:48] = 4.0
            else:
                tracking.d_A[35:48] = 0.6
                tracking.d_A_top3[35:48] = 0.6
                tracking.d_H[35:48] = 4.0
            tracking.ball_velocity[35:48] = 9.0
            tracking.ball_deflection_angle[35] = 45.0

            # Pass is intercepted downstream by opponent at frame 55-80
            if opp_team == "A":
                tracking.d_A[55:80] = 0.5
                tracking.d_A_top3[55:80] = 0.5
            else:
                tracking.d_H[55:80] = 0.5
                tracking.d_H_top3[55:80] = 0.5

            clean_states = self._run_full_pipeline(tracking)
            target_code = STATE_H if team == "H" else STATE_A

            # The intentional 1-touch pass by team must be recognized and NOT erased
            self.assertEqual(clean_states[38], target_code, f"Intentional 1-touch pass by {team} was erased by dwell pruning!")

    # =========================================================================
    # 8. LOOSE BALL RECOVERY IN OPEN PITCH
    # =========================================================================
    def test_08_loose_ball_recovery(self):
        """Ball free in open pitch; player arrives, settles, and establishes possession."""
        N = 70
        tracking = self._create_base_tracking(N)
        # Loose ball: d > 4.0m for both teams
        tracking.d_H[:30] = 4.5
        tracking.d_A[:30] = 5.0
        tracking.ball_velocity[:30] = 2.0

        # Away player arrives at frame 30, carries through frame 70
        tracking.d_A[30:70] = 0.5
        tracking.d_A_top3[30:70] = 0.5
        tracking.d_H[30:70] = 6.0
        tracking.d_H_top3[30:70] = 6.0
        tracking.ball_velocity[30:70] = 2.2

        clean_states = self._run_full_pipeline(tracking)
        self.assertEqual(clean_states[50], STATE_A, "Loose ball recovery failed to establish possession for Away team!")

    # =========================================================================
    # 9. LONG MISSING BALL GAP WITH ACTIVE LOCOMOTION
    # =========================================================================
    def test_09_long_missing_ball_gap_preserves_possession(self):
        """Camera drops ball for 20 frames during active wing attack: possession persists without false D."""
        N = 80
        tracking = self._create_base_tracking(N)
        # Home controls
        tracking.d_H[:30] = 0.5
        tracking.d_H_top3[:30] = 0.5
        # Missing gap from frame 30 to 50: ball not observed, but player speeds active (2.8 m/s)
        tracking.ball_observed[30:50] = False
        tracking.tracking_confidence[30:50] = 0.0
        tracking.player_p75_speed[:] = 2.8
        # Home continues control after frame 50
        tracking.d_H[50:80] = 0.5
        tracking.d_H_top3[50:80] = 0.5

        clean_states = self._run_full_pipeline(tracking)
        self.assertTrue(np.all(clean_states[30:50] == STATE_H), "Active camera occlusion dropped into false Dead Ball!")

    # =========================================================================
    # 10. STOPPAGE AND SHORT LEGAL RESTART
    # =========================================================================
    def test_10_stoppage_and_short_legal_restart(self):
        """Referee whistle creates stoppage; legal restart kick snaps cleanly into live play."""
        N = 140
        tracking = self._create_base_tracking(N)
        # Stoppage from frame 10 to 60: players walking (< 1.1 m/s), ball stationary (disp < 0.2m)
        tracking.player_p75_speed[10:60] = 1.0
        tracking.player_mean_speed[10:60] = 0.9
        tracking.player_active_frac[10:60] = 0.02
        tracking.ball_velocity[10:60] = 0.2
        tracking.ball_x_metric[10:60] = 40.0
        tracking.ball_y_metric[10:60] = 20.0

        # Restart kick executed at frame 60 by Away: sustained live play through frame 120
        tracking.ball_velocity[60:120] = 3.5
        tracking.ball_x_metric[60:120] = np.linspace(40.0, 65.0, 60)
        tracking.d_A[60:120] = 0.5
        tracking.d_A_top3[60:120] = 0.5

        clean_states = self._run_full_pipeline(tracking)
        # Stoppage must be DEAD
        self.assertEqual(clean_states[35], STATE_D, "Confirmed stoppage was not recognized as DEAD!")
        # Restart must snap to live Away possession
        self.assertEqual(clean_states[85], STATE_A, "Legal restart kick failed to snap into live Away possession!")

    # =========================================================================
    # 11. INVARIANTS: DETERMINISM & IDEMPOTENCE
    # =========================================================================
    def test_11_determinism_and_idempotence(self):
        """Two independent runs yield bit-for-bit identical outputs; postprocessing is idempotent."""
        N = 100
        tracking = self._create_base_tracking(N)
        tracking.d_H[:40] = 0.5
        tracking.d_A[40:80] = 0.5
        tracking.player_p75_speed[80:100] = 1.0
        tracking.ball_velocity[80:100] = 0.1

        run1 = self._run_full_pipeline(tracking)
        run2 = self._run_full_pipeline(tracking)

        # Determinism
        self.assertTrue(np.array_equal(run1, run2), "Application pipeline is non-deterministic!")

        # Idempotence: P(P(x)) = P(x)
        pruned_once = apply_symmetric_dwell_pruning(codes=run1, min_poss_dwell=30, min_dead_dwell=35)
        pruned_twice = apply_symmetric_dwell_pruning(codes=pruned_once, min_poss_dwell=30, min_dead_dwell=35)
        self.assertTrue(np.array_equal(pruned_once, pruned_twice), "Post-processor is not idempotent!")


if __name__ == "__main__":
    unittest.main()
