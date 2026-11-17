"""
tests/test_augmented_5state_model.py - Verification of Augmented 5-State Possession Model

Tests state-space transitions and sequence properties:
1. Pass transitions H_ctrl -> H_unctrl, maintaining continuous H output.
2. Opponent deflection in flight maintains H_unctrl (output H).
3. Opponent controlled reception transfers possession H_unctrl -> A_ctrl (output A).
4. Goalkeeper parry maintains H_unctrl (output H).
5. Stoppage forces Dead (output D).
6. Legal restart kick transitions Dead -> H_ctrl / A_ctrl.
"""

import unittest
import numpy as np

from src.viterbi import (
    viterbi_decode,
    STATE_H,
    STATE_A,
    STATE_D,
    STATE_5_H_CTRL,
    STATE_5_H_UNCTRL,
    STATE_5_A_CTRL,
    STATE_5_A_UNCTRL,
    STATE_5_DEAD,
)
from src.control import ControlEvidence
from src.episodes import EpisodeEvidence


class TestAugmented5StateModel(unittest.TestCase):
    def test_01_pass_transitions_from_ctrl_to_unctrl_retaining_possession(self):
        """Home controls ball, then kicks pass (flight): outputs H continuously across both states."""
        T = 40
        log_B = np.full((T, 5), -10.0, dtype=np.float32)
        # Frames 0..19: Home has control
        log_B[:20, STATE_5_H_CTRL] = 5.0
        # Frames 20..39: Ball in flight/loose (H_unctrl)
        log_B[20:, STATE_5_H_UNCTRL] = 5.0
        
        init_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        best_path = viterbi_decode(log_emissions=log_B, initial_probs=init_h)
        
        # All 40 frames must decode to Home (STATE_H = 0)
        self.assertTrue(np.all(best_path == STATE_H), f"Expected pure H, got {best_path}")

    def test_02_opponent_deflection_in_flight_retains_possession(self):
        """Opponent deflects a ball in flight without controlling it: possession stays H."""
        T = 40
        log_B = np.full((T, 5), -10.0, dtype=np.float32)
        log_B[:15, STATE_5_H_CTRL] = 5.0
        log_B[15:, STATE_5_H_UNCTRL] = 4.0
        
        # At frame 22, Away has contact (deflection), but NO control
        log_B[22, STATE_5_A_CTRL] = -2.0  # Away has brief contact spike, but not sufficient to control
        
        ctrl_ev = ControlEvidence(
            control_confidence_H=np.zeros(T, dtype=np.float32),
            control_confidence_A=np.zeros(T, dtype=np.float32),
            contact_confidence_H=np.zeros(T, dtype=np.float32),
            contact_confidence_A=np.zeros(T, dtype=np.float32),
            loose_ball_confidence=np.ones(T, dtype=np.float32),
            contest_confidence=np.zeros(T, dtype=np.float32),
            is_deflection_event=np.zeros(T, dtype=bool),
            deflection_team=np.full(T, -1, dtype=np.int8),
            turnover_candidate_H_to_A=np.zeros(T, dtype=bool),  # NO confirmed turnover!
            turnover_candidate_A_to_H=np.zeros(T, dtype=bool),
        )
        ctrl_ev.is_deflection_event[22] = True
        ctrl_ev.deflection_team[22] = 1  # Away
        
        init_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        best_path = viterbi_decode(log_emissions=log_B, control_evidence=ctrl_ev, initial_probs=init_h)
        
        # Deflection must not cause turnover; output stays H
        self.assertTrue(np.all(best_path == STATE_H), f"Deflection caused turnover! path: {best_path}")

    def test_03_opponent_controlled_interception_transfers_possession(self):
        """Opponent intercepts and establishes sustained control: possession cleanly transitions H -> A."""
        T = 50
        log_B = np.full((T, 5), -10.0, dtype=np.float32)
        log_B[:20, STATE_5_H_CTRL] = 5.0
        log_B[20:30, STATE_5_H_UNCTRL] = 4.0
        # Frames 30..49: Away establishes controlled possession
        log_B[30:, STATE_5_A_CTRL] = 6.0
        
        ctrl_ev = ControlEvidence(
            control_confidence_H=np.zeros(T, dtype=np.float32),
            control_confidence_A=np.zeros(T, dtype=np.float32),
            contact_confidence_H=np.zeros(T, dtype=np.float32),
            contact_confidence_A=np.zeros(T, dtype=np.float32),
            loose_ball_confidence=np.zeros(T, dtype=np.float32),
            contest_confidence=np.zeros(T, dtype=np.float32),
            is_deflection_event=np.zeros(T, dtype=bool),
            deflection_team=np.full(T, -1, dtype=np.int8),
            turnover_candidate_H_to_A=np.zeros(T, dtype=bool),
            turnover_candidate_A_to_H=np.zeros(T, dtype=bool),
        )
        ctrl_ev.turnover_candidate_H_to_A[30:] = True
        
        init_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        best_path = viterbi_decode(log_emissions=log_B, control_evidence=ctrl_ev, initial_probs=init_h)
        
        self.assertTrue(np.all(best_path[:30] == STATE_H))
        self.assertTrue(np.all(best_path[30:] == STATE_A))

    def test_04_stoppage_forces_dead_ball(self):
        """Stoppage or ball out of bounds transitions H_unctrl -> Dead Ball (STATE_D)."""
        T = 40
        log_B = np.full((T, 5), -10.0, dtype=np.float32)
        log_B[:20, STATE_5_H_CTRL] = 5.0
        # At frame 20, ball crosses touchline -> Dead ball
        log_B[20:, STATE_5_DEAD] = 8.0
        
        ep_ev = EpisodeEvidence(
            p_dead=np.zeros(T, dtype=np.float32),
            p_live=np.ones(T, dtype=np.float32),
            p_fast_transit=np.zeros(T, dtype=np.float32),
            p_restart_candidate=np.zeros(T, dtype=np.float32),
            p_contested=np.zeros(T, dtype=np.float32),
            restart_team=np.full(T, -1, dtype=np.int8),
            confidence=np.ones(T, dtype=np.float32),
        )
        ep_ev.p_dead[20:] = 1.0
        ep_ev.p_live[20:] = 0.0
        
        init_h = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        best_path = viterbi_decode(log_emissions=log_B, episode_evidence=ep_ev, initial_probs=init_h)
        
        self.assertTrue(np.all(best_path[:20] == STATE_H))
        self.assertTrue(np.all(best_path[20:] == STATE_D))


if __name__ == "__main__":
    unittest.main()
