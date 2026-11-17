"""Tests for the two opt-in changes: turnover confirmation and the in-flight barrier.

Both are off by default, so the submitted run is unchanged.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.postprocess import (
    STATE_A,
    STATE_D,
    STATE_H,
    apply_symmetric_dwell_pruning,
    rle_encode,
)
from src.viterbi import viterbi_decode


def _codes(*runs):
    return np.concatenate([np.full(n, s, dtype=np.int8) for s, n in runs])


def _states(codes):
    return [(v, e - s) for s, e, v in rle_encode(codes)]


class TestTurnoverConfirmation(unittest.TestCase):
    def test_short_sandwich_goes_back_to_holding_team(self):
        codes = _codes((STATE_H, 100), (STATE_A, 40), (STATE_H, 100))
        out = apply_symmetric_dwell_pruning(codes, confirm_turnover_frames=50)
        self.assertEqual(_states(out), [(STATE_H, 240)])

    def test_short_sandwich_kept_when_option_off(self):
        codes = _codes((STATE_H, 100), (STATE_A, 40), (STATE_H, 100))
        out = apply_symmetric_dwell_pruning(codes)
        self.assertEqual(_states(out), [(STATE_H, 100), (STATE_A, 40), (STATE_H, 100)])

    def test_long_opponent_spell_is_kept(self):
        codes = _codes((STATE_H, 100), (STATE_A, 60), (STATE_H, 100))
        out = apply_symmetric_dwell_pruning(codes, confirm_turnover_frames=50)
        self.assertEqual(_states(out), [(STATE_H, 100), (STATE_A, 60), (STATE_H, 100)])

    def test_short_spell_before_dead_ball_is_not_a_sandwich(self):
        codes = _codes((STATE_H, 100), (STATE_A, 40), (STATE_D, 100))
        out = apply_symmetric_dwell_pruning(codes, confirm_turnover_frames=50)
        self.assertEqual(_states(out), [(STATE_H, 100), (STATE_A, 40), (STATE_D, 100)])

    def test_intentional_kick_no_longer_protects_short_spell(self):
        codes = _codes((STATE_H, 100), (STATE_A, 20), (STATE_H, 100))
        kick_a = np.zeros(len(codes), dtype=bool)
        kick_a[105] = True
        kept = apply_symmetric_dwell_pruning(codes, intentional_action_A=kick_a)
        self.assertEqual(_states(kept), [(STATE_H, 100), (STATE_A, 20), (STATE_H, 100)])
        merged = apply_symmetric_dwell_pruning(codes, intentional_action_A=kick_a, confirm_turnover_frames=50)
        self.assertEqual(_states(merged), [(STATE_H, 220)])

    def test_restart_inside_short_spell_keeps_it(self):
        codes = _codes((STATE_H, 100), (STATE_A, 40), (STATE_H, 100))
        p_restart = np.zeros(len(codes))
        p_restart[110] = 0.9
        out = apply_symmetric_dwell_pruning(codes, p_restart=p_restart, confirm_turnover_frames=50)
        self.assertEqual(_states(out), [(STATE_H, 100), (STATE_A, 40), (STATE_H, 100)])

    def test_both_teams_symmetric(self):
        codes = _codes((STATE_A, 100), (STATE_H, 40), (STATE_A, 100))
        out = apply_symmetric_dwell_pruning(codes, confirm_turnover_frames=50)
        self.assertEqual(_states(out), [(STATE_A, 240)])

    def test_idempotent(self):
        codes = _codes((STATE_H, 100), (STATE_A, 40), (STATE_H, 70), (STATE_A, 45), (STATE_H, 100), (STATE_A, 200))
        once = apply_symmetric_dwell_pruning(codes, confirm_turnover_frames=50)
        twice = apply_symmetric_dwell_pruning(once, confirm_turnover_frames=50)
        np.testing.assert_array_equal(once, twice)
        self.assertEqual(_states(once), [(STATE_H, 355), (STATE_A, 200)])


class TestFlightBarrier(unittest.TestCase):
    """Home holds the ball. For 20 frames in fast flight the emissions lean Away by 0.8 per frame."""

    def setUp(self):
        n = 100
        log_e = np.full((n, 3), -10.0)
        log_e[:, STATE_H] = 0.0
        log_e[:, STATE_A] = -5.0
        log_e[40:60, STATE_H] = -0.8
        log_e[40:60, STATE_A] = 0.0
        self.log_e = log_e
        p_fast = np.zeros(n, dtype=np.float32)
        p_fast[40:61] = 0.9
        self.evidence = SimpleNamespace(
            p_dead=np.zeros(n),
            p_restart_candidate=np.zeros(n),
            restart_team=np.full(n, -1),
            p_fast_transit=p_fast,
            is_deflection=np.zeros(n, dtype=bool),
        )

    def test_without_barrier_ball_changes_team_in_flight(self):
        path = viterbi_decode(self.log_e, episode_evidence=self.evidence)
        self.assertTrue(np.all(path[40:60] == STATE_A))

    def test_barrier_keeps_holding_team_during_flight(self):
        path = viterbi_decode(self.log_e, episode_evidence=self.evidence, flight_barrier=True)
        self.assertTrue(np.all(path == STATE_H))

    def test_deflection_lifts_barrier(self):
        self.evidence.is_deflection[40] = True
        self.evidence.is_deflection[60] = True
        path = viterbi_decode(self.log_e, episode_evidence=self.evidence, flight_barrier=True)
        self.assertTrue(np.all(path[40:60] == STATE_A))

    def test_barrier_off_by_default_matches_explicit_off(self):
        default_path = viterbi_decode(self.log_e, episode_evidence=self.evidence)
        off_path = viterbi_decode(self.log_e, episode_evidence=self.evidence, flight_barrier=False)
        np.testing.assert_array_equal(default_path, off_path)


if __name__ == "__main__":
    unittest.main()
