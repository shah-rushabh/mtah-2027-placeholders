"""Tests for improvement #3: move each H<->A boundary to the receiver's first controlled frame.

The step is opt-in (move_turnovers_to_receiver, CLI --turnover-at-receiver), so the submitted
run is unchanged.
"""

import unittest

import numpy as np

from src.postprocess import (
    STATE_A,
    STATE_D,
    STATE_H,
    move_turnovers_to_receiver,
    rle_encode,
    team_control_mask,
)

N = 300
FAR = 20.0


def _codes(*runs):
    return np.concatenate([np.full(n, s, dtype=np.int8) for s, n in runs])


def _states(codes):
    return [(v, e - s) for s, e, v in rle_encode(codes)]


class Scene:
    """Synthetic ball and team distances. Default: ball far from everyone, moving slowly."""

    def __init__(self, n=N):
        self.d_home = np.full(n, FAR)
        self.d_away = np.full(n, FAR)
        self.speed = np.full(n, 3.0)
        self.valid = np.ones(n, dtype=bool)

    def control(self, team, start, stop, speed=2.0):
        near = self.d_home if team == STATE_H else self.d_away
        near[start:stop] = 0.8
        self.speed[start:stop] = speed

    def flight(self, start, stop, speed=15.0):
        self.speed[start:stop] = speed

    def run(self, codes, **kw):
        return move_turnovers_to_receiver(codes, self.d_home, self.d_away, self.speed, self.valid, **kw)


class TestControlMask(unittest.TestCase):
    def test_fast_ball_near_player_is_not_control(self):
        d_team = np.full(10, 0.5)
        d_other = np.full(10, FAR)
        speed = np.full(10, 15.0)
        self.assertFalse(team_control_mask(d_team, d_other, speed, np.ones(10, bool)).any())

    def test_opponent_nearer_is_not_control(self):
        d_team = np.full(10, 1.0)
        d_other = np.full(10, 0.5)
        self.assertFalse(team_control_mask(d_team, d_other, np.full(10, 1.0), np.ones(10, bool)).any())

    def test_two_frame_touch_is_too_short(self):
        d_team = np.full(10, FAR)
        d_team[4:6] = 0.5
        mask = team_control_mask(d_team, np.full(10, FAR), np.full(10, 1.0), np.ones(10, bool))
        self.assertFalse(mask.any())
        d_team[4:7] = 0.5
        mask = team_control_mask(d_team, np.full(10, FAR), np.full(10, 1.0), np.ones(10, bool))
        self.assertEqual(list(np.flatnonzero(mask)), [4, 5, 6])

    def test_missing_ball_or_player_is_not_control(self):
        d_team = np.full(10, 0.5)
        d_team[0:5] = np.nan
        valid = np.ones(10, bool)
        valid[5:] = False
        mask = team_control_mask(d_team, np.full(10, np.nan), np.full(10, 1.0), valid)
        self.assertFalse(mask.any())


class TestTurnoverAtReceiver(unittest.TestCase):
    def test_early_boundary_moves_to_reception(self):
        # Home passes at 90, the ball flies until 115, Away receives at 115.
        # The decoder switched at 100, in mid-flight.
        s = Scene()
        s.control(STATE_H, 60, 90)
        s.flight(90, 115)
        s.control(STATE_A, 115, 160)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_H, 115), (STATE_A, 185)])

    def test_late_boundary_moves_back_to_reception(self):
        # Away controls from 85, the decoder switched only at 100.
        s = Scene()
        s.control(STATE_H, 50, 75)
        s.flight(75, 85)
        s.control(STATE_A, 85, 140)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_H, 85), (STATE_A, 215)])

    def test_old_team_regains_control_before_boundary(self):
        # Away touches at 70, Home recovers 80-95, Away receives for real at 108.
        s = Scene()
        s.control(STATE_A, 70, 76)
        s.control(STATE_H, 80, 95)
        s.flight(95, 108)
        s.control(STATE_A, 108, 150)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_H, 108), (STATE_A, 192)])

    def test_reception_beyond_two_seconds_does_not_move(self):
        s = Scene()
        s.control(STATE_A, 160, 200)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        out = s.run(codes)
        np.testing.assert_array_equal(out, codes)

    def test_fly_by_does_not_count_as_reception(self):
        # The ball flies past an Away player at 105 at 15 m/s; Away controls only at 130.
        s = Scene()
        s.flight(95, 130)
        s.d_away[103:108] = 0.5
        s.control(STATE_A, 130, 170)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_H, 130), (STATE_A, 170)])

    def test_move_keeps_two_second_spells(self):
        # The Away spell is 60 frames. A move to 120 would leave 40 frames, so it is refused.
        s = Scene()
        s.control(STATE_A, 120, 160)
        codes = _codes((STATE_H, 100), (STATE_A, 60), (STATE_H, 140))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_H, 100), (STATE_A, 60), (STATE_H, 140)])

    def test_short_spell_is_never_shortened(self):
        s = Scene()
        s.control(STATE_A, 85, 140)
        codes = _codes((STATE_H, 100), (STATE_A, 30), (STATE_H, 170))
        out = s.run(codes)
        # The Away spell may grow back to 85, but must not lose frames at its far end.
        segs = rle_encode(out)
        self.assertEqual(segs[1][2], STATE_A)
        self.assertEqual(segs[1][1], 130)
        self.assertGreaterEqual(segs[1][1] - segs[1][0], 30)

    def test_boundary_next_to_dead_ball_does_not_move(self):
        s = Scene()
        s.control(STATE_A, 115, 160)
        codes = _codes((STATE_H, 100), (STATE_D, 10), (STATE_A, 190))
        out = s.run(codes)
        np.testing.assert_array_equal(out, codes)

    def test_no_control_evidence_leaves_boundary(self):
        s = Scene()
        s.valid[:] = False
        s.control(STATE_A, 110, 150)
        codes = _codes((STATE_H, 100), (STATE_A, 200))
        np.testing.assert_array_equal(s.run(codes), codes)

    def test_away_to_home_is_symmetric(self):
        s = Scene()
        s.control(STATE_A, 60, 90)
        s.flight(90, 115)
        s.control(STATE_H, 115, 160)
        codes = _codes((STATE_A, 100), (STATE_H, 200))
        out = s.run(codes)
        self.assertEqual(_states(out), [(STATE_A, 115), (STATE_H, 185)])

    def test_labels_and_frame_count_are_preserved(self):
        s = Scene()
        s.control(STATE_A, 115, 160)
        s.control(STATE_H, 215, 260)
        codes = _codes((STATE_H, 100), (STATE_A, 100), (STATE_H, 100))
        out = s.run(codes)
        self.assertEqual(len(out), len(codes))
        self.assertEqual([v for v, _ in _states(out)], [STATE_H, STATE_A, STATE_H])
        self.assertEqual(_states(out), [(STATE_H, 115), (STATE_A, 100), (STATE_H, 85)])


if __name__ == "__main__":
    unittest.main()
