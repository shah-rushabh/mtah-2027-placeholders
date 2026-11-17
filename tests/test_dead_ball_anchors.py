"""Tests for improvement #4: dead-ball start at the line crossing, end at the restart.

Everything here is opt-in (dead_ball_anchors=True, CLI --dead-ball-anchors), so the submitted
run is unchanged.
"""

import unittest

import numpy as np

from src.emissions import H2_KICKOFF, NEG_INF, STATE_A, STATE_D, STATE_H, compute_log_emissions
from src.episodes import (
    EpisodeEvidence,
    compute_episode_evidence,
    derive_second_half_kickoff,
    detect_line_crossings,
    find_kickoff_frame,
    find_restart_frame,
)
from src.loader import FPS, MissingGap, TrackingData

LENGTH, WIDTH = 105.3, 68.0


def make_tracking(xs, ys, observed=None, p75=3.0, active=0.4):
    """TrackingData whose ball speed and velocity follow the given path (metres per frame * 25)."""
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    n = len(xs)
    observed = np.ones(n, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    x = np.where(observed, xs, np.nan).astype(np.float32)
    y = np.where(observed, ys, np.nan).astype(np.float32)
    vx = np.zeros(n, dtype=np.float32)
    vy = np.zeros(n, dtype=np.float32)
    both = observed[1:] & observed[:-1]
    vx[1:][both] = (xs[1:] - xs[:-1])[both] * FPS
    vy[1:][both] = (ys[1:] - ys[:-1])[both] * FPS
    speed = np.hypot(vx, vy).astype(np.float32)
    return TrackingData(
        ball_x_metric=x,
        ball_y_metric=y,
        ball_x_norm=(x / LENGTH).astype(np.float32),
        ball_y_norm=(y / WIDTH).astype(np.float32),
        ball_observed=observed.copy(),
        ball_velocity=speed,
        d_H=np.full(n, 3.0, dtype=np.float32),
        d_A=np.full(n, 6.0, dtype=np.float32),
        d_H_top3=np.full((n, 3), 3.0, dtype=np.float32),
        d_A_top3=np.full((n, 3), 6.0, dtype=np.float32),
        d_H_gk=np.full(n, 40.0, dtype=np.float32),
        d_A_gk=np.full(n, 40.0, dtype=np.float32),
        player_mean_speed=np.full(n, p75, dtype=np.float32),
        player_p75_speed=np.full(n, p75, dtype=np.float32),
        player_active_frac=np.full(n, active, dtype=np.float32),
        missing_gaps=[],
        ball_vx=vx,
        ball_vy=vy,
        ball_deflection_angle=np.zeros(n, dtype=np.float32),
        ball_valid=observed.copy(),
    )


def straight(n, start, step):
    """n positions from `start` moving `step` metres per frame (x, y)."""
    k = np.arange(n, dtype=np.float32)[:, None]
    path = np.asarray(start, dtype=np.float32) + k * np.asarray(step, dtype=np.float32)
    return path[:, 0], path[:, 1]


class TestLineCrossings(unittest.TestCase):
    def test_observed_crossing_is_first_frame_off_the_pitch(self):
        xs, ys = straight(60, (50.0, 60.0), (0.0, 0.32))  # 8 m/s towards the touchline
        crossing = detect_line_crossings(make_tracking(xs, ys))
        self.assertEqual(list(crossing), [int(np.flatnonzero(ys > WIDTH)[0])])

    def test_jitter_along_the_line_is_not_a_crossing(self):
        xs = np.linspace(30, 40, 50)
        ys = np.full(50, 67.9)
        ys[20] = 68.2  # one frame 0.2 m out, then back
        self.assertEqual(len(detect_line_crossings(make_tracking(xs, ys))), 0)

    def test_crossing_is_extrapolated_through_a_gap(self):
        xs, ys = straight(80, (50.0, 64.0), (0.0, 0.3))  # 7.5 m/s towards the touchline
        observed = np.ones(80, dtype=bool)
        observed[11:60] = False  # lost at y = 67.0, seen again well outside
        crossing = detect_line_crossings(make_tracking(xs, ys, observed))
        # 1.0 m to go at 0.3 m/frame: 3.3 frames, so the first frame off the pitch is 14.
        self.assertEqual(list(crossing), [14])

    def test_lost_ball_heading_out_that_comes_back_counts(self):
        xs, ys = straight(200, (50.0, 64.0), (0.0, 0.3))
        ys[100:] = 60.0  # the ball is back on the pitch (thrown back in) when seen again
        observed = np.ones(200, dtype=bool)
        observed[11:100] = False
        crossing = detect_line_crossings(make_tracking(xs, ys, observed))
        self.assertEqual(list(crossing), [14])

    def test_lost_ball_heading_inwards_is_not_a_crossing(self):
        xs, ys = straight(200, (50.0, 60.0), (0.0, -0.2))
        observed = np.ones(200, dtype=bool)
        observed[11:100] = False
        self.assertEqual(len(detect_line_crossings(make_tracking(xs, ys, observed))), 0)


class TestRestartDetectors(unittest.TestCase):
    def test_kickoff_is_closest_approach_to_centre_spot(self):
        n = 200
        xs, ys = straight(n, (LENGTH / 2 + 1.5, WIDTH / 2), (0.6, 0.0))
        observed = np.zeros(n, dtype=bool)
        observed[80:] = True  # the ball at rest on the spot is not tracked
        xs = np.roll(xs, 80)
        tracking = make_tracking(xs, ys, observed)
        self.assertEqual(find_kickoff_frame(tracking, 0, n), 80)

    def test_no_kickoff_far_from_centre(self):
        xs, ys = straight(100, (20.0, 20.0), (0.5, 0.0))
        self.assertIsNone(find_kickoff_frame(make_tracking(xs, ys), 0, 100))

    def test_throw_in_enters_the_pitch(self):
        n = 100
        xs = np.full(n, 60.0)
        ys = np.full(n, 68.6)  # held behind the touchline
        ys[50:] = 68.6 - 0.24 * np.arange(50)  # thrown in at 6 m/s
        observed = np.ones(n, dtype=bool)
        observed[:45] = False
        tracking = make_tracking(xs, ys, observed)
        self.assertEqual(find_restart_frame(tracking, "Throw-in", 0, n), 51)

    def test_ball_boy_throw_off_the_pitch_is_not_a_restart(self):
        xs, ys = straight(100, (60.0, 80.0), (0.0, -0.1))  # rolls at 2.5 m/s, stays 3+ m outside
        ys[:40] = 80.0
        tracking = make_tracking(xs, ys)
        self.assertIsNone(find_restart_frame(tracking, "Throw-in", 0, 100))
        self.assertIsNone(find_restart_frame(tracking, "Corner Kick", 0, 100))

    def test_ball_moving_towards_the_line_is_not_a_throw_in(self):
        n = 100
        xs = np.full(n, 38.0)
        ys = np.full(n, 66.4)
        ys[50:] = 66.4 + 0.1 * np.arange(50)  # from rest 1.6 m inside, rolling out
        self.assertIsNone(find_restart_frame(make_tracking(xs, ys), "Throw-in", 0, 62))

    def test_corner_kick_from_rest(self):
        n = 100
        xs = np.full(n, LENGTH - 0.2)
        ys = np.full(n, 0.3)
        xs[60:] = LENGTH - 0.2 - 0.5 * np.arange(40)
        ys[60:] = 0.3 + 0.3 * np.arange(40)
        self.assertEqual(find_restart_frame(make_tracking(xs, ys), "Corner Kick", 0, n), 61)

    def test_aerial_kick_first_seen_off_the_pitch_counts(self):
        # A long goal kick first tracked in the air projects 4 m off the pitch, then lands inside.
        n = 100
        xs, ys = straight(n, (45.0, -4.2), (-0.5, 1.3))
        observed = np.ones(n, dtype=bool)
        observed[:50] = False
        xs = np.roll(xs, 50)
        ys = np.roll(ys, 50)
        self.assertEqual(find_restart_frame(make_tracking(xs, ys, observed), "Goal Kick", 0, n), 51)

    def test_second_half_kickoff_uses_first_clip_after_half_start(self):
        n = 3000
        xs = np.full(n, 20.0)
        ys = np.full(n, 20.0)
        xs[2100:], ys[2100:] = straight(n - 2100, (LENGTH / 2 + 0.8, WIDTH / 2), (0.6, 0.0))
        observed = np.ones(n, dtype=bool)
        observed[2000:2100] = False
        tracking = make_tracking(xs, ys, observed)
        tags = {"set_pieces": [
            {"start": 100, "end": 225, "type": "Kick Off (Start)"},
            {"start": 2020, "end": 2145, "type": "Kick Off (Start)"},
        ]}
        self.assertEqual(derive_second_half_kickoff(tracking, tags, second_half_start=1500), 2100)
        self.assertIsNone(derive_second_half_kickoff(tracking, {"set_pieces": []}, 1500))


class TestEpisodeAnchors(unittest.TestCase):
    """Live play until the ball goes out at frame 388; a throw-in clip starts at 450; a short
    throw (2.5 m/s) is taken at 700; players walk until 900 while the ball rolls slowly, so the
    kinematic detector never sees the restart and its dead state carries on."""

    N = 1100
    CLIP = 450
    THROW = 700

    def setUp(self):
        n = self.N
        xs = np.full(n, 50.0, dtype=np.float32)
        ys = np.full(n, 40.0, dtype=np.float32)
        xs[:300] = 20.0 + 0.32 * np.arange(300) % 30  # live play, 8 m/s along the pitch
        ys[300:420] = 40.0 + 0.32 * np.arange(120)  # 8 m/s towards the touchline, out at 388
        observed = np.ones(n, dtype=bool)
        observed[420:self.THROW] = False  # ball out and lost until the throw
        ys[420:self.THROW] = 68.6
        ys[self.THROW:] = 68.6 - 0.1 * np.minimum(np.arange(n - self.THROW), 20)  # thrown in
        xs[self.THROW + 20:] = 50.0 + 0.04 * np.arange(n - self.THROW - 20)  # then rolls at 1 m/s
        ys[self.THROW + 20:] = 68.6 - 0.1 * 20
        xs[900:] = xs[899] + 0.2 * np.arange(n - 900)  # play speeds up again at 900
        tracking = make_tracking(xs, ys, observed)
        tracking.player_p75_speed[400:900] = 1.0
        tracking.player_active_frac[400:900] = 0.0
        self.tracking = tracking
        self.tags = {
            "set_pieces": [{"start": self.CLIP, "end": self.CLIP + 875, "type": "Throw-in", "team": "N/A"}],
            "goals": [],
            "active_phases": [],
        }

    def evidence(self, anchors):
        return compute_episode_evidence(self.tracking, self.tags, n_frames=self.N, dead_ball_anchors=anchors)

    def test_legacy_lock_starts_six_seconds_before_clip(self):
        ev = self.evidence(False)
        self.assertGreaterEqual(ev.p_dead[350], 0.9)  # ball still in play, but locked dead
        self.assertIsNone(ev.line_crossings)

    def test_anchor_starts_dead_ball_at_the_crossing(self):
        ev = self.evidence(True)
        self.assertIn(388, list(ev.line_crossings))
        self.assertLess(ev.p_dead[350], 0.5)
        self.assertGreaterEqual(ev.p_dead[390], 0.95)
        self.assertEqual(ev.set_piece_restarts[self.CLIP], self.THROW + 1)

    def test_anchor_clears_dead_state_carried_after_restart(self):
        legacy = self.evidence(False)
        anchored = self.evidence(True)
        self.assertGreaterEqual(legacy.p_dead[800], 0.9)
        self.assertLessEqual(anchored.p_dead[800], 0.05)
        self.assertGreaterEqual(anchored.p_dead[self.THROW - 1], 0.95)


class TestEmissionAnchors(unittest.TestCase):
    def test_h2_kickoff_argument_moves_static_dead_block(self):
        n = 199319
        tracking = make_tracking(np.full(n, 52.0), np.full(n, 30.0))
        legacy = compute_log_emissions(tracking, n_frames=n)
        derived = compute_log_emissions(tracking, n_frames=n, h2_kickoff=H2_KICKOFF - 1)
        self.assertEqual(legacy[H2_KICKOFF - 1, STATE_H], -50.0)
        self.assertNotEqual(derived[H2_KICKOFF - 1, STATE_H], -50.0)
        np.testing.assert_array_equal(legacy[: H2_KICKOFF - 1], derived[: H2_KICKOFF - 1])

    def test_stoppage_gap_is_live_until_the_crossing(self):
        n = 300
        tracking = make_tracking(np.full(n, 52.0), np.full(n, 30.0))
        tracking.ball_observed[100:200] = False
        tracking.missing_gaps = [MissingGap(100, 199, 100, 1.0, 1.2, "stoppage", False)]
        zeros = np.zeros(n, dtype=np.float32)
        evidence = EpisodeEvidence(
            p_dead=np.full(n, 0.5, dtype=np.float32), p_live=np.full(n, 0.5, dtype=np.float32),
            p_fast_transit=zeros, p_restart_candidate=zeros, p_contested=zeros,
            restart_team=np.full(n, -1, dtype=np.int8), confidence=zeros,
            line_crossings=np.array([110]), set_piece_restarts={},
        )
        legacy = compute_log_emissions(tracking, n_frames=n, episode_evidence=evidence)
        anchored = compute_log_emissions(tracking, n_frames=n, episode_evidence=evidence, dead_ball_anchors=True)
        self.assertTrue(np.all(legacy[100:200, STATE_D] == 0.0))
        self.assertTrue(np.all(anchored[100:110, STATE_D] == NEG_INF))
        self.assertTrue(np.all(anchored[100:110, STATE_A] == 0.0))
        self.assertTrue(np.all(anchored[110:200, STATE_D] == 0.0))
        self.assertTrue(np.all(anchored[110:200, STATE_H] == -50.0))


if __name__ == "__main__":
    unittest.main()
