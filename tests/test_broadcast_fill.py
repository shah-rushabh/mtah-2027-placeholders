"""Tests for improvement #5: fill missing tactical ball frames from the broadcast tracking.

The fill is opt-in (load_and_condition_tracking(ball_fill=...), CLI --broadcast-fill), so the
submitted run is unchanged.
"""

import os
import tempfile
import unittest

import numpy as np

from src.loader import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    BroadcastSegment,
    align_broadcast,
    broadcast_ball_fill,
    load_and_condition_tracking,
    read_ball_track,
    read_homography_usable,
)


def random_ball_track(n, seed=0):
    """A smooth random ball path in 0..1 pitch units."""
    rng = np.random.default_rng(seed)
    vel = np.cumsum(rng.normal(0, 0.02, size=(n, 2)), axis=0)
    vel = np.clip(vel, -0.5, 0.5)  # metres per frame
    pos = np.cumsum(vel, axis=0)
    x = 52.0 + 40.0 * np.sin(pos[:, 0] / 40.0)
    y = 34.0 + 25.0 * np.sin(pos[:, 1] / 25.0)
    return x / PITCH_LENGTH, y / PITCH_WIDTH


class Scene:
    """Tactical track of 6000 frames; the broadcast shows tactical 1000-2499 as broadcast 0-1499,
    a 100-frame cut, then tactical 3500-4899 as broadcast 1600-2999."""

    N_T = 6000
    N_B = 3000

    def __init__(self, noise_m=0.2, seed=0):
        rng = np.random.default_rng(seed + 1)
        self.t_x, self.t_y = random_ball_track(self.N_T, seed)
        self.b_x = np.full(self.N_B, np.nan)
        self.b_y = np.full(self.N_B, np.nan)
        for b0, t0, n in ((0, 1000, 1500), (1600, 3500, 1400)):
            self.b_x[b0:b0 + n] = self.t_x[t0:t0 + n] + rng.normal(0, noise_m / PITCH_LENGTH, n)
            self.b_y[b0:b0 + n] = self.t_y[t0:t0 + n] + rng.normal(0, noise_m / PITCH_WIDTH, n)
        self.usable = np.ones(self.N_B, dtype=bool)
        self.usable[1500:1600] = False
        # Each camera loses the ball on its own frames.
        self.t_obs_x = self.t_x.copy()
        self.t_obs_y = self.t_y.copy()
        lost_t = rng.random(self.N_T) < 0.4
        self.t_obs_x[lost_t] = np.nan
        self.t_obs_y[lost_t] = np.nan
        lost_b = rng.random(self.N_B) < 0.4
        self.b_x[lost_b] = np.nan
        self.b_y[lost_b] = np.nan


class TestAlignment(unittest.TestCase):
    def test_offsets_of_both_stretches_are_recovered(self):
        s = Scene()
        segs = align_broadcast(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable)
        self.assertEqual([(g.start, g.stop, g.offset) for g in segs], [(0, 1500, 1000), (1600, 3000, 1900)])
        self.assertTrue(all(g.error_m < 0.5 for g in segs))

    def test_static_ball_is_ambiguous_and_rejected(self):
        s = Scene()
        s.t_obs_x[:] = 0.5
        s.t_obs_y[:] = 0.5
        s.b_x[~np.isnan(s.b_x)] = 0.5
        s.b_y[~np.isnan(s.b_y)] = 0.5
        segs = align_broadcast(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable)
        self.assertTrue(all(g.offset is None for g in segs))

    def test_unrelated_footage_is_rejected(self):
        s = Scene()
        other_x, other_y = random_ball_track(Scene.N_B, seed=7)
        seen = ~np.isnan(s.b_x)
        s.b_x[seen] = other_x[seen]
        s.b_y[seen] = other_y[seen]
        segs = align_broadcast(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable)
        self.assertTrue(all(g.offset is None for g in segs))

    def test_live_offset_prior_and_replays(self):
        """Five live stretches at offset 2000, one with the ball at rest, a replay of earlier play
        and a 'replay' that would map to the future."""
        rng = np.random.default_rng(3)
        n_t, n_b = 20000, 12000
        t_x, t_y = random_ball_track(n_t, seed=4)
        b_x = np.full(n_b, np.nan)
        b_y = np.full(n_b, np.nan)
        usable = np.zeros(n_b, dtype=bool)

        def show(b0, t0, n):
            b_x[b0:b0 + n] = t_x[t0:t0 + n] + rng.normal(0, 0.2 / PITCH_LENGTH, n)
            b_y[b0:b0 + n] = t_y[t0:t0 + n] + rng.normal(0, 0.2 / PITCH_WIDTH, n)
            usable[b0:b0 + n] = True

        for b0 in (0, 1200, 2400, 3600):
            show(b0, b0 + 2000, 1000)
        t_x[6800:7800] = t_x[6800]  # a stoppage: the ball rests for 40 s
        t_y[6800:7800] = t_y[6800]
        show(4800, 6800, 1000)
        show(6000, 3000, 400)  # replay of play 2000 frames before the live time (8000)
        show(7000, 15000, 400)  # maps 6000 frames after the live time (9000): impossible
        segs = align_broadcast(t_x, t_y, b_x, b_y, usable)
        self.assertEqual([g.offset for g in segs], [2000, 2000, 2000, 2000, 2000, -3000, None])

    def test_short_stretch_is_not_aligned(self):
        s = Scene()
        s.usable[:] = False
        s.usable[100:130] = True
        self.assertEqual(align_broadcast(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable), [])


class TestFill(unittest.TestCase):
    def test_fill_only_where_tactical_ball_is_missing(self):
        s = Scene()
        segs = [BroadcastSegment(0, 1500, 1000, 0.2, 500), BroadcastSegment(1600, 3000, None, float("nan"), 0)]
        fill_x, fill_y = broadcast_ball_fill(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable, segs)
        filled = np.flatnonzero(~np.isnan(fill_x))
        self.assertGreater(len(filled), 100)
        self.assertTrue(np.all(np.isnan(s.t_obs_x[filled])))  # never overwrites a tactical ball
        self.assertTrue(np.all((filled >= 1000) & (filled < 2500)))  # unaligned stretch unused
        np.testing.assert_allclose(fill_x[filled], s.b_x[filled - 1000])
        np.testing.assert_allclose(fill_y[filled], s.b_y[filled - 1000])

    def test_unusable_broadcast_frames_are_not_used(self):
        s = Scene()
        s.usable[200:300] = False
        segs = [BroadcastSegment(0, 1500, 1000, 0.2, 500)]
        fill_x, _ = broadcast_ball_fill(s.t_obs_x, s.t_obs_y, s.b_x, s.b_y, s.usable, segs)
        self.assertTrue(np.all(np.isnan(fill_x[1200:1300])))

    def test_replay_overlap_prefers_lower_error(self):
        t_x = np.full(10, np.nan)
        b_x = np.array([0.1] * 5 + [0.9] * 5)
        b_y = np.full(10, 0.5)
        segs = [BroadcastSegment(0, 5, 0, 0.8, 50), BroadcastSegment(5, 10, -5, 0.3, 50)]
        fill_x, _ = broadcast_ball_fill(t_x, t_x, b_x, b_y, np.ones(10, bool), segs)
        np.testing.assert_allclose(fill_x[:5], 0.9)


class TestFiles(unittest.TestCase):
    META = """<?xml version="1.0"?><main><Teams>
<Team id="T1"/><Team id="T2"/></Teams><Players>
<Player id="p1" teamId="T1"/><Player id="p2" teamId="T1"/>
<Player id="p3" teamId="T2"/><Player id="p4" teamId="T2"/></Players></main>"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw = os.path.join(self.tmp.name, "atd.txt")
        self.meta = os.path.join(self.tmp.name, "atd.xml")
        with open(self.meta, "w", encoding="utf-8") as f:
            f.write(self.META)
        lines = []
        for frame_id in range(1, 41):
            players = ";".join(["0.50000,0.50000,1.0", "0.30000,0.30000,1.0", "0.55000,0.50000,1.0", "0.90000,0.90000,1.0"])
            ball = "NaN,NaN,NaN" if 10 <= frame_id <= 29 else f"{0.5 + 0.001 * frame_id:.5f},0.50000,0.0"
            lines.append(f"{frame_id}:{players}:{ball}\n")
        with open(self.raw, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_ball_track(self):
        x, y = read_ball_track(self.raw, id_base=1)
        self.assertEqual(len(x), 40)
        self.assertTrue(np.all(np.isnan(x[9:29])))
        self.assertAlmostEqual(x[0], 0.501)
        self.assertAlmostEqual(y[39], 0.5)

    def test_read_homography_usable(self):
        path = os.path.join(self.tmp.name, "h.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("frame,h00,h01,h02,h10,h11,h12,h20,h21,h22,reprojection_error,valid\n")
            f.write("0,1,0,0,0,1,0,0,0,1,0.3,1\n1,1,0,0,0,1,0,0,0,1,9.0,1\n2,nan,nan,nan,nan,nan,nan,nan,nan,nan,-100,0\n")
        self.assertEqual(list(read_homography_usable(path, 4)), [True, False, False, False])

    def test_loader_uses_fill_only_when_given(self):
        n = 40
        legacy = load_and_condition_tracking(self.raw, self.meta, n_frames=n)
        fill_x = np.full(n, np.nan)
        fill_y = np.full(n, np.nan)
        fill_x[15] = 0.52  # video frame 15 = frame id 16, missing in the tactical file
        fill_y[15] = 0.50
        fill_x[5] = 0.99   # frame 5 has a tactical ball: the fill must be ignored there
        fill_y[5] = 0.99
        filled = load_and_condition_tracking(self.raw, self.meta, n_frames=n, ball_fill=(fill_x, fill_y))
        self.assertIsNone(legacy.ball_from_broadcast)
        self.assertFalse(legacy.ball_observed[15])
        self.assertTrue(filled.ball_observed[15])
        self.assertTrue(filled.ball_from_broadcast[15])
        self.assertFalse(filled.ball_from_broadcast[5])
        self.assertAlmostEqual(float(filled.ball_x_metric[5]), legacy.ball_x_metric[5], places=4)
        self.assertAlmostEqual(float(filled.ball_x_norm[15]), 0.52, places=5)
        # Nearest home outfield player sits at (0.5, 0.5): 0.02 * 105.3 = 2.106 m away.
        self.assertAlmostEqual(float(filled.d_H[15]), 0.02 * PITCH_LENGTH, places=3)
        self.assertAlmostEqual(float(filled.tracking_confidence[15]), 0.85, places=5)


if __name__ == "__main__":
    unittest.main()
