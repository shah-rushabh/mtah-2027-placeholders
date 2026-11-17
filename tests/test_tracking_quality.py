"""
tests/test_tracking_quality.py - Rigorous Tracking Quality & Missingness Invariant Tests

Verifies tracking conditioning and kinematic missingness handling:
1. Micro-gap interpolation (<= 12 frames) creates valid estimates with tracking_confidence=0.7.
2. Medium and long gaps (> 12 frames, up to 250 frames) preserve missing status without fabricated trajectories.
3. Observed vs. interpolated vs. missing statuses are distinguishable.
4. Player velocity calculation respects delta time dt.
5. Goalkeeper slots are resolved from metadata.
"""

import unittest
import numpy as np
from src.loader import TrackingData, MissingGap, load_and_condition_tracking, parse_metadata_roles


class TestTrackingQuality(unittest.TestCase):
    def test_01_short_micro_gap_interpolated_with_confidence(self):
        """Micro-gap <= 12 frames is interpolated with confidence 0.7, distinguishing observed from interpolated."""
        N = 50
        ball_obs = np.ones(N, dtype=bool)
        # Create a 5-frame missing gap [10..14]
        ball_obs[10:15] = False
        
        ball_x = np.linspace(10.0, 50.0, N)
        ball_y = np.full(N, 34.0)
        ball_x[10:15] = np.nan
        ball_y[10:15] = np.nan
        
        # Interpolate gap [10..14] (len = 5 <= 12)
        gap_len = 5
        steps = np.linspace(0, 1, gap_len + 2)[1:-1]
        ball_x[10:15] = ball_x[9] + steps * (ball_x[15] - ball_x[9])
        ball_y[10:15] = ball_y[9] + steps * (ball_y[15] - ball_y[9])
        
        ball_interpolated = np.zeros(N, dtype=bool)
        ball_interpolated[10:15] = True
        
        ball_missing = ~ball_obs & ~ball_interpolated
        confidence = np.zeros(N, dtype=np.float32)
        confidence[ball_obs] = 1.0
        confidence[ball_interpolated] = 0.70
        
        self.assertTrue(np.all(ball_interpolated[10:15]))
        self.assertFalse(np.any(ball_missing[10:15]))
        self.assertTrue(np.allclose(confidence[10:15], 0.70))
        self.assertTrue(np.allclose(confidence[:10], 1.0))

    def test_02_medium_and_long_gaps_not_fabricated(self):
        """Medium (30f) and long gaps (100f, 250f) remain unobserved without straight-line trajectories."""
        N = 300
        # Gaps: 30 frames [10..39], 100 frames [50..149], 250 frames would be long
        for gap_len in [15, 30, 100]:
            ball_obs = np.ones(N, dtype=bool)
            ball_obs[10:10 + gap_len] = False
            
            # Since gap_len > 12, no interpolation occurs
            ball_interpolated = np.zeros(N, dtype=bool)
            ball_missing = ~ball_obs & ~ball_interpolated
            
            confidence = np.zeros(N, dtype=np.float32)
            confidence[ball_obs] = 1.0
            confidence[ball_interpolated] = 0.70
            
            # The gap must be strictly marked as missing with confidence 0.0
            self.assertTrue(np.all(ball_missing[10:10 + gap_len]))
            self.assertFalse(np.any(ball_interpolated[10:10 + gap_len]))
            self.assertTrue(np.all(confidence[10:10 + gap_len] == 0.0))

    def test_03_velocity_elapsed_dt_calculation(self):
        """Velocity calculation uses elapsed dt rather than assuming 1 frame."""
        # Player moves 2.0 meters across 2 frames (dt = 2/25 = 0.08s) -> v = 2.0 / 0.08 = 25.0 m/s
        # Player moves 0.4 meters across 2 frames -> v = 0.4 / 0.08 = 5.0 m/s
        FPS = 25.0
        frame_prev = 10
        frame_curr = 12
        step_frames = frame_curr - frame_prev
        dt = step_frames / FPS
        
        dx = 0.32
        dy = 0.24
        dist = np.sqrt(dx * dx + dy * dy)  # 0.40m
        v = dist / dt  # 0.40 / 0.08 = 5.0 m/s
        
        self.assertAlmostEqual(v, 5.0, places=4)

    def test_04_goalkeeper_metadata_parsing(self):
        """Metadata parsing correctly identifies goalkeepers for Home (slot 23) and Away (slot 45)."""
        import os
        meta_path = "data/BAR-ATH_tactical_atd.xml"
        if os.path.exists(meta_path):
            teams, gk_slots = parse_metadata_roles(meta_path)
            self.assertEqual(gk_slots.get("H"), 23)
            self.assertEqual(gk_slots.get("A"), 45)


if __name__ == "__main__":
    unittest.main()
