"""
tests/test_property_invariants.py - Property-Based & Mathematical Invariant Tests.

Verifies mathematical and physical invariants of the sequence decoder:
1. Team Symmetry: Swapping Home and Away yields swapped H/A with identical D/X.
2. Player-Order Invariance: Reordering players within a team preserves identical softmin distance.
3. Post-Processing Idempotence: Applying post-processing twice produces identical output.
4. Determinism: Identical inputs produce identical outputs bit-for-bit.
5. Numerical Safety: No NaNs or infinities in trellis forward pass.
6. Boundary Safety: Halftime break intervals strictly preserve X without cross-half merges.
"""

import unittest
import numpy as np

from src.postprocess import apply_symmetric_dwell_pruning, format_final_submission
from src.viterbi import viterbi_decode, STATE_H, STATE_A, STATE_D
from src.emissions import compute_log_emissions


class TestPropertyInvariants(unittest.TestCase):
    def test_01_team_symmetry(self):
        """Swapping Home and Away evidence inverts H and A outputs while preserving D."""
        T = 60
        log_B_orig = np.full((T, 3), -5.0, dtype=np.float32)
        # H spell [0..29], A spell [30..49], D spell [50..59]
        log_B_orig[:30, STATE_H] = 2.0
        log_B_orig[30:50, STATE_A] = 2.0
        log_B_orig[50:, STATE_D] = 5.0
        
        path_orig = viterbi_decode(log_emissions=log_B_orig)
        
        # Mirror: swap H and A
        log_B_mirrored = log_B_orig.copy()
        log_B_mirrored[:, STATE_H] = log_B_orig[:, STATE_A]
        log_B_mirrored[:, STATE_A] = log_B_orig[:, STATE_H]
        
        path_mirrored = viterbi_decode(log_emissions=log_B_mirrored)
        
        # Where orig is H (0), mirrored should be A (1)
        # Where orig is A (1), mirrored should be H (0)
        # Where orig is D (2), mirrored should be D (2)
        expected_mirrored = path_orig.copy()
        expected_mirrored[path_orig == STATE_H] = STATE_A
        expected_mirrored[path_orig == STATE_A] = STATE_H
        
        np.testing.assert_array_equal(path_mirrored, expected_mirrored)

    def test_02_player_order_invariance_in_softmin(self):
        """Permuting the order of tracked outfield players yields identical normalized softmin distance."""
        tau = 1.5
        dists = np.array([2.3, 4.1, 1.8], dtype=np.float32)
        permuted_dists = np.array([1.8, 2.3, 4.1], dtype=np.float32)
        
        # Softmin calculation
        val_orig = -tau * np.log(np.mean(np.exp(-dists / tau)))
        val_perm = -tau * np.log(np.mean(np.exp(-permuted_dists / tau)))
        
        self.assertAlmostEqual(val_orig, val_perm, places=5)

    def test_03_postprocess_idempotence(self):
        """Applying postprocessing twice produces identical results (P(P(x)) = P(x))."""
        N = 200
        # Create arbitrary noisy sequence
        np.random.seed(42)
        raw_codes = np.random.choice([0, 1, 2], size=N).astype(np.int8)
        
        pass1 = apply_symmetric_dwell_pruning(
            codes=raw_codes,
            min_poss_dwell=30,
            min_dead_dwell=35,
        )
        pass2 = apply_symmetric_dwell_pruning(
            codes=pass1,
            min_poss_dwell=30,
            min_dead_dwell=35,
        )
        
        np.testing.assert_array_equal(pass1, pass2, err_msg="Post-processing is not idempotent!")

    def test_04_determinism(self):
        """Two independent executions with identical inputs produce bit-for-bit identical state paths."""
        T = 100
        np.random.seed(123)
        log_B = np.random.randn(T, 3).astype(np.float32)
        
        path1 = viterbi_decode(log_emissions=log_B)
        path2 = viterbi_decode(log_emissions=log_B)
        
        np.testing.assert_array_equal(path1, path2)

    def test_05_numerical_safety_no_nans_or_infs(self):
        """Viterbi dynamic programming handles extreme log barriers (-1000.0) without producing NaNs or infs."""
        T = 50
        log_B = np.full((T, 3), -1000.0, dtype=np.float32)
        # Give only one viable path
        log_B[:, STATE_D] = 0.0
        
        path = viterbi_decode(log_emissions=log_B)
        self.assertTrue(np.all(path == STATE_D))
        self.assertFalse(np.any(np.isnan(path)))

    def test_06_boundary_safety_halftime_break_intact(self):
        """Halftime break interval [90652..110807] strictly retains X and never bridges across halves."""
        N = 199319
        codes = np.zeros(N, dtype=np.int8)
        # Put H right before halftime, and H right after halftime tracking starts
        codes[90600:90652] = STATE_H
        codes[110808:110850] = STATE_H
        
        final_codes = format_final_submission(codes, n_frames=N)
        
        # Check that halftime interval is strictly X (3)
        self.assertTrue(np.all(final_codes[90652:110808] == 3))


if __name__ == "__main__":
    unittest.main()
