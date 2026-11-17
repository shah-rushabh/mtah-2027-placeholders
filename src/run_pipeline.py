"""Main CLI & Orchestrator: Multimodal 6-Stage Football Possession Engine.

Orchestrates the end-to-end possession inference pipeline:
1. Stage 1: Data Ingestion, Metric Conditioning & Missingness Validity (loader.py)
2. Stage 2: Multimodal Game Phase & Episode Evidence Engine (episodes.py)
3. Stage 3: Player Interaction & Controlled Possession Engine (control.py)
4. Stage 4: Conditioned Spatial Likelihoods & Stoppage Priors (emissions.py)
5. Stage 5: Context-Aware Log-Viterbi HMM Decoder (viterbi.py)
6. Stage 6: Symmetric RLE Minimum-Dwell Pruner & Formatter (postprocess.py)

Generates:
- submission.csv (199,319 rows, columns: frame, label in {H, A, D, X})
- Missing-Gap Audit Report (automated markdown report of optical occlusions)

All parameters and default thresholds are mathematically and physically grounded in
certified pitch geometry, IFAB Laws of the Game, and athletic biomechanics.
"""

import argparse
import os
import sys
import time
from typing import Optional

import numpy as np

# Core Pipeline Stages
from .control import ControlEvidence, compute_control_evidence
from .emissions import compute_log_emissions
from .episodes import EpisodeEvidence, compute_episode_evidence, export_episode_audit
from .loader import (
    HOME_TEAM_CODE,
    PITCH_LENGTH,
    PITCH_WIDTH,
    TOTAL_FRAMES,
    load_and_condition_tracking,
    parse_smart_tagging,
)
from .postprocess import (
    CODE_TO_CHAR,
    apply_symmetric_dwell_pruning,
    export_submission_csv,
    format_final_submission,
    rle_encode,
)
from .viterbi import viterbi_decode


def run_pipeline(
    raw_atd_path: str,
    meta_atd_path: str,
    smart_tagging_path: str,
    output_csv_path: str,
    audit_report_path: Optional[str] = None,
    episode_audit_path: Optional[str] = None,
    gap_report_path: Optional[str] = None,
    trajectory_strategy: str = "active_bridging",
    ownership_strategy: str = "softmin_multi",
    sigma: float = 4.0,
    transition_strategy: str = "asymmetric_turnover",
    min_poss_dwell: int = 30,
    min_dead_dwell: int = 35,
    min_same_team_dead_dwell: int = 45,
    use_episode_engine: bool = True,
    augmented_5state: bool = False,
) -> str:
    """Executes the full 5-stage pipeline and exports the verified submission CSV.

    Args:
        raw_atd_path: Path to raw tactical tracking text file (199,319 frames).
        meta_atd_path: Path to tactical metadata XML containing pitch survey & half bounds.
        smart_tagging_path: Path to Metrica Smart Tagging XML containing set-pieces & goals.
        output_csv_path: Destination path for final submission CSV (199,319 rows).
        audit_report_path: Optional markdown path for automated missing-gap audit.
        trajectory_strategy: 'active_bridging' (physics interpolation) or 'neutral_gaps'.
        ownership_strategy: 'softmin_multi' (top 3 players) or 'single_nearest'.
        sigma: Proximity exponential decay scale in meters (default: 4.0m).
        transition_strategy: 'asymmetric_turnover', 'high_persistence', or 'natural_dwell'.
        min_poss_dwell: Minimum duration for H/A possession spells (default: 30 frames = 1.2s).
        min_dead_dwell: Minimum duration for Dead Ball intervals (default: 35 frames = 1.4s).
        use_episode_engine: Enable Stage 2 multimodal episode evidence engine (default: True).

    Returns:
        output_csv_path: Absolute path to the generated submission file.
    """
    t_start = time.time()
    n_stages = 6

    print("================================================================================")
    print("🏆 MTAH 2027 — Multimodal 6-Stage Football Possession Engine")
    print("================================================================================")

    # =========================================================================
    # STAGE 1: Data Ingestion, Conditioning & Missingness Validity
    # =========================================================================
    print(f"\n[Stage 1/{n_stages}] Streaming EPTS tracking & conditioning kinematics...")
    t0 = time.time()

    tracking = load_and_condition_tracking(
        raw_path=raw_atd_path,
        meta_path=meta_atd_path,
        smart_tagging_path=smart_tagging_path,
        n_frames=TOTAL_FRAMES,
        pitch=(PITCH_LENGTH, PITCH_WIDTH),
        micro_gap_max=12,  # 0.48s ballistic flight threshold
        trajectory_strategy=trajectory_strategy,
    )
    smart_tags = parse_smart_tagging(smart_tagging_path)

    print(f"  ✓ Processed {TOTAL_FRAMES:,} frames in {time.time() - t0:.2f}s")
    print(f"  ✓ Ball observed on {np.sum(tracking.ball_observed):,} frames ({np.mean(tracking.ball_observed)*100:.1f}%)")
    print(f"  ✓ Missing gaps detected: {len(tracking.missing_gaps)} contiguous intervals (Strategy: {trajectory_strategy})")

    # =========================================================================
    # STAGE 2: Game Phase & Multimodal Episode Evidence Engine
    # =========================================================================
    episode_evidence: Optional[EpisodeEvidence] = None

    if use_episode_engine:
        print(f"\n[Stage 2/{n_stages}] Extracting factorized episode evidence (Kinematic hysteresis, Restarts)...")
        t0 = time.time()

        episode_evidence = compute_episode_evidence(
            tracking=tracking,
            smart_tags=smart_tags,
            n_frames=TOTAL_FRAMES,
            pitch=(PITCH_LENGTH, PITCH_WIDTH),
        )

        print(f"  ✓ Episode evidence computed in {time.time() - t0:.2f}s")
        print(f"  ✓ Estimated dead stoppage frames: {np.sum(episode_evidence.p_dead > 0.7):,} frames")
        print(f"  ✓ Estimated fast transit frames: {np.sum(episode_evidence.p_fast_transit > 0.5):,} frames")
        print(f"  ✓ Candidate restart events detected: {np.sum(episode_evidence.p_restart_candidate > 0.8):,}")

    # =========================================================================
    # STAGE 3: Player Interaction & Controlled Possession Engine
    # =========================================================================
    print(f"\n[Stage 3/{n_stages}] Evaluating player interaction vs. controlled possession evidence...")
    t0 = time.time()
    control_evidence = compute_control_evidence(
        tracking=tracking,
        episodes=episode_evidence,
        pitch=(PITCH_LENGTH, PITCH_WIDTH),
    )
    print(f"  ✓ Control evidence computed in {time.time() - t0:.2f}s")
    print(f"  ✓ Grouped deflection events: {np.sum(control_evidence.is_deflection_event):,}")
    print(f"  ✓ Candidate turnovers (H->A): {np.sum(control_evidence.turnover_candidate_H_to_A):,}, (A->H): {np.sum(control_evidence.turnover_candidate_A_to_H):,}")

    # =========================================================================
    # STAGE 4: Conditioned Spatial Likelihoods & Stoppage Priors
    # =========================================================================
    print(f"\n[Stage 4/{n_stages}] Computing spatial likelihoods (Strategy: {ownership_strategy}, sigma: {sigma}m)...")
    t0 = time.time()

    log_emissions = compute_log_emissions(
        tracking=tracking,
        smart_tags=smart_tags,
        n_frames=TOTAL_FRAMES,
        sigma=sigma,
        ownership_strategy=ownership_strategy,
        episode_evidence=episode_evidence,
        control_evidence=control_evidence if augmented_5state else None,
        augmented_5state=augmented_5state,
    )

    print(f"  ✓ Log-emissions calculated in {time.time() - t0:.2f}s (Shape: {log_emissions.shape})")

    # =========================================================================
    # STAGE 5: Context-Aware Log-Viterbi HMM Decoder
    # =========================================================================
    print(f"\n[Stage 5/6] Running Log-Viterbi dynamic programming (Transition: {transition_strategy})...")
    t0 = time.time()

    raw_path = viterbi_decode(
        log_emissions=log_emissions,
        transition_strategy=transition_strategy,
        episode_evidence=episode_evidence,
        control_evidence=control_evidence if augmented_5state else None,
    )

    print(f"  ✓ Optimal MAP state sequence decoded in {time.time() - t0:.2f}s")

    # =========================================================================
    # STAGE 6: Symmetric RLE Minimum-Dwell Pruner & Graded Formatter
    # =========================================================================
    print(f"\n[Stage 6/6] Applying context-aware dwell pruning (Dwell: H/A={min_poss_dwell}f, D={min_dead_dwell}f)...")
    t0 = time.time()

    pruned_path = apply_symmetric_dwell_pruning(
        codes=raw_path,
        ball_x_norm=tracking.ball_x_norm,
        ball_y_norm=tracking.ball_y_norm,
        player_p75_speed=tracking.player_p75_speed,
        min_poss_dwell=min_poss_dwell,
        min_dead_dwell=min_dead_dwell,
        min_same_team_dead_dwell=min_same_team_dead_dwell,
        p_dead=episode_evidence.p_dead if episode_evidence is not None else None,
        p_restart=episode_evidence.p_restart_candidate if episode_evidence is not None else None,
        is_deflection=getattr(episode_evidence, "is_deflection", None),
        intentional_action_H=getattr(control_evidence, "intentional_action_H", None),
        intentional_action_A=getattr(control_evidence, "intentional_action_A", None),
    )

    final_codes = format_final_submission(codes=pruned_path, n_frames=TOTAL_FRAMES)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    export_submission_csv(final_codes, output_csv_path)

    print(f"  ✓ Post-processing & CSV exported in {time.time() - t0:.2f}s")
    print(f"  ✓ Submission saved to: {output_csv_path}")

    # Export episode audit CSV report if requested
    actual_episode_audit_path = episode_audit_path or (audit_report_path if audit_report_path and audit_report_path.endswith(".csv") else None)
    if actual_episode_audit_path and episode_evidence is not None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(actual_episode_audit_path)), exist_ok=True)
            export_episode_audit(episode_evidence, tracking, actual_episode_audit_path)
            print(f"  ✓ Episode audit saved to: {actual_episode_audit_path}")
        except Exception as e:
            print(f"Warning: could not export episode audit: {e}")

    # =========================================================================
    # VERIFICATION & SUMMARY STATISTICS
    # =========================================================================
    total_time = time.time() - t_start
    print("\n================================================================================")
    print(f"✓ Pipeline execution completed successfully in {total_time:.2f} seconds!")
    print("================================================================================")

    # State timeline distribution breakdown
    counts = {char: np.sum(final_codes == code) for char, code in [("H", 0), ("A", 1), ("D", 2), ("X", 3)]}
    print("\nFinal Timeline Distribution (199,319 frames):")
    for char in ("H", "A", "D", "X"):
        pct = (counts[char] / TOTAL_FRAMES) * 100
        mins = (counts[char] / 25.0) / 60.0
        print(f"  [{char}] {counts[char]:>7,} frames ({pct:>5.1f}%) — {mins:>5.1f} minutes")

    # Possession spell statistics
    segments = rle_encode(final_codes)
    poss_spells = [seg for seg in segments if seg[2] in (0, 1)]
    dead_spells = [seg for seg in segments if seg[2] == 2]

    print("\nPossession Spell Statistics:")
    print(f"  Total Possessions (H + A): {len(poss_spells):,}")
    print(f"    - Home (Barcelona): {sum(1 for s in poss_spells if s[2] == 0):,} spells")
    print(f"    - Away (Athletic):  {sum(1 for s in poss_spells if s[2] == 1):,} spells")
    print(f"  Total Dead Intervals (D):  {len(dead_spells):,} intervals")
    if poss_spells:
        mean_dur_s = np.mean([s[1] - s[0] for s in poss_spells]) / 25.0
        print(f"  Mean Possession Duration:  {mean_dur_s:.2f} seconds")

    # Missing gap audit report
    actual_gap_report_path = gap_report_path or (audit_report_path if audit_report_path and audit_report_path.endswith(".md") else None)
    if actual_gap_report_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(actual_gap_report_path)), exist_ok=True)
            with open(actual_gap_report_path, "w", encoding="utf-8") as f:
                f.write("# Automated Missing-Ball Gap Audit Report\n\n")
                f.write(f"Total Missing Gaps: {len(tracking.missing_gaps)}\n\n")
                f.write("| Gap ID | Start Frame | End Frame | Length (Frames) | Length (s) | Category | Overlaps XML | Mean Player Speed (m/s) |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
                for i, gap in enumerate(tracking.missing_gaps):
                    sec = gap.length / 25.0
                    f.write(f"| {i+1} | {gap.start_frame} | {gap.end_frame} | {gap.length} | {sec:.2f}s | {gap.category} | {gap.overlaps_xml_stoppage} | {gap.mean_player_speed:.2f} |\n")
            print(f"\n✓ Missing gap audit report saved to: {actual_gap_report_path}")
        except Exception as e:
            print(f"\n[Note] Could not write gap audit report: {e}")

    return output_csv_path


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTAH 2027 Multimodal 6-Stage Football Possession Engine")
    parser.add_argument("--raw", default="data/BAR-ATH_tactical_atd.txt", help="Path to raw tactical ATD text file")
    parser.add_argument("--meta", default="data/BAR-ATH_tactical_atd.xml", help="Path to tactical ATD metadata XML")
    parser.add_argument("--smart-tagging", default="data/BAR-ATH_tactical_smart_tagging.xml", help="Path to Smart Tagging XML")
    parser.add_argument("--out", default="submission/Placeholders.csv", help="Path for output submission CSV")
    parser.add_argument("--audit", default=None, help="Legacy generic audit path")
    parser.add_argument("--episode-audit", default=None, help="Optional path for episode evidence audit CSV (opt-in diagnostic)")
    parser.add_argument("--gap-report", default=None, help="Optional path for missing gap audit markdown (opt-in diagnostic)")
    parser.add_argument("--trajectory", default="active_bridging", choices=["neutral_gaps", "active_bridging"])
    parser.add_argument("--ownership", default="softmin_multi", choices=["single_nearest", "softmin_multi"])
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--transition", default="asymmetric_turnover", choices=["high_persistence", "natural_dwell", "asymmetric_turnover"])
    parser.add_argument("--min-poss", type=int, default=30)
    parser.add_argument("--min-dead", type=int, default=35)
    parser.add_argument("--min-same-team-dead", type=int, default=45)
    parser.add_argument("--episodes", action="store_true", default=True, help="Enable Stage 2 Episode Evidence Engine")
    parser.add_argument("--no-episodes", dest="episodes", action="store_false", help="Disable Episode Evidence Engine")
    parser.add_argument("--augmented-5state", action="store_true", default=False, help="Enable Augmented 5-State Temporal Possession Model")

    args = parser.parse_args()

    # Resolve paths gracefully supporting both root data/ and official data/tactical/ bundles
    raw_path = args.raw
    if not os.path.exists(raw_path) and os.path.exists("data/tactical/BAR-ATH_tactical_atd.txt"):
        raw_path = "data/tactical/BAR-ATH_tactical_atd.txt"
    meta_path = args.meta
    if not os.path.exists(meta_path) and os.path.exists("data/tactical/BAR-ATH_tactical_atd.xml"):
        meta_path = "data/tactical/BAR-ATH_tactical_atd.xml"
    smart_path = args.smart_tagging
    if not os.path.exists(smart_path) and os.path.exists("data/tactical/BAR-ATH_tactical_smart_tagging.xml"):
        smart_path = "data/tactical/BAR-ATH_tactical_smart_tagging.xml"

    run_pipeline(
        raw_atd_path=raw_path,
        meta_atd_path=meta_path,
        smart_tagging_path=smart_path,
        output_csv_path=args.out,
        audit_report_path=args.audit,
        episode_audit_path=args.episode_audit,
        gap_report_path=args.gap_report,
        trajectory_strategy=args.trajectory,
        ownership_strategy=args.ownership,
        sigma=args.sigma,
        transition_strategy=args.transition,
        min_poss_dwell=args.min_poss,
        min_dead_dwell=args.min_dead,
        min_same_team_dead_dwell=args.min_same_team_dead,
        use_episode_engine=args.episodes,
        augmented_5state=args.augmented_5state,
    )


if __name__ == "__main__":
    main()
