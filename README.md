# MTAH 2027 — Multimodal Football Possession Engine

Official competition entry for the **Metrica Sports × FC Barcelona × Mobile World Capital Barcelona** challenge (*More Than A Hack 2027*, Talent Arena Barcelona).

### Team
* **Team Name:** "Placeholders"
* **Members:** Rushabh Shah, Karan Kewalramani, Samarth Gupta, Madhur Kalia (India)
* **Target Match:** FC Barcelona vs. Athletic Club (`BAR-ATH`, 199,319 frames @ 25.0 fps, ~106 minutes)
* **Target Alphabet:** `{'H', 'A', 'D', 'X'}` (Home, Away, Dead-ball, Abstain)

### Deliverables
* **Possession File:** [`submission/Placeholders.csv`](submission/Placeholders.csv)  
  Certified submission file containing exactly 199,319 frames formatted as `frame,label`.
* **Technical Document:** [`submission/Placeholders.pdf`](submission/Placeholders.pdf)  
  Technical document covering the approach, pipeline steps, possession rules, assets used, limitations, and reproduction.

---

## Quickstart & Reproduction Guide

### 1. Requirements & Setup
The engine requires Python 3.9–3.12 (tested on Python 3.9.6) and standard scientific dependencies:
```bash
pip install -r requirements.txt
```
*(Pinned dependencies: `numpy==2.0.2`, `pandas==2.3.3`).*

### 2. Run Test Suite
Run the full unit and invariant test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
*(All 260 unit and adversarial tests pass in ~0.5 seconds).*

### 3. Generate Submission CSV
Execute the full 6-stage end-to-end pipeline. It reads the tactical tracking, its metadata XML,
the Smart Tagging XML, and the broadcast tracking and homography (used only to fill missing ball
frames). It supports a flat `data/` folder or the organisers' `data/tactical/` + `data/broadcast/` layout:
```bash
# Standard reproduction:
bash run.sh

# Or specifying a custom data directory:
bash run.sh /path/to/data [OUTPUT_CSV]
```
`run.sh` changes into the repository folder before it runs. Run it from the repository root, or pass
absolute paths for `DATA_DIR` and `OUTPUT_CSV`.
Or directly via Python:
```bash
python3 -m src.run_pipeline \
    --raw data/BAR-ATH_tactical_atd.txt \
    --meta data/BAR-ATH_tactical_atd.xml \
    --smart-tagging data/BAR-ATH_tactical_smart_tagging.xml \
    --flight-barrier \
    --confirm-turnover-frames 50 \
    --turnover-at-receiver \
    --broadcast-fill \
    --broadcast-raw data/BAR-ATH_broadcast_atd.txt \
    --broadcast-homography data/BAR-ATH_broadcast_homography.csv \
    --out submission/Placeholders.csv
```
- `--flight-barrier` stops the ball changing team while it is in fast flight and nothing deflects it.
- `--confirm-turnover-frames 50` gives a short opponent spell between two spells of the same team
  (A-B-A, under 2 s) back to that team: a touch is not a change of possession.
- `--turnover-at-receiver` moves each change of team (by at most 2 s) to the first frame where the
  new team controls the ball.
- `--broadcast-fill` fills missing tactical ball frames with the broadcast ball position. Each broadcast
  stretch is aligned to the tactical clock by matching ball positions.
- `--dead-ball-anchors` exists but is not used in the submission.

*(Executes end-to-end across all 199,319 frames in ~30 seconds).*

### 4. Validate Submission Format
Verify that the output file adheres strictly to the competition submission contract:
```bash
python3 -m src.validate_submission submission/Placeholders.csv
```
*(Confirms exactly 199,319 rows, columns `frame,label`, and alphabet `{'H', 'A', 'D', 'X'}`).*

---

## Repository Layout

```
./
├── .gitignore                 # Excludes raw data, video, and local caches
├── LICENSE                    # MIT License (includes Metrica Sports attribution)
├── README.md                  # This document
├── requirements.txt           # Pinned dependencies (numpy==2.0.2, pandas==2.3.3)
├── run.sh                     # Reproduction script (generates Placeholders.csv in ~30s)
├── submission/                # Final competition deliverables
│   ├── Placeholders.csv       # The certified 199,319-row submission file
│   └── Placeholders.pdf       # Final technical document for submission
├── src/
│   ├── __init__.py
│   ├── loader.py              # Stage 1: Data Ingestion & Missingness Engine
│   ├── episodes.py            # Stage 2: Game Phase & Episode Evidence Engine
│   ├── control.py             # Stage 3: Player Interaction & Control Engine
│   ├── emissions.py           # Stage 4: Conditioned Spatial Likelihoods & Stoppages
│   ├── viterbi.py             # Stage 5: Context-Aware Log-Viterbi HMM Decoder
│   ├── postprocess.py         # Stage 6: Symmetric RLE Minimum-Dwell Pruner & Formatter
│   ├── run_pipeline.py        # Unified CLI Pipeline Runner
│   ├── games.py               # Official match constants & hardware frame offsets
│   └── validate_submission.py # Official submission format & schema validator
└── tests/                     # Complete test suite (invariants, geometry, stress tests)
    ├── __init__.py
    ├── test_adversarial_scenarios.py
    ├── test_augmented_5state_model.py
    ├── test_end_to_end_scenarios.py
    ├── test_episode_scenarios.py
    ├── test_possession_control_semantics.py
    ├── test_property_invariants.py
    ├── test_scaled_adversarial_suite.py
    ├── test_tracking_quality.py
    ├── test_broadcast_fill.py
    ├── test_dead_ball_anchors.py
    ├── test_turnover_at_receiver.py
    └── test_turnover_confirm_flight_barrier.py
```
