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
  Methodology document covering core axioms, pipeline stages, IFAB rules, running pace table, and limitations ([Markdown source](submission/Placeholders.md)).

---

## Quickstart & Reproduction Guide

### 1. Requirements & Setup
The engine requires Python >= 3.9 (tested on Python 3.9.6) and standard scientific dependencies:
```bash
pip install -r requirements.txt
```
*(Pinned dependencies: `numpy==2.0.2`, `pandas==2.3.3`).*

### 2. Run Test Suite
Run the full unit and invariant test suite:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
*(All 204 unit and adversarial tests pass in ~0.2 seconds).*

### 3. Generate Submission CSV
Execute the full 6-stage end-to-end pipeline (supports `data/` or `data/tactical/`):
```bash
# Standard reproduction:
bash run.sh

# Or specifying a custom data directory:
bash run.sh /path/to/data [OUTPUT_CSV]
```
Or directly via Python:
```bash
python3 -m src.run_pipeline \
    --raw data/BAR-ATH_tactical_atd.txt \
    --meta data/BAR-ATH_tactical_atd.xml \
    --smart-tagging data/BAR-ATH_tactical_smart_tagging.xml \
    --out submission/Placeholders.csv
```
*(Executes end-to-end across all 199,319 frames in ~13 seconds).*

### 4. Validate Submission Format
Verify that the output file adheres strictly to the competition submission contract:
```bash
python3 -m src.validate_submission submission/Placeholders.csv
```
*(Confirms exactly 199,319 rows, columns `frame,label`, and alphabet `{'H', 'A', 'D', 'X'}`).*

---

## Repository Layout

```
final-repo/
├── .gitignore                 # Excludes raw data, video, and local caches
├── LICENSE                    # MIT License (includes Metrica Sports attribution)
├── README.md                  # This document
├── requirements.txt           # Pinned dependencies (numpy==2.0.2, pandas==2.3.3)
├── run.sh                     # Reproduction script (generates Placeholders.csv in ~13s)
├── submission/                # Final competition deliverables
│   ├── Placeholders.csv       # The certified 199,319-row submission file
│   ├── Placeholders.md        # Technical approach document
│   └── Placeholders.pdf       # Compiled technical document for submission
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
    └── test_tracking_quality.py
```
