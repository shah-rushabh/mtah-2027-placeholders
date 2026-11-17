#!/usr/bin/env bash
# =============================================================================
# MTAH 2027: More Than A Hack 2027 Challenge — Official Submission Runner
# Target Match: FC Barcelona vs. Athletic Club (BAR-ATH, 199,319 frames @ 25 Hz)
#
# Usage:
#   bash run.sh [DATA_DIR] [OUTPUT_CSV]
#
# Examples:
#   bash run.sh
#   bash run.sh data submission/Placeholders.csv
#   bash run.sh /path/to/data custom_output.csv
# =============================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

DATA_DIR=${1:-"data"}
OUTPUT_CSV=${2:-"submission/Placeholders.csv"}
mkdir -p "$(dirname "$OUTPUT_CSV")"

# Support both flat data/ and nested data/tactical/ directory structures
if [ -f "$DATA_DIR/tactical/BAR-ATH_tactical_atd.txt" ]; then
    RAW_FILE="$DATA_DIR/tactical/BAR-ATH_tactical_atd.txt"
    META_FILE="$DATA_DIR/tactical/BAR-ATH_tactical_atd.xml"
    SMART_FILE="$DATA_DIR/tactical/BAR-ATH_tactical_smart_tagging.xml"
else
    RAW_FILE="$DATA_DIR/BAR-ATH_tactical_atd.txt"
    META_FILE="$DATA_DIR/BAR-ATH_tactical_atd.xml"
    SMART_FILE="$DATA_DIR/BAR-ATH_tactical_smart_tagging.xml"
fi

# Detect Python environment
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_CMD="$VIRTUAL_ENV/bin/python"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_CMD="../.venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "================================================================="
echo "⚽ MTAH 2027 — Multimodal Football Possession Engine"
echo "Team: Placeholders | Match: FC Barcelona vs. Athletic Club (BAR-ATH)"
echo "Python:          $PYTHON_CMD"
echo "Input Raw ATD:   $RAW_FILE"
echo "Input Metadata:  $META_FILE"
echo "Input Smart Tag: $SMART_FILE"
echo "Output CSV:      $OUTPUT_CSV"
echo "================================================================="

if [ ! -f "$RAW_FILE" ]; then
    echo "Error: Raw tracking file not found at $RAW_FILE"
    echo "Usage: bash run.sh [DATA_DIR] [OUTPUT_CSV]"
    exit 1
fi

# Execute the 6-stage possession pipeline (Production 3-State HMM with Factorized Episode Evidence)
$PYTHON_CMD -m src.run_pipeline \
    --raw "$RAW_FILE" \
    --meta "$META_FILE" \
    --smart-tagging "$SMART_FILE" \
    --out "$OUTPUT_CSV" \
    --ownership softmin_multi \
    --transition asymmetric_turnover \
    --min-poss 30 \
    --min-dead 35 \
    --min-same-team-dead 45 \
    --episodes

echo ""
echo "================================================================="
echo "✓ Pipeline complete. Validating output format..."
$PYTHON_CMD -m src.validate_submission "$OUTPUT_CSV"
echo "================================================================="
echo "Submission file ready: $OUTPUT_CSV"
