#!/bin/bash
# Phase D — pull Modal artifacts, render manuscript, run audits, build supplementary archive.
# Run after Phase C orchestrate completes (i.e., Modal app ap-JjXLd4wTclB0pJBSMWe6qK
# transitions out of ephemeral state).
#
# Usage:
#   ./phase_d_render.sh <RUN_ID>
#
# If multiple Modal orchestrate runs have written to the same volume (e.g.,
# due to Modal worker restarts), pass the specific run_id to use as canonical.
# To list run_ids:
#   modal volume ls rl-llm-safety-v3-predictions /canonical | awk -F_ '{print $NF}' | sort -u

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <run_id>"
    echo ""
    echo "Available run_ids in Modal volume:"
    modal volume ls rl-llm-safety-v3-predictions /canonical 2>&1 | \
        grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | \
        sort -u
    exit 1
fi
RUN_ID="$1"
echo "Using run_id: $RUN_ID"

REPO_DIR="/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3"
OUT_DIR="/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3"

mkdir -p "$REPO_DIR/predictions/canonical"
mkdir -p "$REPO_DIR/results/tables"
mkdir -p "$REPO_DIR/results/figures"
mkdir -p "$OUT_DIR/drafts"
mkdir -p "$OUT_DIR/supplementary"

echo "===================================================================="
echo "Phase D — finalize manuscript from canonical CSVs"
echo "===================================================================="

echo ""
echo "=== Step 1: pull per-message prediction CSVs from Modal ==="
modal volume get rl-llm-safety-v3-predictions /canonical/ "$REPO_DIR/predictions/" --force 2>&1 | tail -3

# Also pull run_manifest.json
modal volume get rl-llm-safety-v3-predictions /run_manifest.json "$REPO_DIR/predictions/run_manifest.json" --force 2>&1 | tail -2

echo ""
echo "=== Step 2: pull canonical results CSVs from Modal ==="
modal volume get rl-llm-safety-v3-results / "$REPO_DIR/results/" --force 2>&1 | tail -3

echo ""
echo "=== Step 3: ensure linguistic features computed ==="
cd "$REPO_DIR/code/pipeline"
python3 compute_linguistic_features.py \
    --realworld /Users/sanjaybasu/waymark-local/data/official/realworld_cases_n2000.json \
    --physician /Users/sanjaybasu/waymark-local/data/official/physician_holdout_n41.json \
    --output "$REPO_DIR/results/linguistic_features.json"

echo ""
echo "=== Step 3.5: filter to single run_id ==="
mkdir -p "$REPO_DIR/predictions/canonical_filtered"
rm -f "$REPO_DIR/predictions/canonical_filtered/"*.csv
cp "$REPO_DIR/predictions/canonical/"*"_${RUN_ID}.csv" "$REPO_DIR/predictions/canonical_filtered/" 2>/dev/null || true
echo "Filtered to run_id=${RUN_ID}:"
ls "$REPO_DIR/predictions/canonical_filtered/"

echo ""
echo "=== Step 4: ROC monotonicity audit ==="
cd "$REPO_DIR/code/audit"
python3 roc_monotonicity.py "$REPO_DIR/predictions/canonical_filtered/" --strict

echo ""
echo "=== Step 5: render canonical metrics tables (Phase 12 was already in Modal; rerun locally for review) ==="
cd "$REPO_DIR/code/pipeline"
python3 render_tables_figures.py \
    --predictions-dir "$REPO_DIR/predictions/canonical_filtered" \
    --results-dir "$REPO_DIR/results"

echo ""
echo "=== Step 6: render manuscript main text ==="
python3 manuscript_renderer.py \
    --template "$REPO_DIR/manuscript_template/main_text_template.md" \
    --predictions "$REPO_DIR/predictions/canonical_filtered" \
    --results "$REPO_DIR/results" \
    --manifest "$REPO_DIR/predictions/run_manifest.json" \
    --output "$OUT_DIR/drafts/main_text.md"

echo ""
echo "=== Step 7: render cover letter ==="
python3 manuscript_renderer.py \
    --template "$REPO_DIR/manuscript_template/cover_letter_template.md" \
    --predictions "$REPO_DIR/predictions/canonical_filtered" \
    --results "$REPO_DIR/results" \
    --manifest "$REPO_DIR/predictions/run_manifest.json" \
    --output "$OUT_DIR/drafts/cover_letter.md"

echo ""
echo "=== Step 8: render Multimedia Appendix 1 ==="
python3 manuscript_renderer.py \
    --template "$REPO_DIR/manuscript_template/appendix_template.md" \
    --predictions "$REPO_DIR/predictions/canonical_filtered" \
    --results "$REPO_DIR/results" \
    --manifest "$REPO_DIR/predictions/run_manifest.json" \
    --output "$OUT_DIR/drafts/appendix.md"

echo ""
echo "=== Step 9: build Multimedia Appendix 2 (per-message supplementary ZIP) ==="
python3 build_supplementary_package.py \
    --predictions-dir "$REPO_DIR/predictions/canonical_filtered" \
    --results-dir "$REPO_DIR/results" \
    --manifest "$REPO_DIR/predictions/run_manifest.json" \
    --output "$OUT_DIR/supplementary/multimedia_appendix_2.zip"

echo ""
echo "=== Step 10: consistency-audit linter on rendered manuscript + appendix ==="
python3 "$REPO_DIR/code/audit/consistency_audit.py" \
    "$OUT_DIR/drafts/main_text.md" \
    "$OUT_DIR/drafts/appendix.md" \
    --strict || echo "(non-strict: review warnings above)"

echo ""
echo "===================================================================="
echo "Phase D complete"
echo "===================================================================="
echo "Manuscript draft:           $OUT_DIR/drafts/main_text.md"
echo "Cover letter draft:         $OUT_DIR/drafts/cover_letter.md"
echo "Multimedia Appendix 1:      $OUT_DIR/drafts/appendix.md"
echo "Multimedia Appendix 2 ZIP:  $OUT_DIR/supplementary/multimedia_appendix_2.zip"
echo ""
echo "Per-message predictions:    $REPO_DIR/predictions/canonical/"
echo "Canonical metrics CSVs:     $REPO_DIR/results/"
echo "Rendered tables:            $REPO_DIR/results/tables/"
echo "Rendered figures:           $REPO_DIR/results/figures/"
echo ""
echo "Next steps:"
echo "  1. Corresponding-author manual cold-read (see PRE_SUBMISSION_CHECKLIST.md)"
echo "  2. Fill in any TBD metadata in results/manuscript_metadata.json"
echo "  3. Upload to https://bmcmedinformdecismak.biomedcentral.com"
