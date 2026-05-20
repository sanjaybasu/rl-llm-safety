#!/bin/bash
# Convert the rendered markdown drafts to DOCX with descriptive submission filenames.
# Output files match the naming convention expected by BMC MIDM submission.
#
# Naming convention (lead-author + journal-abbrev + file-type + content-key):
#   Basu_BMC-MIDM_main-manuscript_AI-safety-Medicaid-messaging.docx
#   Basu_BMC-MIDM_multimedia-appendix-1_supplementary-methods-and-tables.docx
#   Basu_BMC-MIDM_multimedia-appendix-2_per-message-predictions.zip  (copied; not converted)
#   Basu_BMC-MIDM_cover-letter.docx
#
# Prerequisites:
#   - pandoc installed (brew install pandoc)
#   - Phase D rendered the three markdown drafts to ${OUT_DIR}/drafts/
#   - Multimedia Appendix 2 ZIP exists at ${REPO}/results_modal_pull/supplementary/

set -e

OUT_DIR=/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3
DOCX_DIR=${OUT_DIR}/docx
DRAFTS_DIR=${OUT_DIR}/drafts
REPO=/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
REFERENCE_DOC=/Users/sanjaybasu/.claude/templates/sanjay_paper_reference.docx

mkdir -p ${DOCX_DIR}/figures
cp ${REPO}/results/figures/*.png ${DOCX_DIR}/figures/

# Stage rendered markdown into docx dir (pandoc reads from same directory as figures)
cp ${DRAFTS_DIR}/main_text.md ${DOCX_DIR}/main_text.md
cp ${DRAFTS_DIR}/cover_letter.md ${DOCX_DIR}/cover_letter.md
cp ${DRAFTS_DIR}/appendix.md ${DOCX_DIR}/appendix.md

cd ${DOCX_DIR}

REFERENCE=""
[ -f "${REFERENCE_DOC}" ] && REFERENCE="--reference-doc=${REFERENCE_DOC}"

# Convert with submission-ready descriptive filenames
pandoc -o "Basu_BMC-MIDM_main-manuscript_AI-safety-Medicaid-messaging.docx" main_text.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_multimedia-appendix-1_supplementary-methods-and-tables.docx" appendix.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_cover-letter.docx" cover_letter.md ${REFERENCE}

# Copy supplementary archive with descriptive name (do not re-zip)
cp ${REPO}/results_modal_pull/supplementary/multimedia_appendix_2.zip \
   "Basu_BMC-MIDM_multimedia-appendix-2_per-message-predictions.zip"

# Clean up the intermediate-stage markdown copies; keep only Basu_*.{docx,zip}
rm -f main_text.md cover_letter.md appendix.md

echo "=== Submission bundle ready in ${DOCX_DIR}/ ==="
ls -la ${DOCX_DIR}/Basu_BMC-MIDM_*
