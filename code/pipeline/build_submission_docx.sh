#!/bin/bash
set -e
OUT_DIR=/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3
DOCX_DIR=${OUT_DIR}/docx
DRAFTS_DIR=${OUT_DIR}/drafts
REPO=/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
REFERENCE_DOC=/Users/sanjaybasu/.claude/templates/sanjay_paper_reference.docx

rm -f ${DOCX_DIR}/*.docx ${DOCX_DIR}/*.zip
mkdir -p ${DOCX_DIR}/figures
cp ${REPO}/results/figures/*.png ${DOCX_DIR}/figures/
# BMC MIDM convention: ONE combined main manuscript file (title + authors +
# affiliations + abstract + body + declarations + references + figure legends
# + tables). No separate title page; no anonymized variant (BMC uses
# single-blind review by default). Appendix is "additional file 1".
for f in main_text cover_letter appendix; do
  cp ${DRAFTS_DIR}/${f}.md ${DOCX_DIR}/${f}.md
done
cd ${DOCX_DIR}
REFERENCE=""
[ -f "${REFERENCE_DOC}" ] && REFERENCE="--reference-doc=${REFERENCE_DOC}"
pandoc -o "Basu_BMC-MIDM_main-manuscript_AI-safety-Medicaid-messaging.docx" main_text.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_additional-file-1_supplementary-methods-figures-tables.docx" appendix.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_cover-letter.docx" cover_letter.md ${REFERENCE}
# NOTE: per-message-prediction ZIP (results_modal_pull/supplementary/multimedia_appendix_2.zip)
# is NOT included in the submission bundle. It is retained for local audit/revision
# (opaque CUIDs only; reviewers cannot map back to messages, so submission value is limited).
rm -f main_text.md cover_letter.md appendix.md
echo "=== Submission bundle ==="
ls -la ${DOCX_DIR}/Basu_BMC-MIDM_*
