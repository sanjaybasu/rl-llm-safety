#!/bin/bash
set -e
OUT_DIR=/Users/sanjaybasu/waymark-local/notebooks/rl_vs_llm_safety_v3
DOCX_DIR=${OUT_DIR}/docx
DRAFTS_DIR=${OUT_DIR}/drafts
REPO=/Users/sanjaybasu/waymark-local/packaging/rl_llm_safety_github_v3
REFERENCE_DOC=/Users/sanjaybasu/.claude/templates/sanjay_paper_reference.docx

rm -f ${DOCX_DIR}/*.docx
mkdir -p ${DOCX_DIR}/figures
cp ${REPO}/results/figures/*.png ${DOCX_DIR}/figures/
for f in main_text cover_letter appendix title_page; do
  cp ${DRAFTS_DIR}/${f}.md ${DOCX_DIR}/${f}.md
done
cd ${DOCX_DIR}
REFERENCE=""
[ -f "${REFERENCE_DOC}" ] && REFERENCE="--reference-doc=${REFERENCE_DOC}"
pandoc -o "Basu_BMC-MIDM_main-manuscript-anonymized_AI-safety-Medicaid-messaging.docx" main_text.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_multimedia-appendix-1_supplementary-methods-and-tables.docx" appendix.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_cover-letter.docx" cover_letter.md ${REFERENCE}
pandoc -o "Basu_BMC-MIDM_title-page.docx" title_page.md ${REFERENCE}
cp ${REPO}/results_modal_pull/supplementary/multimedia_appendix_2.zip \
   "Basu_BMC-MIDM_multimedia-appendix-2_per-message-predictions.zip"
rm -f main_text.md cover_letter.md appendix.md title_page.md
ls -la ${DOCX_DIR}/Basu_BMC-MIDM_*
