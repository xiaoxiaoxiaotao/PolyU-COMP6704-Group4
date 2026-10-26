#!/usr/bin/env bash
# Wait for a running experiment campaign to finish, then finalize and publish.
#
#   TEMPLATE=/path/to/6704-Group4.pptx experiments/finalize_when_done.sh
#
# It polls for the campaign process to disappear, runs experiments/finalize_results.py
# (correctness refresh, re-aggregation, report figures, deck rebuild with --check),
# and only commits + pushes if that step exits successfully. Nothing is committed
# on failure, so a broken run can never publish half-written results.
set -uo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
TEMPLATE="${TEMPLATE:-}"
DESKTOP_COPY="${DESKTOP_COPY:-}"
PATTERN="${PATTERN:-run_main.py --config configs/main_full.yaml}"
LOG="results/logs/finalize_watch.log"
POLL_SECONDS="${POLL_SECONDS:-300}"

mkdir -p results/logs

echo "watcher started $(date -Is) waiting for: ${PATTERN}" >> "${LOG}"
while pgrep -f "${PATTERN}" > /dev/null; do
  sleep "${POLL_SECONDS}"
done
echo "campaign finished $(date -Is); finalizing" >> "${LOG}"

if [ -n "${TEMPLATE}" ]; then
  "${PYTHON}" experiments/finalize_results.py --template "${TEMPLATE}" >> "${LOG}" 2>&1
  STATUS=$?
else
  "${PYTHON}" experiments/finalize_results.py --skip-deck >> "${LOG}" 2>&1
  STATUS=$?
fi

if [ "${STATUS}" -ne 0 ]; then
  echo "finalize failed with status ${STATUS}; not committing $(date -Is)" >> "${LOG}"
  exit "${STATUS}"
fi

if [ -n "${DESKTOP_COPY}" ]; then
  cp slides/COMP6704_Group4_PART_IV.pptx "${DESKTOP_COPY}" \
    && echo "deck copied to ${DESKTOP_COPY}" >> "${LOG}"
fi

git add results/tables results/figures slides/COMP6704_Group4_PART_IV.pptx
git commit -m "results: finalize the main ladder and rebuild the PART IV slides" >> "${LOG}" 2>&1
git push origin main >> "${LOG}" 2>&1
echo "committed and pushed $(date -Is)" >> "${LOG}"
