#!/bin/bash
# Re-runs only the steps that depend on customer value after the CLV fix.
set -u
cd /Users/emrahfidan/Desktop/MAKALE-2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
PY=./.venv/bin/python
LOG=outputs/logs/rerun_profit.log
echo "=== CLV DUZELTMESI SONRASI KISMI RERUN: $(date '+%H:%M:%S') ===" > "$LOG"
for ad in adim5_kar_roi_rq3 adim7_saglamlik adim8_profit_baseline adim9_imza_bulgu; do
  echo "=== $(date '+%H:%M') calisiyor: $ad ===" | tee -a "$LOG"
  $PY -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
      "notebooks/$ad.ipynb" > "outputs/logs/${ad}_nbconvert.log" 2>&1
  kod=$?
  echo "   exit=$kod bitti: $(date '+%H:%M')" | tee -a "$LOG"
  if [ $kod -ne 0 ]; then
    echo "   HATA: $ad" | tee -a "$LOG"
    tail -20 "outputs/logs/${ad}_nbconvert.log" | tee -a "$LOG"
    exit $kod
  fi
done
echo "=== KISMI RERUN TAMAMLANDI: $(date '+%H:%M:%S') ===" | tee -a "$LOG"

