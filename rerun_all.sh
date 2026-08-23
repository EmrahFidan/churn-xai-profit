#!/bin/bash
# Re-runs the full notebook chain in order and records timing per step.
# Notebooks are executed in place so that the committed cell outputs stay current.
set -u
cd /Users/emrahfidan/Desktop/MAKALE-2

# Nested parallelism (library OMP threads inside joblib workers) deadlocks on macOS,
# so the per-library thread pools are pinned to one thread and the parallelism is left
# to the randomized search layer.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PY=./.venv/bin/python
LOG=outputs/logs/rerun_all.log
mkdir -p outputs/logs
echo "=== RERUN baslangic: $(date '+%Y-%m-%d %H:%M:%S') ===" > "$LOG"

ADIMLAR=(
  adim1_tani_temizlik
  adim2_model_kalibrasyon
  adim3_dengeleme_rq1
  adim4_shap_rq2
  adim5_kar_roi_rq3
  adim6_transfer_probe
  adim7_saglamlik
  adim8_profit_baseline
  adim9_imza_bulgu
)

for ad in "${ADIMLAR[@]}"; do
  echo "=== $(date '+%H:%M') calisiyor: $ad ===" | tee -a "$LOG"
  $PY -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name=python3 \
      "notebooks/$ad.ipynb" > "outputs/logs/${ad}_nbconvert.log" 2>&1
  kod=$?
  echo "   exit=$kod bitti: $(date '+%H:%M')" | tee -a "$LOG"
  if [ $kod -ne 0 ]; then
    echo "   HATA: $ad basarisiz, zincir durduruldu" | tee -a "$LOG"
    tail -25 "outputs/logs/${ad}_nbconvert.log" | tee -a "$LOG"
    exit $kod
  fi
done

echo "=== TUM NOTEBOOKLAR TAMAMLANDI: $(date '+%H:%M:%S') ===" | tee -a "$LOG"

