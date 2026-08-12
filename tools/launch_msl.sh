max_jobs=${MAX_JOBS:-2}
job_count=0

run_job() {
  "$@" &
  job_count=$((job_count + 1))
  if [ "$job_count" -ge "$max_jobs" ]; then
    wait
    job_count=0
  fi
}

for seed in 101 202 303 404 505; do
  run_job python3 scripts/MSL/stage3_train.py --config configs/MSL/uci_har.config --seed $seed
done

for fold in 1 2 3 4 5; do
  run_job python3 scripts/MSL/stage3_train.py --config configs/MSL/mhealth.config --fold $fold --seed 42
done

for fold in 1 2 3 4 5 6 7 8 9; do
  run_job python3 scripts/MSL/stage3_train.py --config configs/MSL/pamap2.config --fold $fold --seed 42
done

for fold in 1 2 3 4 5; do
  run_job python3 scripts/MSL/stage3_train.py --config configs/MSL/iemocap.config --fold $fold --seed 42
done

wait
python3 scripts/MSL/summarize_results.py --results-root results/MSL
