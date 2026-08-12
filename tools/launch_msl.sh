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
  run_job python3 scripts/MSL/stage3_train.py --dataset uci_har --seed $seed
done

for fold in 1 2 3 4 5; do
  run_job python3 scripts/MSL/stage3_train.py --dataset mhealth --fold $fold --seed 42
done

for fold in 1 2 3 4 5 6 7 8 9; do
  run_job python3 scripts/MSL/stage3_train.py --dataset pamap2 --fold $fold --seed 42
done

for fold in 1 2 3 4 5; do
  run_job python3 scripts/MSL/stage3_train.py --dataset iemocap --fold $fold --seed 42
done

wait
python3 scripts/MSL/summarize_results.py --results-root results/MSL
