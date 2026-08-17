for seed in 42 123 2025 3407 7777; do
  python scripts/MSL/stage2_discovery.py \
    --dataset uci_har \
    --seed $seed \
    --split-protocol subject_disjoint_70_30_seed${seed}
done
