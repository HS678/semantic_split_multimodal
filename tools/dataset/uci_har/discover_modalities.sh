for seed in 42 123 2025 3407 7777; do
  python pipeline/discover_modalities.py \
    --dataset uci_har \
    --seed $seed
done
