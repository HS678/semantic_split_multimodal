for fold in 1 2 3 4 5 6 7 8; do
  python3 pipeline/discover_modalities.py --dataset pamap2 --fold $fold
done
