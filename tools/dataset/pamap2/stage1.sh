for fold in 1 2 3 4 5 6 7 8 9; do
  python3 scripts/MSL/stage1_partition.py --config configs/MSL/pamap2.config --fold $fold
done
