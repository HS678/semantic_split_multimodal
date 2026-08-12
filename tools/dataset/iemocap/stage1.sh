for fold in 1 2 3 4 5; do
  python3 scripts/MSL/stage1_partition.py --config configs/MSL/iemocap.config --fold $fold
done
