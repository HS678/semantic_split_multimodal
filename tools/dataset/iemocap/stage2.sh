for fold in 1 2 3 4 5; do
  python3 scripts/MSL/stage2_discovery.py --dataset iemocap --fold $fold
done
