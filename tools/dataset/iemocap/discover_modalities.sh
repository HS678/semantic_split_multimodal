for fold in 1 2 3 4 5; do
  python3 pipeline/discover_modalities.py --dataset iemocap --fold $fold
done
