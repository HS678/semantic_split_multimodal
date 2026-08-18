for fold in 1 2 3 4 5; do
  python3 pipeline/prepare_clients.py --dataset mhealth --fold $fold
done
