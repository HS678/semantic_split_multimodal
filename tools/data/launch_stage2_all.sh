#!/usr/bin/env bash

mkdir -p tools/logs/uci_har tools/logs/mhealth tools/logs/pamap2 tools/logs/iemocap

bash tools/dataset/uci_har/stage2.sh > tools/logs/uci_har/stage2.log 2>&1 &
pid_uci_har=$!

bash tools/dataset/mhealth/stage2.sh > tools/logs/mhealth/stage2.log 2>&1 &
pid_mhealth=$!

bash tools/dataset/pamap2/stage2.sh > tools/logs/pamap2/stage2.log 2>&1 &
pid_pamap2=$!

bash tools/dataset/iemocap/stage2.sh > tools/logs/iemocap/stage2.log 2>&1 &
pid_iemocap=$!

wait "$pid_uci_har"
status_uci_har=$?

wait "$pid_mhealth"
status_mhealth=$?

wait "$pid_pamap2"
status_pamap2=$?

wait "$pid_iemocap"
status_iemocap=$?

echo "uci_har stage2 status: $status_uci_har"
echo "mhealth stage2 status: $status_mhealth"
echo "pamap2 stage2 status: $status_pamap2"
echo "iemocap stage2 status: $status_iemocap"

if [ "$status_uci_har" -ne 0 ] || [ "$status_mhealth" -ne 0 ] || [ "$status_pamap2" -ne 0 ] || [ "$status_iemocap" -ne 0 ]; then
  exit 1
fi
