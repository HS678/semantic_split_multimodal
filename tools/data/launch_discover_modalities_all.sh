#!/usr/bin/env bash

mkdir -p tools/logs/uci_har tools/logs/mhealth tools/logs/pamap2 tools/logs/iemocap

bash tools/dataset/uci_har/discover_modalities.sh > tools/logs/uci_har/discover_modalities.log 2>&1 &
pid_uci_har=$!

bash tools/dataset/mhealth/discover_modalities.sh > tools/logs/mhealth/discover_modalities.log 2>&1 &
pid_mhealth=$!

bash tools/dataset/pamap2/discover_modalities.sh > tools/logs/pamap2/discover_modalities.log 2>&1 &
pid_pamap2=$!

bash tools/dataset/iemocap/discover_modalities.sh > tools/logs/iemocap/discover_modalities.log 2>&1 &
pid_iemocap=$!

wait "$pid_uci_har"
status_uci_har=$?

wait "$pid_mhealth"
status_mhealth=$?

wait "$pid_pamap2"
status_pamap2=$?

wait "$pid_iemocap"
status_iemocap=$?

echo "uci_har discover_modalities status: $status_uci_har"
echo "mhealth discover_modalities status: $status_mhealth"
echo "pamap2 discover_modalities status: $status_pamap2"
echo "iemocap discover_modalities status: $status_iemocap"

if [ "$status_uci_har" -ne 0 ] || [ "$status_mhealth" -ne 0 ] || [ "$status_pamap2" -ne 0 ] || [ "$status_iemocap" -ne 0 ]; then
  exit 1
fi
