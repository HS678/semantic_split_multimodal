#!/usr/bin/env bash
set -euo pipefail

mkdir -p tools/logs/uci_har tools/logs/mhealth tools/logs/pamap2 tools/logs/iemocap

echo "discover_modalities: uci_har"
bash tools/dataset/uci_har/discover_modalities.sh > tools/logs/uci_har/discover_modalities.log 2>&1

echo "discover_modalities: mhealth"
bash tools/dataset/mhealth/discover_modalities.sh > tools/logs/mhealth/discover_modalities.log 2>&1

echo "discover_modalities: pamap2"
bash tools/dataset/pamap2/discover_modalities.sh > tools/logs/pamap2/discover_modalities.log 2>&1

echo "discover_modalities: iemocap"
bash tools/dataset/iemocap/discover_modalities.sh > tools/logs/iemocap/discover_modalities.log 2>&1

echo "discover_modalities all finished"
