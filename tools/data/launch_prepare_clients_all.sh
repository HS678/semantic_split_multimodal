#!/usr/bin/env bash
set -euo pipefail

mkdir -p tools/logs/uci_har tools/logs/mhealth tools/logs/pamap2 tools/logs/iemocap

echo "prepare_clients: uci_har"
bash tools/dataset/uci_har/prepare_clients.sh > tools/logs/uci_har/prepare_clients.log 2>&1

echo "prepare_clients: mhealth"
bash tools/dataset/mhealth/prepare_clients.sh > tools/logs/mhealth/prepare_clients.log 2>&1

echo "prepare_clients: pamap2"
bash tools/dataset/pamap2/prepare_clients.sh > tools/logs/pamap2/prepare_clients.log 2>&1

echo "prepare_clients: iemocap"
bash tools/dataset/iemocap/prepare_clients.sh > tools/logs/iemocap/prepare_clients.log 2>&1

echo "prepare_clients all finished"
