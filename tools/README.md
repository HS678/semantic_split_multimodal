# Tools

默认参数唯一来自：

```bash
src/MSL/protocol.py
```

推荐执行顺序：

```bash
prepare_clients -> discover_modalities -> experiments
```

单数据集：

```bash
bash tools/dataset/uci_har/prepare_clients.sh
bash tools/dataset/uci_har/discover_modalities.sh

bash tools/dataset/mhealth/prepare_clients.sh
bash tools/dataset/mhealth/discover_modalities.sh

bash tools/dataset/pamap2/prepare_clients.sh
bash tools/dataset/pamap2/discover_modalities.sh

bash tools/dataset/iemocap/prepare_clients.sh
bash tools/dataset/iemocap/discover_modalities.sh
```

全部数据集 pipeline：

```bash
bash tools/data/launch_prepare_clients_all.sh
bash tools/data/launch_discover_modalities_all.sh
```

实验：

```bash
python experiments/msl/run_all.py --results-root results --device auto --require-cuda
python experiments/run_all.py --results-root results --device auto --require-cuda
```

日志建议写入：

```bash
tools/logs/<dataset>/prepare_clients.log
tools/logs/<dataset>/discover_modalities.log
tools/logs/all/*.log
```
