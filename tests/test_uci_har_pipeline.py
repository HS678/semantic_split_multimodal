import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from utils.config import load_config
from utils.seed import set_seed
from data.uci_har_adapter import load_uci_har_dataset
from data.synthetic_dataset import build_client_pool
from trainers.stage2_trainer import Stage2Trainer
from utils.device import select_device


def test_uci_har_pipeline_smoke():
    cfg = load_config(str(ROOT / "configs" / "uci_har.yaml"))
    cfg = copy.deepcopy(cfg)

    # smoke-size run: 1 round x 1 local step
    cfg["global_rounds"] = 1
    cfg["local_steps_per_round"] = 1
    cfg["clients_per_modality"] = 4
    cfg["batch_size"] = 16
    cfg["device"] = "cpu"

    set_seed(cfg["seed"])
    device = select_device(cfg.get("device", "cpu"))

    split = load_uci_har_dataset(cfg, ROOT)
    train_clients_raw = build_client_pool(split["train"], cfg)

    trainer = Stage2Trainer(cfg, train_clients_raw, split["test"], device)
    trainer.run()

    # If we got here without exception/assertion, pipeline is healthy.
    assert True
