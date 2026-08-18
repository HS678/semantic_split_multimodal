import importlib


def test_semantic_entrypoints_import():
    for module_name in [
        "pipeline.prepare_clients",
        "pipeline.discover_modalities",
        "experiments.discovery_comparison",
        "experiments.run_all",
        "experiments.run_all_discovery",
        "experiments.training",
        "experiments.msl.train",
        "experiments.msl.run_all",
        "experiments.baselines.random_sl",
        "experiments.baselines.kmeans_sl",
        "experiments.baselines.oracle_sl",
    ]:
        assert importlib.import_module(module_name)


def test_kmeans_baseline_wrapper_maps_k_to_shared_method(monkeypatch):
    from experiments.baselines import kmeans_sl

    captured = {}

    def fake_train_main(argv):
        captured["argv"] = list(argv)

    monkeypatch.setattr(kmeans_sl, "train_main", fake_train_main)
    kmeans_sl.main(["--k", "4", "--dataset", "mhealth", "--fold", "1", "--seed", "42"])
    assert captured["argv"] == ["--method", "kmeans4", "--dataset", "mhealth", "--fold", "1", "--seed", "42"]


def test_default_baseline_wrappers_select_shared_methods(monkeypatch):
    from experiments.baselines import oracle_sl, random_sl

    captured = {}

    def fake_random(argv):
        captured["random"] = list(argv)

    def fake_oracle(argv):
        captured["oracle"] = list(argv)

    monkeypatch.setattr(random_sl, "train_main", fake_random)
    monkeypatch.setattr(oracle_sl, "train_main", fake_oracle)
    random_sl.main(["--dataset", "uci_har", "--seed", "42"])
    oracle_sl.main(["--dataset", "uci_har", "--seed", "42"])
    assert captured["random"] == ["--method", "randomsl", "--dataset", "uci_har", "--seed", "42"]
    assert captured["oracle"] == ["--method", "oracle", "--dataset", "uci_har", "--seed", "42"]


def test_training_config_hash_has_no_topology_side_effect(monkeypatch, tmp_path):
    from experiments import training

    def forbidden_prepare(*_args, **_kwargs):
        raise AssertionError("expected_training_config_hash must not prepare topology artifacts")

    monkeypatch.setattr(training, "find_clients_dir", lambda *_args, **_kwargs: tmp_path / "clients")
    monkeypatch.setattr(training, "find_discovery_dir", lambda *_args, **_kwargs: tmp_path / "discovery")
    monkeypatch.setattr(training, "prepare_method_topology", forbidden_prepare)

    value = training.expected_training_config_hash(
        "mhealth",
        1,
        42,
        "ours",
        tmp_path / "results",
        None,
    )
    assert isinstance(value, str)
