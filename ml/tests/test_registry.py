import ml.registry.registry as registry_module
from ml.registry.registry import (
    load_registry,
    register_model,
    get_model,
    latest_model,
    sha256_of,
)


def test_register_and_retrieve(tmp_path, monkeypatch):
    manifest = tmp_path / "models.json"
    monkeypatch.setattr(registry_module, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry_module, "REGISTRY_MANIFEST", manifest)

    weights = tmp_path / "model_a.pt"
    weights.write_bytes(b"fake-weights")

    entry = register_model(
        version="v1",
        weights_path=weights,
        metrics={"recall": 0.9},
        thresholds={"short": 0.2},
    )
    assert entry["sha256"] == sha256_of(weights)
    assert get_model("v1")["metrics"]["recall"] == 0.9
    assert len(load_registry()["models"]) == 1


def test_register_same_version_replaces(tmp_path, monkeypatch):
    manifest = tmp_path / "models.json"
    monkeypatch.setattr(registry_module, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry_module, "REGISTRY_MANIFEST", manifest)

    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    register_model(version="v1", weights_path=weights, metrics={"recall": 0.5})
    register_model(version="v1", weights_path=weights, metrics={"recall": 0.8})

    models = load_registry()["models"]
    assert len(models) == 1
    assert models[0]["metrics"]["recall"] == 0.8


def test_latest_model(tmp_path, monkeypatch):
    manifest = tmp_path / "models.json"
    monkeypatch.setattr(registry_module, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry_module, "REGISTRY_MANIFEST", manifest)

    weights = tmp_path / "m.pt"
    weights.write_bytes(b"x")
    register_model(version="v1", weights_path=weights)
    register_model(version="v2", weights_path=weights)
    assert latest_model()["version"] == "v2"
