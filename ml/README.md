# ML Pipeline

Training, evaluation, threshold tuning, export and a lightweight model registry
for the PCB defect detector. All scripts are run as modules from the **project
root** so imports resolve:

```bash
python -m ml.<module>
```

## Layout

| Path | Purpose |
|---|---|
| `config.py` | Central paths, class/severity map, `TrainConfig` (hyper-params + augmentation) |
| `training/train_yolo.py` | Train, deploy best weights to `models/`, register |
| `evaluation/evaluate_model.py` | Metrics → markdown + JSON, register metrics |
| `evaluation/tune_thresholds.py` | Per-class threshold tuning → backend policy |
| `export/export_model.py` | ONNX / TensorRT export + parity check |
| `registry/registry.py` | JSON model registry (version, checksum, metrics, thresholds) |
| `datasets/industrial_defects/` | Roboflow PCB dataset + `data.yaml` |

## Typical workflow

```bash
# 1. Inspect the dataset
python scripts/inspect_dataset.py

# 2. Train (defect-aware augmentation, 150-epoch schedule by default)
python -m ml.training.train_yolo --version pcb_defect_v2 --epochs 150 --batch 16

# 3. Evaluate on the held-out test split (writes report + JSON, registers metrics)
python -m ml.evaluation.evaluate_model --model ml/models/pcb_defect_v2.pt --split test

# 4. Tune per-class thresholds (writes recommended_thresholds.json)
python -m ml.evaluation.tune_thresholds --model ml/models/pcb_defect_v2.pt --split val

# 5. Export for deployment
python -m ml.export.export_model --model ml/models/pcb_defect_v2.pt --format onnx
```

## How tuning feeds production

`tune_thresholds` picks, per class, the confidence threshold that maximises an
**F-beta** score. Beta is set by severity (`config.SEVERITY_FBETA`):

- **critical** (`short`, `open_circuit`, `missing_hole`) → beta = 2.0 (favour recall)
- **major** → beta = 1.5
- **minor** → beta = 1.0

An optional `--precision-floor` prevents picking a threshold with an unusable
false-call rate. The result is written to
`registry/recommended_thresholds.json`, which the **backend policy engine loads
automatically on startup** — so re-tuning after a retrain updates the live
PASS/FAIL decision thresholds with no code change.

> Recall is the metric that matters for a quality gate: it is the fraction of
> real defects caught. mAP is a model-quality proxy; the threshold tuner is how
> you convert model quality into an acceptable escape rate.

## Model registry

Every train/evaluate run records an entry in `registry/models.json`: version,
weights path + SHA-256, metrics, training config and thresholds. This gives
reproducible, auditable provenance for any model that produced a verdict.

## Notes

- Training needs a CUDA GPU for reasonable speed; the scripts fall back to CPU.
- Model weights (`*.pt`, `*.onnx`) and the dataset are git-ignored — track them
  with DVC or a model/artifact store, not git.
- The current baseline (`pcb_defect_yolo_v1`) has recall ~0.61 on test; reaching
  a factory-grade >99% recall on critical defects requires a substantially
  larger, augmented dataset — that is the main remaining ML task.
