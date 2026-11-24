from pathlib import Path
import yaml

DATASET_DIR = Path("ml/datasets/industrial_defects")
DATA_YAML = DATASET_DIR / "data.yaml"

def count_files(path: Path, suffixes):
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in suffixes)

print("=" * 60)
print("Industrial Defect Dataset Inspection")
print("=" * 60)

if not DATASET_DIR.exists():
    print(f"❌ Dataset folder not found: {DATASET_DIR}")
    raise SystemExit(1)

if not DATA_YAML.exists():
    print(f"❌ data.yaml not found: {DATA_YAML}")
    raise SystemExit(1)

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print(f"Dataset path: {DATASET_DIR.resolve()}")
print(f"data.yaml found: {DATA_YAML}")

print("\nClasses:")
names = data.get("names", {})
if isinstance(names, dict):
    for k, v in names.items():
        print(f"  {k}: {v}")
elif isinstance(names, list):
    for i, v in enumerate(names):
        print(f"  {i}: {v}")
else:
    print("  ⚠️ Could not read class names")

print("\nFile counts:")
for split in ["train", "valid", "val", "test"]:
    images = DATASET_DIR / split / "images"
    labels = DATASET_DIR / split / "labels"

    image_count = count_files(images, {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    label_count = count_files(labels, {".txt"})

    if image_count or label_count:
        print(f"  {split}: {image_count} images, {label_count} labels")

print("\n✅ Dataset inspection complete.")