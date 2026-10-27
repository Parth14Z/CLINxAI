from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
from .config import RAW_DIR, SPLIT_DIR, SEED

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

def prepare_splits(raw_dir=RAW_DIR, split_dir=SPLIT_DIR,
                   train_ratio=0.70, val_ratio=0.15, seed=SEED):
    """Create stratified image-level train/validation/test folders.
    Expected raw layout: data/raw/<class_name>/<image files>.
    If patient/eye IDs exist, use patient/eye-level splitting instead.
    """
    raw_dir, split_dir = Path(raw_dir), Path(split_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset folder not found: {raw_dir}. Place images in data/raw/<class_name>/."
        )
    classes = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if len(classes) < 2:
        raise ValueError("Expected at least two class folders inside data/raw/.")
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for split in ("train", "val", "test"):
        for cls in classes:
            (split_dir / split / cls.name).mkdir(parents=True, exist_ok=True)

    total = 0
    for cls in classes:
        paths = [p for p in cls.rglob("*")
                 if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
        paths = [p for p in paths if verify_image(p)]
        if len(paths) < 3:
            raise ValueError(f"Class '{cls.name}' has only {len(paths)} valid images.")
        train_val, test = train_test_split(
            paths, test_size=1-train_ratio-val_ratio, random_state=seed, shuffle=True
        )
        val_fraction = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, test_size=val_fraction, random_state=seed, shuffle=True
        )
        for split_name, subset in (("train", train), ("val", val), ("test", test)):
            for src in subset:
                dst = split_dir / split_name / cls.name / src.name
                if dst.exists():
                    dst = dst.with_name(f"{src.stem}_{abs(hash(str(src))) % 100000}{src.suffix}")
                shutil.copy2(src, dst)
        print(f"{cls.name}: train={len(train)}, val={len(val)}, test={len(test)}")
        total += len(paths)
    print(f"Prepared {total} valid images in {split_dir}")
    print("Warning: this is image-level splitting, not patient-level splitting.")
