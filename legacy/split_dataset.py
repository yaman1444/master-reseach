"""
Stratified Dataset Split: Train / Val / Test (70 / 15 / 15)
============================================================
Ensures:
  - No file overlap between splits (verified)
  - Stratified by class (preserves class distribution)
  - Reproducible (seed=42)
  - Copies files to new directory structure

Usage:
  python split_dataset.py                        # Execute split
  python split_dataset.py --verify               # Verify integrity only
  python split_dataset.py --source ../datasets/train --dest ../datasets_split
"""
import os
import shutil
import argparse
import random
from pathlib import Path
from collections import Counter

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = ['debut', 'grave', 'normal']
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def collect_images(source_dir):
    """Collect all image paths grouped by class."""
    class_images = {}
    source = Path(source_dir)

    for class_name in CLASS_NAMES:
        class_dir = source / class_name
        if not class_dir.exists():
            print(f"  ⚠️  Classe '{class_name}' non trouvée dans {source_dir}")
            continue

        images = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTENSIONS
        ])
        class_images[class_name] = images
        print(f"  {class_name:10s}: {len(images):4d} images")

    return class_images


def stratified_split(class_images):
    """Split each class independently to preserve distribution."""
    random.seed(SEED)

    splits = {'train': {}, 'val': {}, 'test': {}}

    print(f"\n📊 Split ratios: train={TRAIN_RATIO:.0%} / val={VAL_RATIO:.0%} / test={TEST_RATIO:.0%}")
    print(f"{'':2s} {'Classe':10s} {'Total':>6s} {'Train':>6s} {'Val':>6s} {'Test':>6s}")
    print("  " + "-" * 44)

    for class_name, images in class_images.items():
        # Shuffle with fixed seed
        shuffled = images.copy()
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Rest goes to test (handles rounding)
        n_test = n - n_train - n_val

        splits['train'][class_name] = shuffled[:n_train]
        splits['val'][class_name] = shuffled[n_train:n_train + n_val]
        splits['test'][class_name] = shuffled[n_train + n_val:]

        print(f"  {class_name:10s} {n:6d} {n_train:6d} {n_val:6d} {n_test:6d}")

    return splits


def copy_split(splits, dest_dir):
    """Copy files to train/val/test directories."""
    dest = Path(dest_dir)

    # Clean destination
    if dest.exists():
        print(f"\n🗑️  Suppression de l'ancien split: {dest}")
        shutil.rmtree(dest)

    total_copied = 0
    for split_name, class_dict in splits.items():
        for class_name, files in class_dict.items():
            target_dir = dest / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for src_path in files:
                dst_path = target_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                total_copied += 1

    print(f"\n✅ {total_copied} fichiers copiés vers {dest}")
    return total_copied


def verify_integrity(dest_dir):
    """Verify no file overlap between splits."""
    dest = Path(dest_dir)
    split_names = ['train', 'val', 'test']

    print(f"\n🔍 Vérification d'intégrité de {dest}...")

    # Collect all filenames per split
    split_files = {}
    for split in split_names:
        files = set()
        split_dir = dest / split
        if not split_dir.exists():
            print(f"  ❌ {split}/ n'existe pas!")
            return False
        for class_dir in split_dir.iterdir():
            if class_dir.is_dir():
                for f in class_dir.iterdir():
                    files.add(f.name)
        split_files[split] = files
        print(f"  {split:6s}: {len(files):4d} fichiers uniques")

    # Check overlaps
    all_ok = True
    for i, s1 in enumerate(split_names):
        for s2 in split_names[i+1:]:
            overlap = split_files[s1] & split_files[s2]
            if overlap:
                print(f"  ❌ FUITE DETECTÉE: {len(overlap)} fichiers communs entre {s1} et {s2}")
                for f in list(overlap)[:5]:
                    print(f"     - {f}")
                all_ok = False
            else:
                print(f"  ✅ {s1} ∩ {s2} = ∅ (aucun overlap)")

    # Class distribution
    print(f"\n📊 Distribution par split:")
    print(f"  {'':6s} ", end="")
    for c in CLASS_NAMES:
        print(f"{c:>10s}", end="")
    print(f"{'Total':>10s}{'% classe min':>14s}")

    for split in split_names:
        split_dir = dest / split
        counts = {}
        total = 0
        for c in CLASS_NAMES:
            class_dir = split_dir / c
            n = len(list(class_dir.glob('*'))) if class_dir.exists() else 0
            counts[c] = n
            total += n

        print(f"  {split:6s} ", end="")
        for c in CLASS_NAMES:
            pct = counts[c] / total * 100 if total > 0 else 0
            print(f"{counts[c]:6d}({pct:4.1f}%)", end="")
        print(f"{total:10d}", end="")

        if total > 0:
            min_pct = min(counts[c] / total * 100 for c in CLASS_NAMES)
            print(f"{min_pct:13.1f}%")
        else:
            print()

    if all_ok:
        print(f"\n✅ INTÉGRITÉ VÉRIFIÉE: Aucune fuite de données détectée!")
    else:
        print(f"\n❌ INTÉGRITÉ COMPROMISE: Fuites de données détectées!")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description='Stratified Dataset Split')
    parser.add_argument('--source', type=str, default='../datasets/train',
                        help='Source directory with class subdirectories')
    parser.add_argument('--dest', type=str, default='../datasets_split',
                        help='Destination directory for split dataset')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify existing split integrity')
    args = parser.parse_args()

    print("=" * 60)
    print("STRATIFIED DATASET SPLIT (70/15/15)")
    print("=" * 60)

    if args.verify:
        verify_integrity(args.dest)
        return

    # Step 1: Collect images
    print(f"\n📂 Source: {args.source}")
    class_images = collect_images(args.source)

    if not class_images:
        print("❌ Aucune image trouvée!")
        return

    # Step 2: Stratified split
    splits = stratified_split(class_images)

    # Step 3: Copy files
    copy_split(splits, args.dest)

    # Step 4: Verify
    verify_integrity(args.dest)

    print(f"\n{'=' * 60}")
    print("✅ SPLIT TERMINÉ")
    print(f"{'=' * 60}")
    print(f"\nUtilisation:")
    print(f"  Train: {args.dest}/train/")
    print(f"  Val:   {args.dest}/val/")
    print(f"  Test:  {args.dest}/test/")


if __name__ == '__main__':
    main()
