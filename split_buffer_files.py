import os
import shutil
from pathlib import Path
import argparse
import random
import torch

def copy_files(files, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for f in files:
        print(f"Copying {f.name} -> {dest_dir}")
        shutil.copy(f, dest_dir)

def split_single_pt_file(pt_path: Path, train_dir: Path, val_dir: Path, ratio: float):
    print(f"Splitting single file: {pt_path}")
    payload = torch.load(pt_path)
    length = payload[0].shape[0]
    split_idx = int(length * ratio)
    split_idx = (split_idx // 400) * 400  # round down to multiple of 400


    start, end = [int(x) for x in pt_path.stem.split("_")]
    mid = start + split_idx

    train_payload = [arr[:split_idx] for arr in payload]
    val_payload = [arr[split_idx:] for arr in payload]

    train_out = train_dir / f"{start}_{mid - 1}.pt"
    val_out = val_dir / f"{mid}_{end}.pt"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    torch.save(train_payload, train_out)
    torch.save(val_payload, val_out)
    print(f"✅ Saved split to:\n  {train_out}\n  {val_out}")

def split_buffer_files(source_dir, train_dir, val_dir, ratio=0.8, seed=42):
    source_path = Path(source_dir)
    pt_files = sorted(source_path.glob("*.pt"))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    if not pt_files:
        print("❌ No .pt files found.")
        return

    if len(pt_files) == 1:
        split_single_pt_file(pt_files[0], Path(train_dir), Path(val_dir), ratio)
    else:
        random.seed(seed)
        random.shuffle(pt_files)
        split_idx = int(len(pt_files) * ratio)
        copy_files(pt_files[:split_idx], train_dir)
        copy_files(pt_files[split_idx:], val_dir)
        print(f"✅ Copied {split_idx} files to train, {len(pt_files) - split_idx} to val")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_buffer_files(args.source, args.train, args.val, args.ratio, args.seed)


