import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset, load_from_disk
from huggingface_hub import login

HF_DATASET = "jmhessel/newyorker_caption_contest"
FOLDS = [0, 1, 2, 3, 4]
SPLITS = ["train", "validation", "test"]

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "newyorker"
RAW_DIR = DATASETS_DIR / "raw"


def fold_config(fold: int) -> str:
    return "ranking" if fold == 0 else f"ranking_{fold}"


def normalize_row(row: dict, split: str) -> dict:
    return {
        "sample_id": row["instance_id"],
        "caption_a": row["caption_choices"][0],
        "caption_b": row["caption_choices"][1],
        "expected": row["label"],
        "winner_source": row["winner_source"],
        "image_description": row.get("image_description", ""),
        "image_uncanny_description": row.get("image_uncanny_description", ""),
    }


def prepare_fold(fold: int) -> None:
    cfg = fold_config(fold)
    raw_path = RAW_DIR / cfg

    if raw_path.exists():
        print(f"  Loading fold {fold} from local snapshot...")
        dset = load_from_disk(str(raw_path))
    else:
        print(f"  Downloading fold {fold} from HuggingFace...")
        dset = load_dataset(HF_DATASET, cfg, token=os.getenv("HF_TOKEN"))
        raw_path.mkdir(parents=True, exist_ok=True)
        dset.save_to_disk(str(raw_path))
        print(f"    Cached to {raw_path}")

    for split in SPLITS:
        rows = [normalize_row(row, split) for row in dset[split]]
        out = DATASETS_DIR / f"fold{fold}_{split}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"    {split}: {len(rows)} rows -> {out.name}")


def main():
    if not os.getenv("HF_TOKEN"):
        login()

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for fold in FOLDS:
        print(f"Fold {fold}:")
        prepare_fold(fold)

    print("Done.")


if __name__ == "__main__":
    main()