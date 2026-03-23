import random
from pathlib import Path

import pandas as pd


def get_jester_examples(dataset_num: int, offset_divisor: int = 2): #dataset_num is either 3 or 4
    df = pd.read_csv(f"datasets/jester/cleaned_ranked_dataset{dataset_num}.csv")
    df = df.dropna(subset=["avg_rating", "joke_text"])
    offset = len(df) // offset_divisor

    examples = []
    for i in range(len(df) - offset):
        joke_better = df.iloc[i]
        joke_worse = df.iloc[i + offset]

        if random.choice([True, False]):
            joke_a_text = joke_better["joke_text"]
            joke_b_text = joke_worse["joke_text"]
            expected = "A"
        else:
            joke_a_text = joke_worse["joke_text"]
            joke_b_text = joke_better["joke_text"]
            expected = "B"

        examples.append(
            {
                "sample_id": f"jester-{dataset_num}-{i}",
                "meta": f"Rank {int(joke_better['rank'])} vs {int(joke_worse['rank'])}",
                "joke_a_text": joke_a_text,
                "joke_b_text": joke_b_text,
                "expected": expected,
                "dataset": "jester",
            }
        )
    return examples


def get_newyorker_examples(split: str = "validation", fold: int = 0, limit: int | None = None):
    local_path = Path(f"datasets/newyorker/ranking_fold{fold}_{split}.csv")
    if not local_path.exists():
        raise FileNotFoundError(
            f"Missing {local_path}. Run prepare_newyorker_dataset.py first."
        )

    local_df = pd.read_csv(local_path)
    if limit is not None:
        local_df = local_df.iloc[:limit]

    examples = []
    for _, row in local_df.iterrows():
        examples.append(
            {
                "sample_id": row["sample_id"],
                "meta": row["meta"],
                "joke_a_text": row["joke_a_text"],
                "joke_b_text": row["joke_b_text"],
                "expected": row["expected"],
                "winner_source": row.get("winner_source", ""),
                "image_description": row.get("image_description", ""),
                "dataset": "newyorker",
            }
        )
    return examples


def load_examples(*, dataset: str, jester_dataset_num: int, jester_offset_divisor: int, nycc_split: str, nycc_fold: int, nycc_limit: int | None):
    if dataset == "newyorker":
        return get_newyorker_examples(split=nycc_split, fold=nycc_fold, limit=nycc_limit)
    if dataset == "jester":
        return get_jester_examples(dataset_num=jester_dataset_num, offset_divisor=jester_offset_divisor)
    raise ValueError(f"Unknown DATASET '{dataset}'. Use 'jester' or 'newyorker'.")
