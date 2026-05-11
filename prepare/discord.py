import json
import re
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import login

HF_DATASET = "mookiezi/Discord-Dialogues"
N = 50_000
OUT = Path(__file__).parent.parent / "datasets" / "discord" / "sft.jsonl"

TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)


def parse_chatml(text: str) -> list[dict]:
    return [
        {"role": m.group(1), "content": m.group(2).strip()}
        for m in TURN_RE.finditer(text)
    ]


def main():
    login()

    ds = load_dataset(HF_DATASET, split="train")
    subset = ds.select(range(N))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    skipped = 0
    with open(OUT, "w") as f:
        for ex in subset:
            messages = parse_chatml(ex["text"])
            if len(messages) < 2:
                skipped += 1
                continue
            f.write(json.dumps({"messages": messages}) + "\n")

    saved = N - skipped
    print(f"Saved {saved} examples to {OUT} ({skipped} skipped — fewer than 2 turns)")


if __name__ == "__main__":
    main()