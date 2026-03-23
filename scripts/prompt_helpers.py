def _build_newyorker_prompt(sample: dict) -> str:
    description = str(sample.get("image_description", "")).strip()
    if description:
        return f"""You are looking at an image.

Image description:
{description}

Caption A:
{sample['joke_a_text']}

Caption B:
{sample['joke_b_text']}

Which caption is funnier for this image?
Answer with only "A" or "B"."""
    else:
        print("error, no image description")


def _build_jester_prompt(sample: dict) -> str:
    return f"""Which joke is funnier?

Joke A:
{sample['joke_a_text']}

Joke B:
{sample['joke_b_text']}

Answer with only "A" or "B"."""


def build_prompt(sample: dict) -> str:
    dataset = sample.get("dataset", "")

    if dataset == "newyorker":
        return _build_newyorker_prompt(sample)
    elif dataset == "jester":
        return _build_jester_prompt(sample)
    
    raise ValueError(f"Unknown dataset '{dataset}' for prompt building.")
