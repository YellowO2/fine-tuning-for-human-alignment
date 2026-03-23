from interact import ask_llm
from result_io import build_output_file, save_accuracy_summary, save_results
from dataset_helpers import load_examples
from prompt_helpers import build_prompt

# Parameters
THINK = False
MODEL = "qwen3:1.7b" # available models: qwen3:14b, qwen3:4b-instruct, qwen3:8b, qwen3:32b, qwen3:1.7
DATASET = "jester"  # "jester" or "newyorker"

# Jester options
JESTER_DATASET_NUM = 4
JESTER_OFFSET_DIVISOR = 2

# New Yorker options
NYCC_SPLIT = "validation"  # train / validation / test
NYCC_FOLD = 0  # 0..4 where 0 corresponds to "ranking"
NYCC_LIMIT = 150 # e.g., 10 for quick test
examples = load_examples(
    dataset=DATASET,
    jester_dataset_num=JESTER_DATASET_NUM,
    jester_offset_divisor=JESTER_OFFSET_DIVISOR,
    nycc_split=NYCC_SPLIT,
    nycc_fold=NYCC_FOLD,
    nycc_limit=NYCC_LIMIT,
)

results = []
OUTPUT_FILE = build_output_file(MODEL, is_thinking=THINK, output_dir=f"results/{DATASET}")
print(f"Starting evaluation with {len(examples)} comparisons...\n")

for i, sample in enumerate(examples):
    prompt = build_prompt(sample)

    expected = sample["expected"]

    try:
        llm_output = ask_llm(prompt, model=MODEL, think=THINK)
        response = llm_output["content"]
        thinking = llm_output["thinking"]
        response_clean = response.replace(".", "").replace('"', "").strip()
        is_correct = response_clean.lower() == expected.lower()

        status = "✓" if is_correct else "✗"
        print(f"[{i + 1}] {sample['meta']}: LLM={response_clean} Expected={expected} {status}")

        results.append(
            {
                "dataset": DATASET,
                "sample_id": sample["sample_id"],
                "meta": sample["meta"],
                "expected": expected,
                "llm_response": response,
                "llm_thinking": thinking,
                "is_correct": is_correct,
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        break

# Save results
results_df = save_results(results, OUTPUT_FILE)

if not results_df.empty:
    accuracy = results_df['is_correct'].mean() * 100
    summary_file = save_accuracy_summary(
        model_name=MODEL,
        output_file=OUTPUT_FILE,
        input_dataset_file=f"{DATASET}_{NYCC_SPLIT}_fold{NYCC_FOLD}.csv" if DATASET == "newyorker" else f"dataset{JESTER_DATASET_NUM}",
        is_thinking=THINK,
        accuracy_percent=accuracy,
        correct_count=int(results_df['is_correct'].sum()),
        total_count=len(results_df),
        summary_file=f"results/{DATASET}/accuracy_summary.csv",
    )
    print(f"\n✓ Results saved to {OUTPUT_FILE}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Accuracy summary updated: {summary_file}")

