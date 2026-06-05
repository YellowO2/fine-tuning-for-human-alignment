# ═══════════════════════════════════════════════════════════════════════════════
# Export Qwen 3.5 9B Discord LoRA -> GGUF (Ollama / llama.cpp / LM Studio)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Done in explicit, debuggable steps instead of unsloth's save_pretrained_gguf
# (which patches llama.cpp's convert script in a temp dir and breaks on newer
# llama.cpp -> "No module named 'conversion'"). Qwen has standard architecture,
# so the merge is lossless (unlike Gemma 4 E2B/E4B's KV-shared layers).
#
#   step 1 (this file): merge LoRA adapter -> full 16-bit HF model
#   step 2 (run_convert below): HF -> GGUF f16 via local convert_hf_to_gguf.py
#   step 3: quantize f16 -> q4_k_m via llama-quantize
#
# Usage:
#   python training/export_qwen_gguf.py merge      # step 1 only
#   python training/export_qwen_gguf.py convert    # steps 2+3
#   python training/export_qwen_gguf.py all        # everything
# ═══════════════════════════════════════════════════════════════════════════════
import os
import subprocess
import sys
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CHECKPOINT  = ROOT / "outputs" / "qwen9b-discord" / "checkpoint-4011"
MERGED_DIR  = ROOT / "outputs" / "qwen9b-discord-merged"
GGUF_DIR    = ROOT / "outputs" / "qwen9b-discord-gguf"
F16_GGUF    = GGUF_DIR / "qwen9b-discord-f16.gguf"
Q4_GGUF     = GGUF_DIR / "qwen9b-discord-q4_k_m.gguf"

LLAMA_CPP   = Path.home() / ".unsloth" / "llama.cpp"
CONVERT_PY  = LLAMA_CPP / "convert_hf_to_gguf.py"
QUANTIZE    = LLAMA_CPP / "llama-quantize"


def merge():
    """Load base + LoRA adapter, merge to a full 16-bit HF checkpoint."""
    import torch
    from unsloth import FastLanguageModel

    print(f"Loading adapter checkpoint: {CHECKPOINT}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(CHECKPOINT),
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print(f"Merging to 16-bit -> {MERGED_DIR}")
    model.save_pretrained_merged(str(MERGED_DIR), tokenizer, save_method="merged_16bit")
    del model
    torch.cuda.empty_cache()
    print("Merge done.")


def convert():
    """HF merged model -> GGUF f16 -> quantized q4_k_m, using the local llama.cpp
    convert script directly (no unsloth temp-patching)."""
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2: HF -> GGUF f16. Run with the local gguf-py on PYTHONPATH so the
    # script resolves its bundled gguf package, not a stale installed one.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(LLAMA_CPP / "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable, str(CONVERT_PY),
        str(MERGED_DIR),
        "--outfile", str(F16_GGUF),
        "--outtype", "f16",
    ]
    print("Converting HF -> GGUF f16:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"f16 GGUF written: {F16_GGUF}")

    # Step 3: quantize f16 -> q4_k_m
    cmd = [str(QUANTIZE), str(F16_GGUF), str(Q4_GGUF), "q4_k_m"]
    print("Quantizing -> q4_k_m:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"q4_k_m GGUF written: {Q4_GGUF}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("merge", "all"):
        merge()
    if stage in ("convert", "all"):
        convert()
