"""Merge a LoRA adapter saved by the verl actor into the base model -> a plain bf16 HF directory.

verl (lora_rank > 0) writes ``global_step_N/actor/lora_adapter/{adapter_model.safetensors, adapter_config.json}``.
Our evaluators load plain HF directories, so:

  PYTHONPATH=.:training/verl python -m training.sigma_rl.merge_lora \
      --base /mnt/data/peilin/HF_MODEL/Qwen3-4B --adapter outputs/rl/<EXP>/global_step_40/actor/lora_adapter \
      --out outputs/rl/<EXP>/hf/global_step_40

The trainer calls ``merge_lora_adapter`` itself at every checkpoint (trainer.keep_hf_checkpoints), so the
command above is only needed for checkpoints saved by other means.
"""
from __future__ import annotations

import argparse
import os
import shutil


def merge_lora_adapter(base: str, adapter: str, out: str, dtype: str = "bfloat16") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch_dtype, low_cpu_mem_usage=True, local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model = model.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(out)
    kept = os.path.join(out, "lora_adapter")
    shutil.copytree(adapter, kept, dirs_exist_ok=True)   # the adapter itself stays next to the merged weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()
    merge_lora_adapter(args.base, args.adapter, args.out, args.dtype)
    print(f"[merge-lora] merged {args.adapter} into {args.base} -> {args.out}")


if __name__ == "__main__":
    main()
