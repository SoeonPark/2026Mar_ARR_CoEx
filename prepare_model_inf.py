import os
import gc
import argparse
import random
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil


def get_module_attr(model, attr_path: str):
    """
    Safely resolve nested attributes like:
    "model.layers.0.self_attn.q_proj"
    """
    obj = model
    for part in attr_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj




@torch.no_grad()
def forward_logits(model, tokenizer, text: str, device: str = "cpu", max_len: int = 256):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    # logits: [B, T, V]
    return out.logits.detach().float().cpu()


@torch.no_grad()
def weight_stats_qproj(model, qproj_path: str, rows: int = 4, cols: int = 8):
    """
    Grab a small slice from q_proj.weight via module attribute access (not state_dict keys).
    """
    qproj = get_module_attr(model, qproj_path)
    w = qproj.weight.detach().float().cpu()
    r = min(rows, w.shape[0])
    c = min(cols, w.shape[1]) if w.ndim == 2 else min(cols, w.numel())
    sl = w[:r, :c] if w.ndim == 2 else w.flatten()[:c]
    return {
        "shape": tuple(w.shape),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "slice": sl.tolist(),
    }


@torch.no_grad()
def diff_tensor(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    return float(d.mean()), float(d.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=".")
    parser.add_argument("--step", type=int, default=500)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--adapter_name", type=str, default=None)  # usually "default"
    parser.add_argument("--test_text", type=str, default="Solve: 17 + 25 = ? Answer:")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--save_merged", action="store_true")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--qproj_path", type=str, default="auto",
                    help="Phi-4-mini uses qkv_proj (combined). For Qwen/DeepSeek use q_proj.")
    args = parser.parse_args()

    if args.adapter_path is None:
        args.adapter_path = f"{args.base_dir}/trainer_output/{args.experiment_name}/checkpoint-{args.step}"
    if args.out_dir is None:
        args.out_dir  = f"{args.base_dir}/merged_output/{args.experiment_name}/checkpoint-{args.step}"
        
    print(f"[INFO] adapter_path: {args.adapter_path}")
        
    if args.base_model_path is None:
        adapter_config = os.path.join(args.adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config):
            raise ValueError("base_model_path not provided and adapter_config.json not found to infer it.")
        import json
        with open(adapter_config, "r") as f:
            acfg = json.load(f)
        args.base_model_path = acfg["base_model_name_or_path"]
        
    print(f"[INFO] base_model_path: {args.base_model_path}")

    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(f"adapter_path not found: {args.adapter_path}")
    if not os.path.exists(os.path.join(args.adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"adapter_config.json not found under: {args.adapter_path}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    print(f"[INFO] Using dtype: {args.dtype}")
    torch_dtype = dtype_map[args.dtype]

    # tokenizer from base is safest
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=args.trust_remote_code)

    # (A) reference base: never passed into PEFT (prevents aliasing)
    base_ref = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if args.qproj_path == "auto":
        for name, _ in base_ref.named_modules():
            if "layers.0" in name and "self_attn" in name:
                if name.endswith("q_proj") or name.endswith("qkv_proj"):
                    args.qproj_path = name
                    break
        print(f"[INFO] auto-detected qproj_path: {args.qproj_path}")

    print("==== 1) Base weight slice (attribute access) ====")
    base_w = weight_stats_qproj(base_ref, args.qproj_path)

    # (B) base for PEFT: separate instance
    base_for_peft = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    print("==== 1) Base weight slice (attribute access) ====")
    base_w = weight_stats_qproj(base_ref, args.qproj_path)
    print("[BASE] q_proj.weight", base_w["shape"], "mean", base_w["mean"], "std", base_w["std"])
    # print small slice
    print("[BASE] slice:", base_w["slice"])

    print("\n==== 2) Load PEFT adapter and check LoRA presence ====")
    peft_model = PeftModel.from_pretrained(base_for_peft, args.adapter_path, is_trainable=False)

    # adapter info
    try:
        print("[PEFT] adapters:", list(peft_model.peft_config.keys()))
    except Exception:
        print("[PEFT] adapters: (unknown)")
    if hasattr(peft_model, "active_adapters"):
        print("[PEFT] active_adapters:", peft_model.active_adapters)
    if hasattr(peft_model, "active_adapter"):
        print("[PEFT] active_adapter:", peft_model.active_adapter)

    if args.adapter_name is not None and hasattr(peft_model, "set_adapter"):
        try:
            peft_model.set_adapter(args.adapter_name)
            print(f"[PEFT] set_adapter('{args.adapter_name}') OK")
        except Exception as e:
            print(f"[PEFT] set_adapter('{args.adapter_name}') FAILED:", e)

    # quick LoRA param check
    lora_cnt = 0
    lora_norm_sum = 0.0
    for n, p in peft_model.named_parameters():
        if ("lora_A" in n) or ("lora_B" in n):
            lora_cnt += 1
            lora_norm_sum += float(p.detach().float().norm().cpu())
    print(f"[LoRA] param_count={lora_cnt}, norm_sum={lora_norm_sum:.6f}")
    if lora_cnt == 0:
        print("[LoRA] WARNING: no LoRA params found. Adapter may not be LoRA or load failed.")

    print("\n==== 3) Does PEFT change model behavior? (logits diff) ====")
    # logits on the same prompt
    logits_base = forward_logits(base_ref, tokenizer, args.test_text, device=args.device, max_len=args.max_len)
    logits_peft = forward_logits(peft_model, tokenizer, args.test_text, device=args.device, max_len=args.max_len)

    mean_bp, max_bp = diff_tensor(logits_base, logits_peft)
    print(f"[LOGITS base->peft] mean_abs={mean_bp:.8e}, max_abs={max_bp:.8e}")
    if mean_bp == 0.0 and max_bp == 0.0:
        print("[WARN] logits identical. Either LoRA effect is extremely tiny, adapter not applied, or prompt too short.")
        print("       Try different --test_text (longer / more diverse), and ensure dtype=float16/float32.")

    print("\n==== 4) Merge and verify equivalence (peft vs merged logits) ====")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    logits_merged = forward_logits(merged_model, tokenizer, args.test_text, device=args.device, max_len=args.max_len)
    mean_pm, max_pm = diff_tensor(logits_peft, logits_merged)
    print(f"[LOGITS peft->merged] mean_abs={mean_pm:.8e}, max_abs={max_pm:.8e}")
    if mean_pm > 1e-6:
        print("[WARN] peft vs merged logits differ more than expected. Check dtype/device and whether merge succeeded.")

    print("\n==== 5) Optional: weight-level diff via attribute access ====")
    # Compare q_proj.weight between base_ref and merged (should differ if LoRA had effect)
    merged_w = weight_stats_qproj(merged_model, args.qproj_path)
    print("[MERGED] q_proj.weight", merged_w["shape"], "mean", merged_w["mean"], "std", merged_w["std"])
    # compute mean/max abs diff on weight tensors (full tensor) for stronger evidence
    w_base_full = get_module_attr(base_ref, args.qproj_path).weight.detach().float().cpu()
    w_merged_full = get_module_attr(merged_model, args.qproj_path).weight.detach().float().cpu()
    mean_w, max_w = diff_tensor(w_base_full, w_merged_full)
    print(f"[WEIGHT base->merged] mean_abs={mean_w:.8e}, max_abs={max_w:.8e}")

    print("\n==== 6) Save merged (optional) ====")
    # if args.save_merged:
    if args.out_dir is None:
        raise ValueError("--save_merged requires --out_dir")
    os.makedirs(args.out_dir, exist_ok=True)
    merged_model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(args.out_dir)
    # preserve chat_template.jinja if present
    src = os.path.join(args.adapter_path, "chat_template.jinja")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(args.out_dir, "chat_template.jinja"))
    # trust_remote_code 모델용 .py 파일 복사 (Phi-4-mini 등)
    # base_model_path가 HuggingFace 모델 ID인 경우 캐시 경로를 찾아야 함
    import glob
    base_model_local = args.base_model_path
    if not os.path.isdir(args.base_model_path):
        try:
            from huggingface_hub import snapshot_download
            base_model_local = snapshot_download(args.base_model_path, local_files_only=True)
        except Exception as e:
            print(f"[WARN] Could not resolve HuggingFace cache path for .py files: {e}")
            base_model_local = None
    if base_model_local:
        for py_file in glob.glob(os.path.join(base_model_local, "*.py")):
            shutil.copy(py_file, args.out_dir)
            print(f"[SAVE] copied {os.path.basename(py_file)} -> {args.out_dir}")
    print("[SAVE] merged saved to:", args.out_dir)

    # cleanup
    del merged_model, peft_model, base_for_peft, base_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n[DONE]")


if __name__ == "__main__":
    main()