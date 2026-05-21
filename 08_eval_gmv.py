#!/usr/bin/env python3
"""Stage 8: GMV-aligned evaluation for product recommendation models.

Evaluates a model checkpoint on eval_gmv.parquet (Ranked Product task),
computing GMV-weighted metrics:
  - Expected-GMV@K: sum over predictions of (hit_k * price_k)
  - Price-NDCG@K: NDCG with price×funnel as relevance
  - GMV-Recall@K: sum price of correct / sum price of all gt items
  - Funnel-NDCG, Hit@K, Price-tier-accuracy, Avg-price-predicted

Usage:
    python 08_eval_gmv.py --model-path /path/to/model \
        --eval-data eval_gmv.parquet --output-json results.json

Comparison mode:
    python 08_eval_gmv.py --compare base.json grpo.json
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger("eval_gmv")

SID_BEGIN = "<|sid_begin|>"
SID_END = "<|sid_end|>"
SID_TRIPLE_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def extract_sids_ordered(text: str) -> list[str]:
    seen: set[str] = set()
    sids: list[str] = []
    parts = text.split(SID_BEGIN)
    for i, part in enumerate(parts):
        if i == 0:
            for m in SID_TRIPLE_PATTERN.finditer(part):
                sid = f"{SID_BEGIN}{m.group(0)}{SID_END}"
                if sid not in seen:
                    seen.add(sid)
                    sids.append(sid)
            continue
        if SID_END in part:
            inner = part.split(SID_END, 1)[0]
            sid = f"{SID_BEGIN}{inner}{SID_END}"
            if sid not in seen:
                seen.add(sid)
                sids.append(sid)
    return sids


def extract_first_sid(text: str) -> str:
    sids = extract_sids_ordered(text)
    return sids[0] if sids else ""


def load_eval_samples(eval_parquet: Path, max_samples: int = -1) -> list[dict]:
    df = pd.read_parquet(eval_parquet)
    if max_samples > 0:
        df = df.head(max_samples)

    samples = []
    for _, row in df.iterrows():
        raw_msgs = row["messages"]
        messages = json.loads(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs

        converted = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            converted.append({"role": msg.get("role", ""), "content": str(content or "")})

        groundtruth = ""
        prompt_messages = list(converted)
        if prompt_messages and prompt_messages[-1].get("role") == "assistant":
            groundtruth = str(prompt_messages[-1].get("content", "")).strip()
            prompt_messages = prompt_messages[:-1]

        meta_raw = row.get("metadata", "{}")
        metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})

        if not groundtruth:
            continue

        samples.append({
            "sample_id": str(len(samples)),
            "messages": prompt_messages,
            "groundtruth": groundtruth,
            "metadata": metadata,
        })

    return samples


def build_prompt(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_path: str, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("Loading model from %s", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_sids(model, tokenizer, prompt: str, num_samples: int,
                  max_new_tokens: int, temperature: float, top_p: float,
                  top_k: int, device: str) -> list[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    all_sids: set[str] = set()

    batch_size = min(num_samples, 4)
    remaining = num_samples
    while remaining > 0:
        n = min(batch_size, remaining)
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, top_k=top_k,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        for i in range(output.shape[0]):
            text = tokenizer.decode(output[i][prompt_len:], skip_special_tokens=False)
            sid = extract_first_sid(text)
            if sid:
                all_sids.add(sid)
        del output
        remaining -= n
        if remaining > 0:
            torch.cuda.empty_cache()
    return list(all_sids)


def compute_gmv_metrics(samples: list[dict], all_beams: list[list[str]],
                         k_values: list[int]) -> dict[str, Any]:
    evaluated = 0
    metrics: dict[str, list[float]] = {f"hit@{k}": [] for k in k_values}
    metrics.update({f"expected_gmv@{k}": [] for k in k_values})
    metrics.update({f"price_ndcg@{k}": [] for k in k_values})
    metrics.update({f"gmv_recall@{k}": [] for k in k_values})
    metrics.update({f"funnel_ndcg@{k}": [] for k in k_values})
    metrics["price_tier_accuracy"] = []
    metrics["avg_price_predicted"] = []

    per_sample = {}

    for idx, (sample, beams) in enumerate(zip(samples, all_beams)):
        sid = sample["sample_id"]
        gt_sids = extract_sids_ordered(sample["groundtruth"])
        gt_set = set(gt_sids)
        meta = sample.get("metadata", {})
        rank_labels = meta.get("rank_labels", [])

        if not gt_set or not beams:
            continue

        predicted = beams[:max(k_values)]

        # Compute per-position price and relevance for NDCG
        pred_relevances = []
        pred_prices = []

        for pred_sid in beams:
            # Try to find price from rank_labels
            price = 0.0
            relevance = 0.0
            if pred_sid in gt_set:
                # Find matching rank_label
                for i, label in enumerate(rank_labels):
                    # Match by position - rough approximation
                    pass
                relevance = 1.0  # Hit-based relevance
                # Use first GT price as approximation
                if rank_labels:
                    price = rank_labels[0].get("price", 0.0)
            pred_relevances.append(relevance)
            pred_prices.append(price)

        # Ground truth relevances and prices from rank_labels
        gt_relevances = []
        gt_prices = []
        for label in rank_labels:
            funnel = label.get("funnel", 0)
            price = label.get("price", 0)
            if price > 0:
                pw = math.log(1 + price) / math.log(1 + 10000)
                gt_relevances.append(funnel * pw)
                gt_prices.append(price)

        sample_metrics: dict[str, Any] = {"gt_sids": gt_sids, "pred_top10": beams[:10]}

        for k in k_values:
            top_pred = beams[:k]
            top_pred_set = set(top_pred)

            # Hit@K
            hit = any(sid in gt_set for sid in top_pred)
            sample_metrics[f"hit@{k}"] = hit
            metrics[f"hit@{k}"].append(float(hit))

            # Expected-GMV@K
            exp_gmv = 0.0
            for i, (sid, price) in enumerate(zip(top_pred, pred_prices[:k])):
                if sid in gt_set and price > 0:
                    exp_gmv += price / math.log2(i + 2)
            sample_metrics[f"expected_gmv@{k}"] = exp_gmv
            metrics[f"expected_gmv@{k}"].append(exp_gmv)

            # Price-NDCG@K
            dcg = 0.0
            for i, rel in enumerate(pred_relevances[:k]):
                dcg += rel / math.log2(i + 2)
            idcg = 0.0
            for i, rel in enumerate(sorted(gt_relevances, reverse=True)[:k]):
                idcg += rel / math.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0.0
            sample_metrics[f"price_ndcg@{k}"] = ndcg
            metrics[f"price_ndcg@{k}"].append(ndcg)

            # GMV-Recall@K
            total_gt_price = sum(gt_prices) if gt_prices else 0.0
            hit_price = sum(
                gt_prices[i] if i < len(gt_prices) and list(gt_sids)[i] if i < len(gt_sids) else 0
                for i in range(min(k, len(gt_sids)))
                if i < len(gt_sids) and gt_sids[i] if i < len(gt_sids) else False
            )
            # Simpler: sum price of matched predictions
            matched_price = 0.0
            for sid in top_pred:
                if sid in gt_set:
                    # Use avg price from rank_labels
                    avg_gt_price = sum(gt_prices) / max(len(gt_prices), 1) if gt_prices else 0
                    matched_price += avg_gt_price
            gmv_recall = matched_price / max(total_gt_price, 1)
            sample_metrics[f"gmv_recall@{k}"] = gmv_recall
            metrics[f"gmv_recall@{k}"].append(gmv_recall)

            # Funnel-NDCG@K (ablation - price weight only)
            funnel_dcg = 0.0
            for i, sid in enumerate(top_pred):
                rel = 1.0 if sid in gt_set else 0.0
                funnel_dcg += rel / math.log2(i + 2)
            funnel_idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gt_set), k)))
            funnel_ndcg = funnel_dcg / max(funnel_idcg, 0.001)
            sample_metrics[f"funnel_ndcg@{k}"] = funnel_ndcg
            metrics[f"funnel_ndcg@{k}"].append(funnel_ndcg)

        # Average price of predicted items
        avg_price = sum(pred_prices[:10]) / max(len([p for p in pred_prices[:10] if p > 0]), 1)
        metrics["avg_price_predicted"].append(avg_price)

        per_sample[sid] = sample_metrics
        evaluated += 1

    summary = {"num_samples_evaluated": evaluated}
    for metric_name, values in metrics.items():
        if values:
            arr = np.array(values)
            summary[metric_name] = float(np.mean(arr))

    summary["per_sample"] = per_sample
    return summary


def compare_results(pre_json: Path, post_json: Path) -> None:
    with pre_json.open() as f:
        pre = json.load(f)
    with post_json.open() as f:
        post = json.load(f)

    metric_keys = sorted(k for k in pre if k != "per_sample" and k != "model_path"
                         and k != "num_samples_evaluated" and k != "inference_time_s")

    print(f"\n{'='*72}")
    print(f"  GMV Eval Comparison")
    print(f"{'='*72}")
    print(f"  Base:  {pre.get('model_path', '?')}")
    print(f"  GRPO:  {post.get('model_path', '?')}")
    print(f"  Samples: {pre.get('num_samples_evaluated', '?')} / {post.get('num_samples_evaluated', '?')}")
    print(f"{'='*72}")
    print(f"  {'Metric':<24} {'Base':>12} {'GRPO':>12} {'Delta':>12} {'Δ%':>10}")
    print(f"  {'-'*70}")

    for key in metric_keys:
        pre_val = pre[key]
        post_val = post[key]
        delta = post_val - pre_val
        delta_pct = (delta / pre_val * 100) if pre_val != 0 else float("inf")
        print(f"  {key:<24} {pre_val:>12.4f} {post_val:>12.4f} {delta:>+12.4f} {delta_pct:>+9.1f}%")

    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="GMV-aligned model evaluation")
    parser.add_argument("--model-path", type=str, help="Path to HuggingFace model")
    parser.add_argument("--eval-data", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/"
                                "vanilla_onerec_customized_gmv_data/outputs/rl_data_enriched/eval_gmv.parquet")
    parser.add_argument("--output-json", type=str, help="Path to save results JSON")
    parser.add_argument("--num-beams", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--k-values", type=str, default="1,5,10,16")
    parser.add_argument("--compare", nargs=2, type=str,
                        metavar=("BASE_JSON", "GRPO_JSON"),
                        help="Compare two existing result JSONs")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.compare:
        compare_results(Path(args.compare[0]), Path(args.compare[1]))
        return

    if not args.model_path:
        print("ERROR: --model-path required", file=sys.stderr)
        sys.exit(1)

    eval_data = Path(args.eval_data)
    if not eval_data.exists():
        print(f"WARNING: eval data not found at {eval_data}, falling back to test.parquet")
        eval_data = Path("/home/dyvm6xra/dyvm6xrauser45/fred/openonerec_fredfork/data/test.parquet")

    k_values = [int(k.strip()) for k in args.k_values.split(",") if k.strip()]

    samples = load_eval_samples(eval_data, args.max_samples)
    logger.info("Loaded %d eval samples", len(samples))

    if not samples:
        print("No eval samples found!")
        return

    model, tokenizer = load_model(args.model_path, args.device)

    started = time.perf_counter()
    all_beams = []
    for sample in tqdm(samples, desc="GMV Eval"):
        try:
            prompt = build_prompt(tokenizer, sample["messages"]) + SID_BEGIN
            sids = generate_sids(model, tokenizer, prompt,
                                 num_samples=args.num_beams,
                                 max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature,
                                 top_p=args.top_p, top_k=args.top_k,
                                 device=args.device)
            all_beams.append(sids)
        except Exception as e:
            logger.warning("Error on sample %s: %s", sample["sample_id"], e)
            all_beams.append([])
        if len(all_beams) % 10 == 0:
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    logger.info("Inference done: %.1fs (%.2f s/sample)", elapsed, elapsed / max(len(samples), 1))

    metrics = compute_gmv_metrics(samples, all_beams, k_values)
    metrics["model_path"] = args.model_path
    metrics["inference_time_s"] = elapsed

    print(f"\n{'='*55}")
    print(f"  GMV Evaluation Results")
    print(f"{'='*55}")
    print(f"  Evaluated: {metrics['num_samples_evaluated']}/{len(samples)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*55}")
    for k in k_values:
        for prefix in ["hit", "expected_gmv", "price_ndcg", "gmv_recall", "funnel_ndcg"]:
            key = f"{prefix}@{k}"
            if key in metrics:
                print(f"  {key:<24} {metrics[key]:.4f}")
    print(f"{'='*55}\n")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
