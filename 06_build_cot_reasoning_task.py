#!/usr/bin/env python3
"""Stage 6: Build CoT Reasoning task (~3K samples).

Enriches existing reco_cot samples from the source with price context.
Uses the CoT samples extracted by 03_extract_cot_samples.py and further
augments them into the standard messages format suitable for GRPO training.
"""

import argparse
import json
import os
import random
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

SYSTEM_PROMPTS = [
    "你是一个推荐理由解释助手，能够分析用户行为并给出有深度的推荐原因。",
    "你是一个推荐系统分析师，擅长解释推荐结果背后的用户行为逻辑和商业价值。",
    "你是一个内容推荐解释引擎，能够基于用户偏好和消费数据，生成详细的推荐理由。",
]


def get_price_tier(price: float, thresholds: dict) -> str:
    if price <= thresholds["low"]:
        return "¥低档"
    elif price <= thresholds["med"]:
        return "¥中档"
    else:
        return "¥高档"


def main():
    parser = argparse.ArgumentParser(description="Build CoT reasoning task")
    parser.add_argument("--input", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet")
    parser.add_argument("--item_prices", type=str, required=True)
    parser.add_argument("--cat_prices", type=str, required=True)
    parser.add_argument("--price_stats", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load price data
    df_prices = pd.read_parquet(args.item_prices)
    price_map = {int(r["pid"]): float(r["median_price"]) for _, r in df_prices.iterrows()}
    cat_map = {int(r["pid"]): str(r.get("category", "")) for _, r in df_prices.iterrows()}

    df_cats = pd.read_parquet(args.cat_prices)
    cat_price_map = {str(r["category"]): float(r["median_price"]) for _, r in df_cats.iterrows()}

    with open(args.price_stats) as f:
        stats = json.load(f)
    global_median = stats["global_median"]
    thresholds = stats["tier_thresholds"]

    # Load source and filter CoT rows
    print(f"Loading source data from {args.input} ...")
    df = pd.read_parquet(args.input)
    cot_mask = df["reco_cot"].notna()
    df_cot = df[cot_mask].copy()
    print(f"  {len(df_cot):,} rows with CoT data")

    results = []
    for _, row in tqdm(df_cot.iterrows(), total=len(df_cot), desc="Building CoT data"):
        uid = int(row["uid"])
        cot_text = str(row["reco_cot"])
        target_caption = str(row.get("reco_target_caption", ""))
        gsu_captions = row.get("reco_gsu_caption")

        # Enrich with price context
        profile_raw = row.get("inter_user_profile_with_pid")
        profile = {}
        if isinstance(profile_raw, str):
            try:
                profile = json.loads(profile_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        # Get user price tier
        cart_items = profile.get("最近添加的购物车", [])
        prices = []
        for item in cart_items:
            p = item.get("商品单价")
            if isinstance(p, (int, float)) and p > 0:
                prices.append(float(p))
        user_tier = "¥中档"
        if prices:
            avg = sum(prices) / len(prices)
            user_tier = get_price_tier(avg, thresholds)

        # Build user context
        context_parts = []

        # Video interests
        interests = profile.get("关注博主类型", [])
        if interests:
            cats = [item.get("博主类型", "") for item in interests[:3] if item.get("博主类型")]
            if cats:
                context_parts.append(f"用户偏好视频类型：{'、'.join(cats)}")

        # Purchase categories
        purchase_cats = []
        for item in cart_items[:5]:
            cat = item.get("类目", "")
            if cat:
                top = cat.split("-")[0] if "-" in cat else cat
                if top not in purchase_cats:
                    purchase_cats.append(top)
        if purchase_cats:
            context_parts.append(f"用户购买类目：{'、'.join(purchase_cats[:5])}")

        context_parts.append(f"用户消费水平：{user_tier}")

        # Recent purchases
        recent = profile.get("最近对平台上这些挂车商品感兴趣并产生了消费行为", "")
        if recent and str(recent).strip():
            context_parts.append(f"最近消费：{str(recent)[:100]}")

        user_context = "\n".join(context_parts)

        # GSU caption context
        gsu_text = ""
        if gsu_captions is not None and hasattr(gsu_captions, "__iter__"):
            gsu_items = []
            for cap in gsu_captions[:5]:
                if isinstance(cap, str) and cap.strip():
                    gsu_items.append(f"  - {cap[:80]}")
            if gsu_items:
                gsu_text = "候选内容描述：\n" + "\n".join(gsu_items) + "\n"

        # Task prompt
        task = f"{gsu_text}\n{user_context}\n\n请分析该用户的兴趣偏好和消费习惯，解释推荐以下内容的原因：\n{target_caption[:200]}"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": random.choice(SYSTEM_PROMPTS)}]},
            {"role": "user", "content": [{"type": "text", "text": task}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"<think>\n{cot_text}\n</think>\n{target_caption}"}]},
        ]

        results.append({
            "source": "RecIF_CotReason",
            "uuid": str(uuid.uuid4()),
            "messages": json.dumps(messages, ensure_ascii=False),
            "metadata": json.dumps({
                "uid": uid, "task": "cot_reason",
                "price_tier": user_tier,
            }, ensure_ascii=False),
        })

    df_out = pd.DataFrame(results)
    out_path = output_dir / "cot_reason.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_out):,} CoT reasoning samples to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
