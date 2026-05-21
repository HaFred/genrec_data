#!/usr/bin/env python3
"""Stage 3: Extract and enrich CoT reasoning samples from OpenOneRec-RecIF.

Filters rows where reco_cot is non-null (3,646 samples), parses the CoT text,
enriches with price context from user profiles.

Output: cot_samples.parquet
"""

import argparse
import json
import os

import pandas as pd
from tqdm import tqdm


def parse_item_id(pid_str: str) -> int | None:
    """Parse '<itemid|795666>' → 795666, '<photoid|123>' → 123."""
    if not isinstance(pid_str, str):
        return None
    for prefix in ["itemid|", "photoid|"]:
        if prefix in pid_str:
            try:
                return int(pid_str.split(prefix)[1].rstrip(">"))
            except (ValueError, IndexError):
                return None
    return None


def get_user_price_tier(profile: dict, price_stats: dict) -> str:
    """Determine user's typical price tier from shopping cart history."""
    cart_items = profile.get("最近添加的购物车", [])
    if not cart_items:
        return "未知"

    prices = []
    for item in cart_items:
        p = item.get("商品单价")
        if isinstance(p, (int, float)) and p > 0:
            prices.append(float(p))

    if not prices:
        return "未知"

    avg_price = sum(prices) / len(prices)
    low = price_stats["tier_thresholds"]["low"]
    med = price_stats["tier_thresholds"]["med"]

    if avg_price <= low:
        return "¥低档"
    elif avg_price <= med:
        return "¥中档"
    else:
        return "¥高档"


def extract_purchase_categories(profile: dict) -> list[str]:
    """Extract categories user purchased from."""
    cats = []
    for item in profile.get("最近添加的购物车", []):
        cat = item.get("类目", "")
        if cat:
            # Take top-level category
            top = cat.split("-")[0] if "-" in cat else cat
            if top not in cats:
                cats.append(top)
    return cats[:5]


def get_video_interest_categories(profile: dict) -> list[str]:
    """Extract video interest categories from user profile."""
    interests = profile.get("关注博主类型", [])
    if not interests:
        return []
    return [item.get("博主类型", "") for item in interests[:5] if item.get("博主类型")]


def get_recent_purchases(profile: dict) -> str:
    """Get recent purchase summary."""
    recent = profile.get("最近对平台上这些挂车商品感兴趣并产生了消费行为", "")
    if not recent:
        cart_items = profile.get("最近添加的购物车", [])
        if cart_items:
            names = [item.get("商品名称", "") for item in cart_items[:3] if item.get("商品名称")]
            return "、".join(names)
        return "暂无购买记录"
    return str(recent)[:200]


def main():
    parser = argparse.ArgumentParser(description="Extract CoT reasoning samples")
    parser.add_argument("--input", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet")
    parser.add_argument("--price_stats", type=str, required=True,
                        help="Path to price_stats.json from 01_build_price_lookup.py")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(args.price_stats) as f:
        price_stats = json.load(f)

    print(f"Loading source data from {args.input} ...")
    df = pd.read_parquet(args.input)

    # Filter rows with CoT data
    cot_mask = df["reco_cot"].notna()
    df_cot = df[cot_mask].copy()
    print(f"  {len(df_cot):,} rows with CoT data (out of {len(df):,} total)")

    records = []
    for _, row in tqdm(df_cot.iterrows(), total=len(df_cot), desc="Processing CoT samples"):
        uid = int(row["uid"])
        cot_text = str(row["reco_cot"])
        target_caption = str(row.get("reco_target_caption", ""))
        gsu_captions = row.get("reco_gsu_caption")

        # Parse user profile for enrichment
        profile_raw = row.get("inter_user_profile_with_pid")
        profile = {}
        if isinstance(profile_raw, str):
            try:
                profile = json.loads(profile_raw)
            except (json.JSONDecodeError, TypeError):
                pass

        price_tier = get_user_price_tier(profile, price_stats)
        purchase_cats = extract_purchase_categories(profile)
        video_interests = get_video_interest_categories(profile)
        recent_purchases = get_recent_purchases(profile)

        # Build enriched CoT prompt context
        user_context_parts = []

        if video_interests:
            user_context_parts.append(f"用户感兴趣的视频类型：{'、'.join(video_interests)}")

        if purchase_cats:
            user_context_parts.append(f"用户购买过的商品类目：{'、'.join(purchase_cats)}")

        user_context_parts.append(f"用户消费水平：{price_tier}")

        if recent_purchases and recent_purchases != "暂无购买记录":
            user_context_parts.append(f"最近购买：{recent_purchases}")

        user_context = "\n".join(user_context_parts)

        # Build messages
        system_prompt = "你是一个推荐理由解释助手，能够分析用户行为并给出有深度的推荐原因。"
        user_prompt = f"{user_context}\n\n请分析用户的兴趣偏好和消费习惯，给出推荐该内容的详细理由。"

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": f"<think>\n{cot_text}\n</think>\n{target_caption}"}]},
        ]

        records.append({
            "source": "RecIF_CotReason",
            "uuid": str(uid),
            "messages": json.dumps(messages, ensure_ascii=False),
            "metadata": json.dumps({
                "uid": uid,
                "task": "cot_reason",
                "gt_sid": "",  # No SID in CoT data, reward is think_format based
                "price_tier": price_tier,
            }, ensure_ascii=False),
        })

    df_out = pd.DataFrame(records)
    out_path = os.path.join(output_dir, "cot_samples.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_out):,} CoT samples to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
