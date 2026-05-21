#!/usr/bin/env python3
"""Stage 5: Build Purchase Prediction task (~50K samples).

Binary classification with price tier: given user history + candidate product + price,
predict whether user will purchase/buy this product.

Positive samples: target_goods_pid (products user purchased)
Negative samples: hist_goods_pid NOT in target (products user browsed but didn't buy)
"""

import argparse
import json
import math
import os
import random
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'

SYSTEM_PROMPTS = [
    "你是一个用户消费行为预测助手，能够根据用户的观看和购物历史，判断用户是否会购买某个商品。",
    "你是一个消费预测专家，擅长分析用户的购买意向，判断用户对某商品的购买可能性。",
    "你是一个用户行为分析系统，基于用户的浏览和购物记录，判断用户是否会购买指定商品。",
]

VIDEO_PROMPTS = ["用户观看过的视频：", "用户浏览过的视频内容："]
PRODUCT_PROMPTS = ["用户最近购买/加购的商品：", "用户历史购物记录："]
PRICE_CONTEXT = ["用户消费水平：{tier}", "用户价格偏好：{tier}"]

POS_TASK_PROMPTS = [
    "候选商品：{sid}（¥{price:.0f}）\n请判断该用户是否会购买此商品，并说明理由。",
]
NEG_TASK_PROMPTS = [
    "候选商品：{sid}（¥{price:.0f}）\n该用户浏览过但未购买此商品，请分析可能的原因。",
]

POS_ANSWER = "<think>该商品与用户历史购买记录和消费水平匹配，用户有较高的购买可能性。</think>\n会购买，价格段：{tier}"
NEG_ANSWER = "<think>该商品可能不符合用户的消费偏好或价格预期，用户浏览后未购买。</think>\n不会购买"


def pids_to_sids(pids, pid2sid: dict) -> str:
    if pids is None or (isinstance(pids, float) and pd.isna(pids)):
        return ""
    sids = []
    for pid in pids:
        pid_int = int(pid)
        if pid_int in pid2sid:
            code = pid2sid[pid_int]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            sids.append(sid)
    return ''.join(sids)


def get_price_tier(price: float, thresholds: dict) -> str:
    if price <= thresholds["low"]:
        return "¥低档"
    elif price <= thresholds["med"]:
        return "¥中档"
    else:
        return "¥高档"


def main():
    parser = argparse.ArgumentParser(description="Build purchase prediction task")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--video_pid2sid", type=str, required=True)
    parser.add_argument("--product_pid2sid", type=str, required=True)
    parser.add_argument("--item_prices", type=str, required=True)
    parser.add_argument("--cat_prices", type=str, required=True)
    parser.add_argument("--price_stats", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings
    print("Loading mappings...")
    df_v = pd.read_parquet(args.video_pid2sid)
    video_pid2sid = dict(zip(df_v["pid"], df_v["sid"]))
    df_p = pd.read_parquet(args.product_pid2sid)
    product_pid2sid = dict(zip(df_p["pid"], df_p["sid"]))

    df_prices = pd.read_parquet(args.item_prices)
    price_map = {int(r["pid"]): float(r["median_price"]) for _, r in df_prices.iterrows()}
    cat_map = {int(r["pid"]): str(r.get("category", "")) for _, r in df_prices.iterrows()}

    df_cats = pd.read_parquet(args.cat_prices)
    cat_price_map = {str(r["category"]): float(r["median_price"]) for _, r in df_cats.iterrows()}

    with open(args.price_stats) as f:
        stats = json.load(f)
    global_median = stats["global_median"]
    thresholds = stats["tier_thresholds"]

    print(f"  Prices: {len(price_map):,} items  Categories: {len(cat_price_map):,}")

    # Load source
    print(f"\nLoading source data from {args.input} ...")
    df = pd.read_parquet(args.input)

    results = []
    pos_count = 0
    neg_count = 0
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building purchase pred data"):
        if row["split"] != 0:
            continue

        uid = int(row["uid"])
        target_pids = row["target_goods_pid"]
        hist_goods = row["hist_goods_pid"]
        hist_videos = row.get("hist_longview_video_list")

        # Collect positive and negative candidates
        target_set = set(int(p) for p in target_pids) if target_pids is not None and hasattr(target_pids, "__iter__") else set()
        hist_list = [int(p) for p in hist_goods] if hist_goods is not None and hasattr(hist_goods, "__iter__") else []

        # Determine user price tier
        user_prices = []
        for pid in target_set:
            p = price_map.get(pid)
            if p:
                user_prices.append(p)
        if not user_prices:
            for pid in hist_list[:20]:
                p = price_map.get(pid)
                if p:
                    user_prices.append(p)
        avg_user_price = sum(user_prices) / len(user_prices) if user_prices else global_median
        user_tier = get_price_tier(avg_user_price, thresholds)

        # Build prompt prefix (shared across samples for this user)
        user_parts = []
        if hist_videos is not None and hasattr(hist_videos, "__iter__"):
            video_sids = pids_to_sids(hist_videos[-100:], video_pid2sid)
            if video_sids:
                user_parts.append(f"{random.choice(VIDEO_PROMPTS)}{video_sids}")
        if hist_list:
            prod_sids = pids_to_sids(hist_list[-100:], product_pid2sid)
            if prod_sids:
                user_parts.append(f"{random.choice(PRODUCT_PROMPTS)}{prod_sids}")
        user_parts.append(random.choice(PRICE_CONTEXT).format(tier=user_tier))

        if not user_parts:
            continue

        user_prefix = "\n".join(user_parts)

        # Positive samples (target products)
        neg_candidates = [p for p in hist_list if p not in target_set and p in product_pid2sid]
        random.shuffle(neg_candidates)
        neg_candidates = neg_candidates[:5]  # Max 5 negatives per user

        for pid in target_set:
            if pid not in product_pid2sid:
                continue
            if pos_count >= 30000:  # Cap positives
                break

            code = product_pid2sid[pid]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            price = price_map.get(pid, cat_price_map.get(cat_map.get(pid, ""), global_median))
            tier = get_price_tier(price, thresholds)

            task = random.choice(POS_TASK_PROMPTS).format(sid=sid, price=price)
            answer = POS_ANSWER.format(tier=tier)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": random.choice(SYSTEM_PROMPTS)}]},
                {"role": "user", "content": [{"type": "text", "text": user_prefix + "\n" + task}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            results.append({
                "source": "RecIF_PurchasePred",
                "uuid": str(uuid.uuid4()),
                "messages": json.dumps(messages, ensure_ascii=False),
                "metadata": json.dumps({
                    "uid": uid, "task": "purchase_pred",
                    "label": 1, "price": price, "price_tier": tier,
                }, ensure_ascii=False),
            })
            pos_count += 1

        # Negative samples
        for pid in neg_candidates:
            if pid not in product_pid2sid or pid in target_set:
                continue
            if neg_count >= 25000:
                break

            code = product_pid2sid[pid]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            price = price_map.get(pid, cat_price_map.get(cat_map.get(pid, ""), global_median))

            task = random.choice(NEG_TASK_PROMPTS).format(sid=sid, price=price)
            answer = NEG_ANSWER

            messages = [
                {"role": "system", "content": [{"type": "text", "text": random.choice(SYSTEM_PROMPTS)}]},
                {"role": "user", "content": [{"type": "text", "text": user_prefix + "\n" + task}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]

            tier = get_price_tier(price, thresholds)
            results.append({
                "source": "RecIF_PurchasePred",
                "uuid": str(uuid.uuid4()),
                "messages": json.dumps(messages, ensure_ascii=False),
                "metadata": json.dumps({
                    "uid": uid, "task": "purchase_pred",
                    "label": 0, "price": price, "price_tier": tier,
                }, ensure_ascii=False),
            })
            neg_count += 1

    print(f"\nResults: {len(results):,} samples")
    print(f"  Positive: {pos_count:,}  Negative: {neg_count:,}  Skipped: {skipped:,}")

    df_out = pd.DataFrame(results)
    out_path = output_dir / "purchase_pred.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
