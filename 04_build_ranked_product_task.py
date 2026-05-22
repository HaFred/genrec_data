#!/usr/bin/env python3
"""Stage 4: Build Enriched Product Ranking task (~100K samples).

Follows the existing Chinese prompt style from product_rec.py.
Each sample includes:
  - Video watch history (longview videos in SID format)
  - Product purchase history (goods in SID format + price tier)
  - Target: ranked list of product SIDs

Metadata carries per-rank price and funnel scores for NDCG reward.
"""

import argparse
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
VIDEO_HIST_MAX_LEN = 100
PRODUCT_HIST_MAX_LEN = 100
TARGET_MAX_LEN = 10

SYSTEM_PROMPTS = [
    "你是一个智能跨域推荐助手，能够根据用户观看的视频内容、历史购物行为和消费水平，预测用户接下来最可能购买的商品并按优先级排序。",
    "你是一个跨域推荐专家，擅长分析用户的观看习惯、购物偏好和消费能力，预测用户的商品兴趣并排序。",
    "你是一个个性化推荐系统，能够基于用户的视频观看历史、购物记录和价格偏好，推荐并按优先级排列用户可能购买的商品。",
    "你是一个用户消费行为分析助手，专注于理解用户的内容偏好、购物兴趣和消费水平，推荐相关商品并按购买可能性排序。",
]

VIDEO_WATCH_PROMPTS = [
    "用户观看过的视频：",
    "用户浏览过的视频内容：",
    "用户长时间观看的视频：",
    "用户感兴趣的视频：",
]

PRODUCT_HIST_PROMPTS = [
    "用户最近购买/加购的商品：",
    "用户浏览过的商品：",
    "用户历史购物记录：",
    "用户感兴趣的商品：",
]

PRICE_CONTEXT_PROMPTS = [
    "用户消费水平：{tier}",
    "用户价格偏好：{tier}",
    "用户购物价位：{tier}",
]

TASK_PROMPTS = [
    "请根据以上信息，按购买可能性从高到低推荐用户可能购买的商品。",
    "基于以上记录，推荐用户可能购买的商品，并按优先级排序。",
    "分析用户的行为偏好和消费能力，预测用户下一步会购买哪些商品，并按可能性排序。",
    "请推荐用户接下来最可能购买的商品，从高到低排列。",
]


def pids_to_sids(pids, pid2sid: dict) -> str:
    """Convert a list of pids to SID string."""
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
    """Classify price into tier."""
    if price <= thresholds["low"]:
        return "¥低档"
    elif price <= thresholds["med"]:
        return "¥中档"
    else:
        return "¥高档"


def get_price_weight(price: float, max_price: float) -> float:
    """Log-scaled price weight in [0, 1]."""
    import math
    if price <= 0:
        return 0.0
    return math.log(1 + price) / math.log(1 + max_price)


def main():
    parser = argparse.ArgumentParser(description="Build ranked product task")
    parser.add_argument("--input", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet")
    parser.add_argument("--video_pid2sid", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/video_ad_pid2sid.parquet")
    parser.add_argument("--product_pid2sid", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/product_pid2sid.parquet")
    parser.add_argument("--item_prices", type=str, required=True,
                        help="Path to item_price_lookup.parquet")
    parser.add_argument("--cat_prices", type=str, required=True,
                        help="Path to category_price_lookup.parquet")
    parser.add_argument("--price_stats", type=str, required=True,
                        help="Path to price_stats.json")
    parser.add_argument("--engagement", type=str, required=True,
                        help="Path to user_item_engagement.parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings
    print("Loading mappings...")
    df_vpid2sid = pd.read_parquet(args.video_pid2sid)
    video_pid2sid = dict(zip(df_vpid2sid["pid"], df_vpid2sid["sid"]))

    df_ppid2sid = pd.read_parquet(args.product_pid2sid)
    product_pid2sid = dict(zip(df_ppid2sid["pid"], df_ppid2sid["sid"]))

    df_prices = pd.read_parquet(args.item_prices)
    # Build fast lookup
    price_map: dict[int, float] = {}
    cat_map: dict[int, str] = {}
    for _, row in df_prices.iterrows():
        pid = int(row["pid"])
        price_map[pid] = float(row["median_price"])
        cat_map[pid] = str(row.get("category", ""))

    df_cat_prices = pd.read_parquet(args.cat_prices)
    cat_price_map: dict[str, float] = {}
    for _, row in df_cat_prices.iterrows():
        cat_price_map[str(row["category"])] = float(row["median_price"])

    with open(args.price_stats) as f:
        price_stats = json.load(f)
    global_median = price_stats["global_median"]
    global_max = price_stats["global_max"]
    thresholds = price_stats["tier_thresholds"]

    df_eng = pd.read_parquet(args.engagement)
    # Build uid → {pid → funnel_score}
    eng_map: dict[int, dict[int, float]] = {}
    for _, row in df_eng.iterrows():
        uid = int(row["uid"])
        pid = int(row["pid"])
        score = float(row["funnel_score"])
        if uid not in eng_map:
            eng_map[uid] = {}
        # Keep max score per (uid, pid)
        eng_map[uid][pid] = max(eng_map[uid].get(pid, 0.0), score)

    print(f"  Videos: {len(video_pid2sid):,}  Products: {len(product_pid2sid):,}")
    print(f"  Price items: {len(price_map):,}  Categories: {len(cat_price_map):,}")
    print(f"  Engagement users: {len(eng_map):,}")

    # Load source data
    print(f"\nLoading source data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows")

    results = []
    skipped_no_target = 0
    skipped_no_history = 0
    skipped_no_sid = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building ranked product data"):
        if row["split"] != 0:
            continue

        uid = int(row["uid"])
        target_pids = row["target_goods_pid"]
        hist_goods = row["hist_goods_pid"]
        hist_videos = row.get("hist_longview_video_list")

        if target_pids is None or (isinstance(target_pids, float) and pd.isna(target_pids)) or len(target_pids) == 0:
            skipped_no_target += 1
            continue

        target_pids = [int(p) for p in target_pids]

        # Build user content parts
        user_parts = []

        # 1. Video watch history
        if hist_videos is not None and not (isinstance(hist_videos, float) and pd.isna(hist_videos)):
            if len(hist_videos) > 0:
                video_sids = pids_to_sids(hist_videos[-VIDEO_HIST_MAX_LEN:], video_pid2sid)
                if video_sids:
                    video_prompt = random.choice(VIDEO_WATCH_PROMPTS)
                    user_parts.append(f"{video_prompt}{video_sids}")

        # 2. Product history (with SIDs)
        if hist_goods is not None and not (isinstance(hist_goods, float) and pd.isna(hist_goods)):
            if len(hist_goods) > 0:
                product_sids = pids_to_sids(hist_goods[-PRODUCT_HIST_MAX_LEN:], product_pid2sid)
                if product_sids:
                    product_prompt = random.choice(PRODUCT_HIST_PROMPTS)
                    user_parts.append(f"{product_prompt}{product_sids}")

        if not user_parts:
            skipped_no_history += 1
            continue

        # 3. Price tier context
        user_prices = []
        for pid in target_pids[:TARGET_MAX_LEN]:
            p = price_map.get(pid)
            if p:
                user_prices.append(p)
        if user_prices:
            avg_price = sum(user_prices) / len(user_prices)
        else:
            # Check hist_goods for price
            hist_prices = []
            if hist_goods is not None and hasattr(hist_goods, "__iter__"):
                for pid in hist_goods[:20]:
                    p = price_map.get(int(pid))
                    if p:
                        hist_prices.append(p)
            avg_price = sum(hist_prices) / len(hist_prices) if hist_prices else global_median

        tier = get_price_tier(avg_price, thresholds)
        tier_prompt = random.choice(PRICE_CONTEXT_PROMPTS).format(tier=tier)
        user_parts.append(tier_prompt)

        user_content = "\n".join(user_parts)
        task_prompt = random.choice(TASK_PROMPTS)

        # 4. Build target: ranked list of product SIDs sorted by price×funnel
        # Include BOTH viewed (hist_goods_pid, CTR) and purchased (target_goods_pid, CVR)
        # items so the NDCG reward captures CTR × CVR × Price.
        user_eng = eng_map.get(uid, {})

        # Collect all candidate items: viewed + purchased
        target_set = set(target_pids)
        all_candidate_pids: list[int] = list(target_pids[:TARGET_MAX_LEN])

        # Add viewed-but-not-purchased items (max 10 more to keep rank_labels ~20)
        if hist_goods is not None and hasattr(hist_goods, "__iter__") and not isinstance(hist_goods, float):
            extra_count = 0
            for pid in reversed(hist_goods):
                pid_int = int(pid)
                if pid_int not in target_set and pid_int in product_pid2sid:
                    all_candidate_pids.append(pid_int)
                    extra_count += 1
                    if extra_count >= TARGET_MAX_LEN:
                        break

        target_entries = []
        filtered_count = 0
        for pid in all_candidate_pids:
            if pid not in product_pid2sid:
                filtered_count += 1
                continue

            # Funnel: 1.0 if purchased (CVR), 0.1 if only viewed (CTR)
            if pid in target_set:
                funnel = 1.0   # purchased / converted
            else:
                funnel = 0.1   # viewed but not purchased
            # Use any higher funnel from engagement data
            funnel = max(funnel, user_eng.get(pid, 0.0))

            # Get price: 3-tier fallback
            if pid in price_map:
                price = price_map[pid]
            else:
                cat = cat_map.get(pid, "")
                price = cat_price_map.get(cat, global_median)

            pw = get_price_weight(price, global_max)
            relevance = funnel * pw
            target_entries.append({
                "pid": pid, "price": price, "funnel": funnel,
                "price_weight": pw, "relevance": relevance,
            })

        if not target_entries:
            skipped_no_sid += 1
            continue

        # Sort by relevance descending (ground truth ranking)
        target_entries.sort(key=lambda x: x["relevance"], reverse=True)

        # Build answer string
        answer_parts = []
        for entry in target_entries:
            code = product_pid2sid[entry["pid"]]
            sid = SID_FORMAT.format(c0=code[0], c1=code[1], c2=code[2])
            answer_parts.append(sid)
        answer = ''.join(answer_parts)

        # Build messages
        system_prompt = random.choice(SYSTEM_PROMPTS)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_content + "\n" + task_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]

        # Metadata with rank_labels for NDCG
        rank_labels = [
            {"price": entry["price"], "funnel": entry["funnel"]}
            for entry in target_entries
        ]
        metadata = {
            "uid": uid, "task": "ranked_product",
            "rank_labels": rank_labels,
            "price_tier": tier,
        }

        results.append({
            "source": "RecIF_RankedProduct",
            "uuid": str(uuid.uuid4()),
            "messages": json.dumps(messages, ensure_ascii=False),
            "metadata": json.dumps(metadata, ensure_ascii=False),
        })

    print(f"\nResults: {len(results):,} samples")
    print(f"  Skipped (no target): {skipped_no_target:,}")
    print(f"  Skipped (no history): {skipped_no_history:,}")
    print(f"  Skipped (no valid SID): {skipped_no_sid:,}")

    df_out = pd.DataFrame(results)
    out_path = output_dir / "ranked_product.parquet"
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
