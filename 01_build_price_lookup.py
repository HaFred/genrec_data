#!/usr/bin/env python3
"""Stage 1: Build item-level price lookup from user shopping cart profiles.

Parses inter_user_profile_with_pid JSON fields from the OpenOneRec-RecIF dataset,
extracts shopping cart entries with prices, and builds:
  - item_id → {price, category, purchase_count} mapping
  - category-level median prices

3-tier fallback:
  Tier 1: direct price from shopping cart
  Tier 2: category-median price
  Tier 3: global-median price

Output: item_price_lookup.parquet, category_price_lookup.parquet
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from statistics import median

import numpy as np
import pandas as pd
from tqdm import tqdm

SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'


def parse_item_id(pid_str: str) -> int | None:
    """Parse '<itemid|795666>' → 795666."""
    if not isinstance(pid_str, str):
        return None
    if "itemid|" in pid_str:
        try:
            return int(pid_str.split("itemid|")[1].rstrip(">"))
        except (ValueError, IndexError):
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Build item price lookup")
    parser.add_argument("--input", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet")
    parser.add_argument("--product_pid2sid", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/product_pid2sid.parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_users", type=int, default=-1,
                        help="Max user profiles to parse (-1 = all)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load source data
    print(f"Loading source data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows")

    # Load product pid2sid for validation
    print(f"Loading product pid2sid from {args.product_pid2sid} ...")
    df_psid = pd.read_parquet(args.product_pid2sid)
    product_pid2sid = dict(zip(df_psid["pid"], df_psid["sid"]))
    print(f"  {len(product_pid2sid):,} product SID mappings")

    # Extract price data from user profiles
    item_prices: dict[int, list[float]] = defaultdict(list)
    item_categories: dict[int, str] = {}
    item_names: dict[int, str] = {}
    category_prices: dict[str, list[float]] = defaultdict(list)
    total_entries = 0

    rows_iter = df.iterrows()
    if args.max_users > 0:
        rows_iter = list(df.head(args.max_users).iterrows())

    for _, row in tqdm(rows_iter, total=min(len(df), args.max_users) if args.max_users > 0 else len(df),
                       desc="Parsing profiles"):
        profile = row.get("inter_user_profile_with_pid")
        if not isinstance(profile, str):
            continue
        try:
            pj = json.loads(profile)
        except (json.JSONDecodeError, TypeError):
            continue

        cart_items = pj.get("最近添加的购物车", [])
        for item in cart_items:
            pid_str = item.get("商品ID", "")
            price = item.get("商品单价")
            name = item.get("商品名称", "")
            cat = item.get("类目", "")

            pid = parse_item_id(pid_str)
            if pid is None:
                continue
            if not isinstance(price, (int, float)) or price <= 0:
                continue

            total_entries += 1
            item_prices[pid].append(float(price))
            if cat and pid not in item_categories:
                item_categories[pid] = cat
            if name and pid not in item_names:
                item_names[pid] = name
            if cat:
                category_prices[cat].append(float(price))

    print(f"\nParsed {total_entries:,} cart entries with prices")

    # Build item-level lookup
    item_lookup = {}
    for pid, prices in item_prices.items():
        item_lookup[pid] = {
            "median_price": float(np.median(prices)),
            "mean_price": float(np.mean(prices)),
            "min_price": float(min(prices)),
            "max_price": float(max(prices)),
            "purchase_count": len(prices),
            "category": item_categories.get(pid, ""),
            "name": item_names.get(pid, ""),
            "has_sid": pid in product_pid2sid,
        }

    df_items = pd.DataFrame.from_dict(item_lookup, orient="index")
    df_items.index.name = "pid"
    df_items = df_items.reset_index()
    print(f"  Unique items with prices: {len(df_items):,}")
    print(f"  Items with valid product SID: {df_items['has_sid'].sum():,}")

    # Build category-level lookup
    cat_lookup = {}
    for cat, prices in category_prices.items():
        cat_lookup[cat] = {
            "median_price": float(np.median(prices)),
            "mean_price": float(np.mean(prices)),
            "item_count": len(prices),
            "price_p10": float(np.percentile(prices, 10)),
            "price_p90": float(np.percentile(prices, 90)),
        }

    df_cats = pd.DataFrame.from_dict(cat_lookup, orient="index")
    df_cats.index.name = "category"
    df_cats = df_cats.reset_index()
    print(f"  Unique categories: {len(df_cats):,}")

    # Compute global stats
    all_prices = [p for prices in item_prices.values() for p in prices]
    global_median = float(np.median(all_prices))
    global_mean = float(np.mean(all_prices))
    print(f"\nGlobal price stats:")
    print(f"  Median: ¥{global_median:.2f}")
    print(f"  Mean:   ¥{global_mean:.2f}")
    print(f"  Min:    ¥{min(all_prices):.2f}")
    print(f"  Max:    ¥{max(all_prices):.2f}")
    print(f"  P10:    ¥{np.percentile(all_prices, 10):.2f}")
    print(f"  P90:    ¥{np.percentile(all_prices, 90):.2f}")

    # Save
    items_path = os.path.join(output_dir, "item_price_lookup.parquet")
    cats_path = os.path.join(output_dir, "category_price_lookup.parquet")
    stats_path = os.path.join(output_dir, "price_stats.json")

    df_items.to_parquet(items_path, index=False)
    df_cats.to_parquet(cats_path, index=False)

    stats = {
        "global_median": global_median,
        "global_mean": global_mean,
        "global_p10": float(np.percentile(all_prices, 10)),
        "global_p90": float(np.percentile(all_prices, 90)),
        "global_max": float(max(all_prices)),
        "num_items_with_price": len(item_lookup),
        "num_categories": len(cat_lookup),
        "num_cart_entries": total_entries,
        "tier_thresholds": {
            "low": float(np.percentile(all_prices, 33)),
            "med": float(np.percentile(all_prices, 67)),
            "high": float(np.percentile(all_prices, 100)),
        },
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  {items_path} ({len(df_items):,} items)")
    print(f"  {cats_path} ({len(df_cats):,} categories)")
    print(f"  {stats_path}")
    print("Done!")


if __name__ == "__main__":
    main()
