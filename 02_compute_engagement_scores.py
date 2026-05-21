#!/usr/bin/env python3
"""Stage 2: Compute per-user×item engagement funnel scores.

Based on the hierarchical engagement signals in the source data:
  Level 0: no engagement (0.0)
  Level 1: longview (0.1)
  Level 2: like (0.2)
  Level 3: forward/share (0.3)
  Level 4: follow creator (0.4)
  Level 5: ad click (0.6)
  Level 6: purchase / cart add (1.0)

Output: user_item_engagement.parquet (uid, pid, funnel_score, level, source)
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

FUNNEL_WEIGHTS = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 1.0}
FUNNEL_NAMES = {
    0: "none", 1: "longview", 2: "like", 3: "forward",
    4: "follow", 5: "ad_click", 6: "purchase",
}


def compute_item_funnel(row, item_idx: int, item_pid: int) -> tuple[int, float]:
    """Determine the max engagement level for a specific item."""
    level = 0

    # Level 6: purchase / cart add (target_goods_pid)
    target_goods = row.get("target_goods_pid")
    if target_goods is not None and hasattr(target_goods, "__iter__"):
        if item_pid in target_goods:
            return 6, FUNNEL_WEIGHTS[6]

    # Level 5: ad click (target_ad_pid)
    target_ads = row.get("target_ad_pid")
    if target_ads is not None and hasattr(target_ads, "__iter__"):
        if item_pid in target_ads:
            level = max(level, 5)

    # Video engagement signals (for video/ad items)
    for signal_name, signal_level in [
        ("target_video_follow", 4),
        ("target_video_forward", 3),
        ("target_video_like", 2),
        ("target_video_longview", 1),
    ]:
        signals = row.get(signal_name)
        if signals is None or not hasattr(signals, "__iter__"):
            continue
        try:
            if item_idx < len(signals) and signals[item_idx] == 1:
                level = max(level, signal_level)
        except (IndexError, TypeError):
            continue

    return level, FUNNEL_WEIGHTS[level]


def main():
    parser = argparse.ArgumentParser(description="Compute engagement funnel scores")
    parser.add_argument("--input", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser45/.cache/huggingface/hub/"
                                "datasets--OpenOneRec--OpenOneRec-RecIF/snapshots/"
                                "8f7cf2ee0b949e955a87a708d02024687be232c8/onerec_bench_release.parquet")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_users", type=int, default=-1)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading source data from {args.input} ...")
    df = pd.read_parquet(args.input)
    print(f"  {len(df):,} rows")

    if args.max_users > 0:
        df = df.head(args.max_users)

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing funnel scores"):
        uid = int(row["uid"])

        # Process target_goods (product purchases)
        target_goods = row.get("target_goods_pid")
        if target_goods is not None and hasattr(target_goods, "__iter__"):
            for pid in target_goods:
                records.append({
                    "uid": uid, "pid": int(pid),
                    "funnel_score": FUNNEL_WEIGHTS[6], "level": 6,
                    "source": "goods",
                })

        # Process hist_goods (browsed products - level 1 if not in target)
        target_goods_set = set(int(p) for p in target_goods) if target_goods is not None and hasattr(target_goods, "__iter__") else set()
        hist_goods = row.get("hist_goods_pid")
        if hist_goods is not None and hasattr(hist_goods, "__iter__"):
            for pid in hist_goods:
                pid_int = int(pid)
                if pid_int not in target_goods_set:
                    records.append({
                        "uid": uid, "pid": pid_int,
                        "funnel_score": FUNNEL_WEIGHTS[1], "level": 1,
                        "source": "goods_browse",
                    })

        # Process target_ads (ad clicks - level 5)
        target_ads = row.get("target_ad_pid")
        if target_ads is not None and hasattr(target_ads, "__iter__"):
            for pid in target_ads:
                records.append({
                    "uid": uid, "pid": int(pid),
                    "funnel_score": FUNNEL_WEIGHTS[5], "level": 5,
                    "source": "ad",
                })

        # Process video engagement per target video
        for label, level in [("target_video_longview", 1), ("target_video_like", 2),
                              ("target_video_forward", 3), ("target_video_follow", 4)]:
            target_videos = row.get("target_video_pid", [])
            labels = row.get(label)
            if target_videos is None or labels is None:
                continue
            if not hasattr(target_videos, "__iter__") or not hasattr(labels, "__iter__"):
                continue
            for i, (vid, lab) in enumerate(zip(target_videos, labels)):
                if lab == 1:
                    records.append({
                        "uid": uid, "pid": int(vid),
                        "funnel_score": FUNNEL_WEIGHTS[level], "level": level,
                        "source": f"video_{FUNNEL_NAMES[level]}",
                    })

    df_out = pd.DataFrame(records)

    # Deduplicate: per (uid, pid) keep max funnel_score
    df_out = df_out.sort_values("funnel_score", ascending=False)
    df_out = df_out.drop_duplicates(subset=["uid", "pid"], keep="first")
    df_out = df_out.reset_index(drop=True)

    print(f"\nEngagement records: {len(df_out):,}")
    print(f"  Unique users: {df_out['uid'].nunique():,}")
    print(f"  Unique items: {df_out['pid'].nunique():,}")
    print(f"  Level distribution:")
    for level in range(7):
        count = (df_out["level"] == level).sum()
        pct = 100 * count / max(1, len(df_out))
        print(f"    Level {level} ({FUNNEL_NAMES[level]}): {count:,} ({pct:.1f}%)")

    out_path = os.path.join(output_dir, "user_item_engagement.parquet")
    df_out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
