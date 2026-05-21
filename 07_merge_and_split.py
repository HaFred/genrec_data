#!/usr/bin/env python3
"""Stage 7: Merge all task parquets and split into train/test/eval_gmv.

Outputs: train.parquet, test.parquet, eval_gmv.parquet
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge and split GRPO datasets")
    parser.add_argument("--ranked_product", type=str, required=True)
    parser.add_argument("--purchase_pred", type=str, required=True)
    parser.add_argument("--cot_reason", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of test samples per task type")
    parser.add_argument("--eval_size", type=int, default=1000,
                        help="Number of eval_gmv samples (ranked product only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading task data...")
    dfs = {}
    for name, path in [
        ("ranked_product", args.ranked_product),
        ("purchase_pred", args.purchase_pred),
        ("cot_reason", args.cot_reason),
    ]:
        df = pd.read_parquet(path)
        dfs[name] = df
        print(f"  {name}: {len(df):,} samples")

    # Split each task into train/test
    train_parts = []
    test_parts = []
    eval_parts = []

    for name, df in dfs.items():
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        if name == "ranked_product":
            # Reserve eval_gmv samples
            eval_chunk = df.head(args.eval_size)
            remaining = df.iloc[args.eval_size:]
            test_chunk = remaining.head(args.test_size)
            train_chunk = remaining.iloc[args.test_size:]
            eval_parts.append(eval_chunk)
        else:
            test_chunk = df.head(args.test_size)
            train_chunk = df.iloc[args.test_size:]

        test_parts.append(test_chunk)
        train_parts.append(train_chunk)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    eval_df = pd.concat(eval_parts, ignore_index=True)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    eval_path = os.path.join(args.output_dir, "eval_gmv.parquet")

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    eval_df.to_parquet(eval_path, index=False)

    print(f"\nSplit results:")
    print(f"  train.parquet:    {len(train_df):,} samples")
    print(f"  test.parquet:     {len(test_df):,} samples")
    print(f"  eval_gmv.parquet: {len(eval_df):,} samples")

    # Print source distribution
    for split_name, split_df in [("train", train_df), ("test", test_df), ("eval_gmv", eval_df)]:
        print(f"\n  {split_name} source distribution:")
        for src, count in split_df["source"].value_counts().items():
            print(f"    {src}: {count:,}")

    print(f"\nSaved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
