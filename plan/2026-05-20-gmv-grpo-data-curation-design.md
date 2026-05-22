# GMV-Targeting GRPO Data Curation Design

**Date**: 2026-05-20
**Scope**: Data curation pipeline for enriching GRPO training data with price × engagement funnel rewards, targeting GMV improvement in product recommendation RLVR.

## Motivation

The current GRPO training data (train.parquet, 115K samples) uses only SID-matching accuracy as the reward signal. The source dataset (OpenOneRec-RecIF, 162K users) contains rich, unused signals: engagement labels (longview/like/follow/forward for 128K+ users), ad click labels (121K users), product purchase history (116K users), user shopping profiles with actual prices and ad conversions (88K users), and CoT reasoning chains (3.6K). This design describes a curation pipeline that extracts these signals into GMV-aligned reward annotations.

## Design Decisions (User-Confirmed)

1. **Primary reward**: Price × Engagement funnel (not pure SID accuracy)
2. **Price extraction**: Item-level price lookup from user shopping cart profiles
3. **Task scope**: Enriched product ranking (NDCG reward) + Purchase Prediction + CoT Reasoning
4. **Evaluation**: Standalone GMV eval script with before/after comparison
5. **Format**: Backward-compatible with existing `messages` + `metadata` parquet schema

## Architecture

### Pipeline Stages

```
Source Dataset (162K users, onerec_bench_release.parquet)
    + pid2caption.parquet + product_pid2sid.parquet + video_ad_pid2sid.parquet
    |
    v
Stage 1: Signal Extraction (3 scripts)
    scripts/01_build_price_lookup.py
      → Parse inter_user_profile_with_pid JSON fields
      → Build item_id → {price, category, purchase_count} mapping
      → Output: item_price_lookup.parquet

    scripts/02_compute_engagement_scores.py
      → From video/ad/goods history + target labels
      → Per user×item: engagement_funnel_score (0..1)
      → Output: user_item_engagement.parquet

    scripts/03_extract_cot_samples.py
      → Filter reco_cot non-null rows (3.6K)
      → Parse CoT text, extract reasoning chain
      → Output: cot_samples.parquet
    |
    v
Stage 2: Dataset Assembly (3 scripts)
    scripts/04_build_ranked_product_task.py      → ~100K samples
    scripts/05_build_purchase_prediction_task.py  → ~50K samples
    scripts/06_build_cot_reasoning_task.py        → ~3K samples
    |
    v
Stage 3: Merge & Split (1 script)
    scripts/07_merge_and_split.py
      → Merge all task parquets
      → Stratified split: train.parquet / test.parquet / eval_gmv.parquet
    |
    v
Stage 4: Integration (modified recipe + unchanged launch)
    Modified: onerec_recipe.py (extract_prompt_fields, compute_score, __getitem__)
    Unchanged: run_openonerec_grpo.sh, two_stage_agent_loop.py
    |
    v
Stage 5: GMV Evaluation (1 script + 1 wrapper)
    scripts/08_eval_gmv.py
    scripts/08_eval_gmv_compare.sh
```

### Output Files

```
openonerec_fredfork/data/outputs/
  item_price_lookup.parquet
  user_item_engagement.parquet
  cot_samples.parquet
  rl_data_enriched/
    train.parquet
    test.parquet
    eval_gmv.parquet
```

## Reward Design

### Engagement Funnel Score (per user×item)

Based on the naturally hierarchical engagement signals in the source data:

| Level | Signal | Weight | Source Field |
|-------|--------|--------|-------------|
| 0 | no engagement | 0.0 | — |
| 1 | longview (watched >N seconds) | 0.1 | `target_video_longview` |
| 2 | like | 0.2 | `target_video_like` |
| 3 | forward/share | 0.3 | `target_video_forward` |
| 4 | follow creator | 0.4 | `target_video_follow` |
| 5 | ad click | 0.6 | `target_ad_pid` (non-empty) |
| 6 | purchase / cart add | 1.0 | `target_goods_pid` (non-empty), shopping cart |

For a given item, `funnel_score = max(level_i where signal_i == 1, default=0)`.

### Price Weight (per item)

Built from `inter_user_profile_with_pid` shopping cart additions. Empirical verification (2026-05-20) found: 57,730 unique products with direct prices, 2,053 unique product categories with median prices. Price distribution: min ¥0.01, median ¥16.88, p90 ¥89.00, max ¥9,999.90 (heavy right skew).

**3-tier fallback for price assignment:**

```
Tier 1 (direct):    price = item_price from shopping cart    — covers ~9% of target items
Tier 2 (category):  price = category_median_price            — extends to ~25% of target items
Tier 3 (global):    price = global_median = ¥16.88           — remaining items (neutral baseline)
```

**Price weight formula (log-scaled):**

```
price_weight(item) = log(1 + price_in_yuan) / log(1 + max_price)
```

Normalized to [0, 1]. With median ¥16.88 and max ¥9,999.90:
- Tier 3 (global median): `log(1+16.88) / log(1+9999.9) ≈ 0.31`
- P90 item (¥89): `≈ 0.49`
- High-end (¥999): `≈ 0.75`
- Premium (¥9999): `≈ 1.0`

This ensures the model gets price signal on 65% of users (those with >=1 priced target item) while maintaining a neutral baseline for items without price data.

### NDCG Reward (for Ranked Product task)

For a ranked list prediction of K items:

```
relevance(item_k) = funnel_score(item_k) * price_weight(item_k)
DCG@K = sum_{k=1}^{K} relevance(item_k) / log2(k+1)
IDCG@K = DCG of ground-truth ranking (items sorted by relevance descending)
NDCG@K = DCG@K / IDCG@K

final_score = NDCG@K
```

### Purchase Prediction Reward

```
score = predict_purchase_correct ? price_weight(item) : 0
```

### CoT Reasoning Reward

```
score = 0.3 * think_format_reward + 0.7 * (SID_correct ? price_weight(item) : 0)
```

### Backward Compatibility

`compute_score` dispatches on `extra_info.get("task")`. Legacy `RecIF_ProductRec` data (no task field) uses the existing SID-matching path unchanged.

## Dataset Specifications

All tasks use the existing Chinese prompt style from `onerec_data/sft/product_rec.py`. Schema: `source`, `uuid`, `messages`, `metadata` — same columns as current data.

### Task 1: Enriched Product Ranking (`source: "RecIF_RankedProduct"`, ~100K)

Uses video watch history + purchase history with price tiers → ranked product list output.

Metadata carries per-rank price and funnel scores:
```json
{"uid": 16332, "task": "ranked_product",
 "rank_labels": [{"price": 999.0, "funnel": 1.0}, {"price": 89.0, "funnel": 0.4}]}
```

### Task 2: Purchase Prediction (`source: "RecIF_PurchasePred"`, ~50K)

Binary classification with price tier: given user history + candidate product + price, predict purchase likelihood.

Metadata: `{"uid": X, "task": "purchase_pred", "label": 1, "price": 1299.0, "price_tier": "high"}`

### Task 3: CoT Reasoning (`source: "RecIF_CotReason"`, ~3K)

Enriches the 3.6K existing `reco_cot` samples with price context from user profiles.

Metadata: `{"uid": X, "task": "cot_reason", "gt_sid": "...", "price": X}`

### Splits

- Train: ~150K (stratified across 3 task types)
- Test: ~3K (1000 per task type)
- Eval GMV: ~3K (Ranked Product only, disjoint from train/test)

## Recipe Modifications (onerec_recipe.py)

Three backward-compatible changes:

1. **`extract_prompt_fields`** (line 88-89): Parse `metadata` JSON, copy `task`, `rank_labels`, `price`, `price_tier` into `reward_model` dict.

2. **`compute_score`** (line 545): Dispatch to `_compute_ndcg_score`, `_compute_purchase_score`, or `_compute_cot_score` based on `extra_info.get("task")`. Legacy path unchanged.

3. **`__getitem__`** (line 284): Read `metadata` from row and add reward fields to `extra_info`.

## GMV Eval Framework

### scripts/08_eval_gmv.py

Standalone script that loads a model checkpoint, runs inference on `eval_gmv.parquet` (Ranked Product task only), and computes:

**Primary GMV metrics:**
- Expected-GMV@K: sum over predictions of (hit_k * price_k)
- Price-NDCG@K: NDCG with price×funnel as relevance
- GMV-Recall@K: sum price of correct predictions / sum price of all gt items

**Secondary metrics:**
- Funnel-NDCG@K: NDCG with funnel only (ablation)
- Hit@K: raw SID hit rate (backward compat)
- Price-tier-accuracy: % predictions in correct tier
- Avg-price-predicted: mean price of top-K predictions

### scripts/08_eval_gmv_compare.sh

Runs `08_eval_gmv.py` on both BASE_MODEL and GRPO checkpoint, produces side-by-side comparison:

```
Metric                  Base      GRPO      Δ        Δ%
Expected-GMV@5          187.6     234.5    +46.9    +25.0%
Price-NDCG@10           0.410     0.478    +0.068   +16.6%
...
```

## Key Risks and Mitigations

1. **Sparse price data**: Only 9.1% of target items have direct price; 25.5% have category-level price. Mitigation: 3-tier fallback (direct → category-median → global-median ¥16.88). 65% of users have >=1 priced target item, so per-sample feedback is frequent. Log-transform handles the ¥0.01–¥9,999.90 skew.

2. **Engagement label sparsity**: Follow (3.1K) and forward (7.5K) have few positives. Mitigation: funnel scores use max across labels, so sparse labels don't zero out reward.

3. **Reward hacking**: Model could learn to predict only high-price items regardless of relevance. Mitigation: NDCG reward requires both relevance (engagement) AND price, preventing pure price gaming.

4. **NDCG requires ranked list output format**: Current model outputs SIDs sequentially. Mitigation: treat beam search results as the ranked list; each beam position maps to rank k.

## Clarifications

**K in NDCG@K**: K = beam_width = 32, matching the rollout config's `stage2_beam_size`. During training, the beam search produces 32 ranked items; NDCG is computed over all 32 positions. During eval, K values of 1, 5, 10, and 32 are reported.

**Cross-domain funnel scoring**: The engagement funnel is computed at the user level (which video signals did this user exhibit), while rewards are for product items. For the Ranked Product task, funnel scores are assigned per product item as follows:
- If the product is in `target_goods_pid` (purchased/converted): funnel = 1.0 (level 6, CVR)
- If the product is in `hist_goods_pid` but NOT in target_goods_pid (viewed but not purchased): funnel = 0.1 (level 1, CTR)
- The user's video engagement levels inform the overall user engagement profile but are not per-product-item signals.

**rank_labels must include BOTH viewed and purchased items** so the NDCG reward captures the full CTR×CVR×Price funnel. Items are sorted by price×funnel descending. IDCG represents the ideal ordering where purchased items (funnel=1.0) rank above merely-viewed items (funnel=0.1), both weighted by price. This encodes the GMV definition:
```
GMV_approx = DCG where relevance = funnel × log(1+price)/log(1+max_price)
           = sum over CTR items (0.1 × price_weight) + sum over CVR items (1.0 × price_weight)
           ∝ CTR × CVR × Price
```

**Price tier thresholds**: Computed from the full item price distribution. Low: 0-33rd percentile, Med: 33rd-67th percentile, High: 67th-100th percentile. These are computed once globally from `item_price_lookup.parquet` and stored as absolute thresholds.

**Ground truth ranking order**: Items in `rank_labels` are sorted by `price × funnel_score` descending. The IDCG is computed from this ordering. The model's predicted order (beam position) is scored against this ground truth via NDCG.

**Legacy task compatibility**: `compute_score` checks `extra_info.get("task")`. If absent or `None` (current data), it falls through to the existing SID-matching path. No changes needed to existing train.parquet/test.parquet files.

## Files Changed/Created

### New Files
- `openonerec_fredfork/data/scripts_advanced/01_build_price_lookup.py`
- `openonerec_fredfork/data/scripts_advanced/02_compute_engagement_scores.py`
- `openonerec_fredfork/data/scripts_advanced/03_extract_cot_samples.py`
- `openonerec_fredfork/data/scripts_advanced/04_build_ranked_product_task.py`
- `openonerec_fredfork/data/scripts_advanced/05_build_purchase_prediction_task.py`
- `openonerec_fredfork/data/scripts_advanced/06_build_cot_reasoning_task.py`
- `openonerec_fredfork/data/scripts_advanced/07_merge_and_split.py`
- `openonerec_fredfork/data/scripts_advanced/08_eval_gmv.py`
- `openonerec_fredfork/data/scripts_advanced/08_eval_gmv_compare.sh`

### Modified Files
- `verl_gr/recipes/openonerec/onerec_recipe.py` (extract_prompt_fields, compute_score, __getitem__)
- `scripts/.matchup_openonerec.sh` (add data curation step before training)
