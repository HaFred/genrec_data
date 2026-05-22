# Model Evaluation Results

## OpenOneRec: Pretrained vs Post-trained

**Setup**: OneRec-1.7B, product recommendation, test.parquet (1000 samples), 16-beam sampling, thinking disabled.

| Metric | Pretrained | Post-trained | Delta | Δ% |
|--------|-----------|-------------|-------|-----|
| pass@1 | 0.00% | 1.00% | +1.00pp | +inf% |
| pass@5 | 0.40% | 6.10% | +5.70pp | +1425% |
| pass@10 | 0.80% | 9.90% | +9.10pp | +1138% |
| pass@16 | 0.80% | 12.20% | +11.40pp | +1425% |
| position1_pass@1 | 0.00% | 0.40% | +0.40pp | +inf% |
| position1_pass@5 | 0.20% | 1.00% | +0.80pp | +400% |
| position1_pass@10 | 0.30% | 2.20% | +1.90pp | +633% |
| position1_pass@16 | 0.30% | 2.40% | +2.10pp | +700% |
| recall@5 | 0.10% | 1.36% | +1.26pp | +1295% |
| recall@10 | 0.23% | 2.24% | +2.01pp | +888% |
| recall@16 | 0.23% | 2.77% | +2.55pp | +1126% |

**Models**:
- Pretrained: `models--OpenOneRec--OneRec-1.7B-pretrain`
- Post-trained: `models--OpenOneRec--OneRec-1.7B-pro`

**Eval script**: `local_backup/vanilla_openonerec_post_train_eval/eval_product_rec.py`

---

## Rank-GRPO: SFT vs GRPO

**Setup**: Qwen2.5-0.5B-Instruct, Reddit-v2 test set (200 unique contexts, 780 samples), greedy decoding, 1024 max tokens.

| Metric | SFT | GRPO (trl) | GRPO (verl-GR, step 40200) |
|--------|-----|-----------|---------------------------|
| Recall@5 | 7.44% | 10.46% | 8.39% |
| Recall@10 | 12.04% | 15.96% | 13.53% |
| Recall@15 | 14.68% | 19.36% | 17.19% |
| Recall@20 | 17.49% | 21.52% | 19.44% |
| NDCG@5 | 5.52% | 8.22% | 6.15% |
| NDCG@10 | 7.14% | 10.02% | 7.95% |
| NDCG@15 | 7.89% | 11.02% | 8.99% |
| NDCG@20 | 8.63% | 11.61% | 9.59% |

**Models**:
- SFT: `Qwen2.5-0.5B-Instruct/checkpoint-1500`
- GRPO: `Qwen2.5-0.5B-Instruct_lr1e-06_kl0.001/checkpoint-15800`

**Eval script**: `local_backup/vanilla_rankgrpo_post_train_eval/eval_rankgrpo.py`
