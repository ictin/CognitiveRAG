# B1 2025 Replication Verdict

Replication: `NECF-001-WP2-REPLICATION-2025-v1`
Authority: `V1_PROVISIONAL_REVISION_SENSITIVE_REPLICATION`

**Verdict: STRONG_REPLICATION.**

Operator MAE: **2016.674 MW**
Frozen routed B1 MAE: **1643.176 MW**
Aggregate routed MAE improvement: **18.520%**
BA-week bootstrap 95% CI: **[10.134%, 14.368%]**
BA-week clusters: **212**

## Frozen replication gate

- aggregate_lift_ge_10pct: **PASS**
- BA_week_bootstrap_lower_gt_zero: **PASS**
- CISO_and_PJM_positive: **PASS**
- fallback_BAs_unchanged: **PASS**

This replication used no 2024 or 2025 outcomes for fitting, hyperparameter selection, stack selection, or router selection. 2024 observations may enter only as prior-state lags for 2025 targets, which is temporally legitimate. The 2025 target archive is a post-period EIA/PUDL vintage, so the result remains revision-sensitive rather than highest-authority time-capsule evidence.
