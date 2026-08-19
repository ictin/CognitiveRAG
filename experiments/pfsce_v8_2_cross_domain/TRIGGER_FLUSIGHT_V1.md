# Trigger FluSight v1 cross-domain validation

Protocol `PFSCE-V8.2-FLUSIGHT-RESIDUAL-V1` was frozen before the 2025-26 holdout scores were inspected. This file triggers execution without changing the model candidates, router, metrics, thresholds, or interpretation rules.

Rerun after implementation-only v1.0.1 fix: canonicalize numeric horizon identity (`1.0` -> `1`) before applying the already-frozen router. No scientific parameter or criterion changed.
