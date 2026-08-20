# PFSCE v8.2 Prospective Frozen Artifact Registry — 2026-08-20

Purpose: provide an append-only registry of the initial prospective validation artifacts and their Git blob identities before the first NECF live issue. Later artifacts are added as new registry generations; existing rows are not rewritten to hide changes.

| Artifact | Git blob SHA |
|---|---|
| `PROSPECTIVE_VALIDATION_FREEZE_20260820.md` | `b238b34ef1873b9f5aa3a88288d8b5a129a92aa2` |
| `NECF_PROSPECTIVE_B1_B2V2_FREEZE_20260820.md` | `fb801c2877fd91f8563fbb5c8e8220b5bc53b556` |
| `NECF_FIRST_RUN_INTEGRITY_AUDIT_FREEZE_20260820.md` | `3ab3d92176b210ad9a65e34a13b851571c19e05f` |
| `NECF_LIVE_SOURCE_SELECTION_CLARIFICATION_20260820.md` | `a5e3b8c5f0aebca8eb7f2c555319e751c21f36ea` |
| `GENERAL_COHORT_20260820.json` | `0970028c341432d5c107cac4db3273f37bd2ad65` |
| `GENERAL_COHORT_20260820.md` | `cbddb7e3d48506d889af6710ce4b86e0118051e5` |
| `GENERAL_COHORT_EXTENSION_20260820_0853Z.md` | `75bb31fb08e629ece2285fc59db4d93dd6cb9465` |
| `cycles/2026-08-20-cycle-001-contract-freeze.md` | `8b0f301e0052785d3c61ca4157f9d8d943fc0f4c` |
| `cycles/2026-08-20-cycle-001-forecast-ledger.json` | `bd04f1c6d70e6246aec65096e18f5b428957b198` |

## Integrity policy

- Git history is the immutability mechanism for first-issue forecasts and contracts.
- Corrections are new artifacts/commits with explicit reason codes; original issue probabilities remain recoverable.
- `validate_prospective_artifacts.py` is an automated structural/denominator/provenance gate, not a substitute for the frozen 15-check NECF first-run audit.
- A successful structural CI run does not imply predictive skill or F2 authority.
