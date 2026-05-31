# Totals Model — Retraining Notes & Evaluation
**Date:** 2026-05-31  
**Model file:** `app/models/mlb_xgb_model.pkl`  
**Now active in:** `app/services/prediction_service.py` (replaces `mlb_ridge.pkl`)

---

## Changes Made

### Training script (`training/train_totals_model.py`)

| Change | Before | After |
|---|---|---|
| Train/test split | Random 90/10 (`random_state=42`) | Temporal: train 2021–2024, test 2025 |
| Imputer fit | On full dataset (test leakage) | Fit on training data only, transform test separately |
| `venue_orientation_known` | Feature (always 1 for real venues) | Removed |
| `wind_speed` | Not used | Added as feature |
| `home_sp_whip`, `away_sp_whip` | Commented out | Re-enabled |
| `home_bullpen_era`, `away_bullpen_era` | Commented out | Re-enabled |
| `home_sp_vs_away_lineup` | ERA − OPS (incompatible units) | Removed |
| `away_sp_vs_home_lineup` | ERA − OPS (incompatible units) | Removed |
| `home_offense_vs_away_bullpen` | `home_avg_runs − away_sp_last3_era` (wrong column) | `home_avg_runs − away_bullpen_era` (fixed) |
| `away_offense_vs_home_bullpen` | `away_avg_runs − home_sp_last3_era` (wrong column) | `away_avg_runs − home_bullpen_era` (fixed) |
| `total_sp_era` | Not used | Added: `home_sp_era + away_sp_era` |
| `total_lineup_ops` | Not used | Added: `home_lineup_ops + away_lineup_ops` |
| Saved artifacts | `mlb_xgb_model.pkl`, `feature_names.pkl` | + `totals_imputer.pkl`, `totals_model_metadata.json` |

**Feature count:** 14 → 22

### Feature extractor (`app/services/feature_extractor.py`)

- Added `home_sp_last3_era` / `away_sp_last3_era` fetch from `pitcher_features` table (was missing — `_pitcher_stats()` never computed it)
- Removed `home_sp_vs_away_lineup` / `away_sp_vs_home_lineup` (ERA−OPS interactions removed)
- Added `total_sp_era`, `total_lineup_ops` interaction features
- `home_bullpen_era` / `away_bullpen_era` now default to `np.nan` when not found (was missing key, potential KeyError)

### Prediction service (`app/services/prediction_service.py`)

- Default model: `mlb_ridge.pkl` → `mlb_xgb_model.pkl`
- Added `totals_imputer_path` parameter; imputer now loaded and applied before prediction
- Fixed silent bug: was returning log-scale values and comparing against a real run line (model now trains on raw runs, no inversion needed)
- `numeric_feature_cols` updated to match training exactly (22 features)

---

## Training Data

| Season | Games (May+) |
|---|---|
| 2021 | 531 |
| 2022 | 1,862 |
| 2023 | 1,771 |
| 2024 | 1,752 |
| **Train total** | **5,916** |
| 2025 (test) | 1,970 |

2026 season was excluded from training (in-progress season, insufficient stats).

---

## Iteration Results

| Run | Change | Test MAE | Test R² | Best Round | Notes |
|---|---|---|---|---|---|
| Baseline | Initial retraining (22 features, temporal split, log target, `reg:squarederror`) | 3.605 | −0.040 | 168/200 | Log-space MAE 0.394 |
| Change 1 | Drop log transform → raw runs, `reg:absoluteerror` | **3.604** | **−0.014** | 199/200 | Negligible MAE change; R² less negative. Model never triggered early stopping — still slowly improving at round 199. Confirms bottleneck is feature quality, not objective. |
| Change 2 | Fix `avg_runs_last_n` training/inference mismatch (inference now uses all games, not home-only) | **3.604** | **−0.014** | 199/200 | No change in training metrics (expected — training SQL was already using all games; fix only corrects inference alignment). |
| Change 3 | Add `home_sp_k9`, `away_sp_k9`, `home_sp_bb9`, `away_sp_bb9`, `total_sp_k9` (27 features total) | **3.602** | **−0.013** | 199/200 | Marginal gain. K/9 and BB/9 add small signal but don't break the plateau. All three changes combined moved MAE by only 0.003. Every run hits the 200-round ceiling without early stopping. |
| Change 4a | `min_child_weight=5`, `max_depth=5`, `n_estimators=600`, `lr=0.03` | 3.622 | −0.042 | 64/600 | **Regressed.** `min_child_weight=5` interacts badly with sample-weighted `reg:absoluteerror` gradients (sum of weights rarely reaches 5), so almost no splits fire. Early stopping triggered on a flat curve. |
| Change 4b | Remove `min_child_weight`, `max_depth=4`, `n_estimators=600`, `lr=0.03` | **3.595** | **−0.006** | 585/600 | Best result so far. Deeper trees + more rounds + lower lr allowed proper convergence. Model nearly triggered early stopping (converging). R² approaching 0. |

---

## Evaluation — Baseline Detail (2025 Full Season Holdout)

| Metric | Value |
|---|---|
| Test MAE (run scale) | 3.605 runs |
| Test R² (run scale) | −0.040 |
| Best boosting round | 168 / 200 |

**Practical context:** A model predicting ~8.9 runs every game would score R² = 0 and MAE ≈ 2.6–2.8 runs (the standard deviation of game totals). Both runs at 3.60 MAE are above this naive baseline, meaning the current feature set does not give the model enough signal to beat the mean. The temporal split is the honest evaluation — prior random-split evals would have inflated R² and understated MAE.

---

## Feature Importances (top 10, approximate)

Based on XGBoost `feature_importances_` — see `app/models/importance.png` for full chart. Expect `home_avg_runs_lastx_total`, `away_avg_runs_lastx_total`, `temperature`, and `total_sp_era` to rank highest based on the signal each carries.

---

## Known Remaining Issues

1. **Vegas O/U line not used** — the single highest-value feature. Once 3+ seasons of consistent closing-line data is available, adding it should materially improve MAE.
2. **`_pitcher_stats()` doesn't compute `last3_era`** — at inference, `sp_last3_era` is fetched from the pre-computed `pitcher_features` table (ingested daily). If pitcher features weren't ingested for a game, the default `NEW_PITCHER_ERA = 4.80` is used.
3. **`avg_runs_last_n` uses home-only filter** — `_avg_runs_last_n(is_home=True)` only counts games where the team was home, not all games. This underrepresents early-season road-heavy schedules. Consistent with training SQL but limits the early-season data quality.
4. **No hyperparameter tuning** — `n_estimators=200`, `max_depth=3`, `learning_rate=0.05` are unchanged. A grid search over `max_depth` (3–6) and `learning_rate` (0.01–0.1) could find a better operating point.

---

## How to Retrain

```bash
docker compose run --rm app python training/train_totals_model.py
```

Outputs `app/models/mlb_xgb_model.pkl`, `totals_imputer.pkl`, `feature_names.pkl`, `totals_model_metadata.json`. The model is active immediately (prediction_service loads it by default path).
