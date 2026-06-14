# Totals Model — Notes & Evaluation
**Last updated:** 2026-06-01
**Model file:** `app/models/mlb_xgb_model.pkl`

---

## What We Built

A baseline totals model that anchors on the Vegas O/U closing line and makes small adjustments based on game-day context. The model is not designed to beat the market on its own — it provides a calibrated baseline. Edge comes from information layered on top (late injury news, line movement, line shopping across books).

---

## Data

### Odds Backfill
Historical O/U lines fetched from The Odds API paid historical endpoint (`/v4/historical/sports/baseball_mlb/odds`). Snapshot taken at **30 minutes before first pitch** (pre-game closing line).

Bookmaker priority: `hardrockbet → draftkings → fanduel → pinnacle → betmgm → ...`

HardRockBet has limited historical coverage; most historical games use DraftKings/FanDuel fallback. Pinnacle coverage is planned as primary for future retraining.

Coverage gaps (credits exhausted mid-backfill):
- 2026, 2025, 2024, 2023 (Apr–Aug 3): complete
- 2022 (Aug 18–Oct): complete
- 2022 (Apr–Aug 17): missing (~11k credits to finish)

### Timestamps
Added `games.timestamp_utc` (authoritative UTC first-pitch from MLB `gameDate` field). This eliminates a date-reconstruction bug where games crossing midnight UTC were snapshotted one day early, corrupting ~2,494 historical lines.

### Training data
- Seasons: 2022–2026 (May+ games only, `hr_total_runs_line` required)
- 7,703 games total after filters

---

## Model

**Algorithm:** XGBRegressor  
**Target:** `total_runs - hr_total_runs_line` (residual from Vegas line)  
At inference: `predicted_total = hr_total_runs_line + model.predict(features)`

**Hyperparameters:** n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective=`reg:absoluteerror`, early_stopping_rounds=50

**Sample weights:** Exponential decay, 365-day half-life (2022 games weighted ~20%)

### Features (14)
| Feature | Purpose |
|---|---|
| `hr_total_runs_line` | Market anchor |
| `temperature`, `wind_speed`, `wind_flag` | Game-day weather (set after line is posted) |
| `venue_run_factor` | Park run factor |
| `home/away_avg_runs_lastx_total` | Recent team run-scoring form (last 10 games) |
| `home/away_avg_runs_vs_arm` | Handedness matchup edge |
| `home/away_sp_last3_era` | Recent SP form (last 3 starts) |
| `home/away_bullpen_era` | Bullpen quality |
| `total_lineup_ops` | Combined lineup offensive strength |

**Dropped vs. prior version:** season-long ERA/WHIP/K9/BB9 (already priced into the line), individual lineup OPS, mixed-unit interaction features. These had flat importance (~3–5% each) and added noise rather than signal.

---

## Train/Test Split

| Set | Data |
|---|---|
| Train (eval mode) | 2022 – Jun 30 2025 (6,150 games) |
| Test (eval mode) | Jul 1 – Oct 2025 (1,162 games) |
| Train (production) | All seasons 2022–2026 YTD (7,703 games) |

---

## Results

### Vegas line baseline (H2-2025)
| Predictor | MAE |
|---|---|
| Naive mean (predict ~8.9 every game) | 3.534 |
| Vegas line alone | 3.500 |
| **Our model** | **3.496** |

### Eval metrics (H2-2025 holdout)
| Metric | Value |
|---|---|
| Residual MAE | 3.496 |
| Predict-zero baseline (= Vegas line) | 3.500 |
| Edge over Vegas line | +0.004 runs |
| Over/under direction accuracy | 50.9% |
| Best boosting round | 129 / 600 |

### Vegas line correlation
| Season | Corr(line, actual) |
|---|---|
| 2022 | 0.246 |
| 2023 | 0.208 |
| 2024 | 0.176 |
| 2025 | 0.165 |
| 2026 | 0.253 |
| Overall | 0.206 |

---

## Interpretation

The Vegas totals market is highly efficient. Our feature set (pitching, lineup, weather, park) is the same information bookmakers use when setting lines. The model adds ~0.004 runs of MAE improvement and calls over/under correctly 50.9% of the time — marginally above random, not enough to overcome -110 vig (breakeven: 52.4%).

**The model's value is as a calibrated baseline:**
- Predictions are anchored to the Vegas line (corr = 0.84 with the line)
- The residual approach ensures we never drift far from the market
- Edge will come from context the model doesn't have: late lineup changes, injury scratches, line movement signals, and shopping across books for the best number

---

## Retrain

```bash
# Eval (honest holdout)
docker compose run --rm app python training/train_totals_model.py --mode eval

# Production (deploy)
docker compose run --rm app python training/train_totals_model.py --mode production
```

After retraining with new credits (fill 2022 Apr–Aug gap):
```bash
docker compose run --rm app python scripts/backfill_historical_odds.py \
  --start 2022-04-01 --end 2022-08-16 --overwrite
```
