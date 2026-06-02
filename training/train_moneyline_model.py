import argparse
import json
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
import joblib
import xgboost as xgb
from infra.db.init_db import engine


class PlattCalibratedPipeline:
    """Preprocessor → XGBoost → Platt sigmoid calibration, saveable with joblib."""
    def __init__(self, preprocessor, xgb_clf, platt):
        self.preprocessor = preprocessor
        self.xgb_clf      = xgb_clf
        self.platt        = platt  # LogisticRegression fit on logit(raw_prob)

    def predict_proba(self, X_raw):
        X_enc   = self.preprocessor.transform(X_raw)
        raw     = self.xgb_clf.predict_proba(X_enc)[:, 1]
        logit   = np.log(raw.clip(1e-9, 1 - 1e-9) / (1 - raw.clip(1e-9, 1 - 1e-9)))
        cal     = self.platt.predict_proba(logit.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])

EVAL_CUTOFF = pd.Timestamp("2025-07-01")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["eval", "production"], default="eval")
args = parser.parse_args()

# Column aliases match feature_extractor.extract_ML_features_from_game() output exactly.
game_features_query = text("""
    WITH game_base AS (
        SELECT g.id AS game_id, g.date, g.season_year, g.home_team, g.away_team,
               g.home_probable_pitcher_id, g.away_probable_pitcher_id
        FROM games g
    ),
    ranked_home_ops AS (
        SELECT gb.game_id, gs.player_id,
            (SUM(gs.hits + gs.walks_batting)*1.0 / NULLIF(SUM(gs.at_bats + gs.walks_batting), 0) +
             SUM((gs.hits-gs.doubles-gs.triples-gs.home_runs)+2*gs.doubles+3*gs.triples+4*gs.home_runs)*1.0
             / NULLIF(SUM(gs.at_bats), 0)
            ) AS ops,
            ROW_NUMBER() OVER (PARTITION BY gb.game_id ORDER BY (
                SUM(gs.hits+gs.walks_batting)*1.0/NULLIF(SUM(NULLIF(gs.at_bats+gs.walks_batting,0)),0)+
                SUM(gs.doubles+2*gs.triples+3*gs.home_runs)*1.0/NULLIF(SUM(NULLIF(gs.at_bats,0)),0)
            ) DESC NULLS LAST) AS rn
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.home_team = gs.team AND g2.season_year = gb.season_year AND g2.date < gb.date
        GROUP BY gb.game_id, gs.player_id
        HAVING SUM(gs.at_bats) >= 50
    ),
    lineup_ops_home AS (
        SELECT game_id, AVG(ops) AS team_top5_avg_ops FROM ranked_home_ops WHERE rn <= 5 GROUP BY game_id
    ),
    ranked_away_ops AS (
        SELECT gb.game_id, gs.player_id,
            (SUM(gs.hits + gs.walks_batting)*1.0 / NULLIF(SUM(gs.at_bats + gs.walks_batting), 0) +
             SUM((gs.hits-gs.doubles-gs.triples-gs.home_runs)+2*gs.doubles+3*gs.triples+4*gs.home_runs)*1.0
             / NULLIF(SUM(gs.at_bats), 0)
            ) AS ops,
            ROW_NUMBER() OVER (PARTITION BY gb.game_id ORDER BY (
                SUM(gs.hits+gs.walks_batting)*1.0/NULLIF(SUM(NULLIF(gs.at_bats+gs.walks_batting,0)),0)+
                SUM(gs.doubles+2*gs.triples+3*gs.home_runs)*1.0/NULLIF(SUM(NULLIF(gs.at_bats,0)),0)
            ) DESC NULLS LAST) AS rn
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.away_team = gs.team AND g2.season_year = gb.season_year AND g2.date < gb.date
        GROUP BY gb.game_id, gs.player_id
        HAVING SUM(gs.at_bats) >= 50
    ),
    lineup_ops_away AS (
        SELECT game_id, AVG(ops) AS team_top5_avg_ops FROM ranked_away_ops WHERE rn <= 5 GROUP BY game_id
    ),
    pitcher_handedness AS (
        SELECT g.id AS game_id,
               p_home.throwing_hand AS home_throwing_hand,
               p_away.throwing_hand AS away_throwing_hand
        FROM games g
        LEFT JOIN players p_home ON p_home.id = g.home_probable_pitcher_id
        LEFT JOIN players p_away ON p_away.id = g.away_probable_pitcher_id
    ),
    team_runs_vs_arm_home AS (
        SELECT gb.game_id, AVG(g.home_score) AS home_avg_runs_vs_arm
        FROM game_base gb
        JOIN pitcher_handedness ph ON ph.game_id = gb.game_id
        JOIN games g ON g.home_team = gb.home_team
        JOIN players p ON p.id = g.away_probable_pitcher_id
        WHERE g.date < gb.date AND g.season_year = gb.season_year
          AND ((ph.away_throwing_hand = 'L' AND p.throwing_hand = 'L') OR
               (ph.away_throwing_hand = 'R' AND p.throwing_hand = 'R'))
        GROUP BY gb.game_id
    ),
    team_runs_vs_arm_away AS (
        SELECT gb.game_id, AVG(g.away_score) AS away_avg_runs_vs_arm
        FROM game_base gb
        JOIN pitcher_handedness ph ON ph.game_id = gb.game_id
        JOIN games g ON g.away_team = gb.away_team
        JOIN players p ON p.id = g.home_probable_pitcher_id
        WHERE g.date < gb.date AND g.season_year = gb.season_year
          AND ((ph.home_throwing_hand = 'L' AND p.throwing_hand = 'L') OR
               (ph.home_throwing_hand = 'R' AND p.throwing_hand = 'R'))
        GROUP BY gb.game_id
    )

    SELECT
        g.id AS game_id, g.date, g.season_year,
        g.home_score, g.away_score,
        g.home_ml_price, g.away_ml_price,

        ph.home_throwing_hand,
        ph.away_throwing_hand,

        th.wins_season       AS home_team_wins,
        th.losses_season     AS home_team_losses,
        th.wins_last10       AS home_team_wins_last10,
        th.avg_runs_last10   AS home_avg_runs_lastx_total,
        th.run_diff_season   AS home_run_diff,
        bh.bullpen_era       AS home_bullpen_era,
        loh.team_top5_avg_ops AS home_lineup_ops,
        trh.home_avg_runs_vs_arm AS home_avg_runs_vs_arm,

        ta.wins_season       AS away_team_wins,
        ta.losses_season     AS away_team_losses,
        ta.wins_last10       AS away_team_wins_last10,
        ta.avg_runs_last10   AS away_avg_runs_lastx_total,
        ta.run_diff_season   AS away_run_diff,
        ba.bullpen_era       AS away_bullpen_era,
        loa.team_top5_avg_ops AS away_lineup_ops,
        tra.away_avg_runs_vs_arm AS away_avg_runs_vs_arm,

        pf_home.sp_era       AS home_sp_era,
        pf_home.sp_whip      AS home_sp_whip,
        pf_home.last3_era    AS home_sp_last3_era,

        pf_away.sp_era       AS away_sp_era,
        pf_away.sp_whip      AS away_sp_whip,
        pf_away.last3_era    AS away_sp_last3_era

    FROM games g
    LEFT JOIN team_features th ON th.game_id = g.id AND th.team = g.home_team
    LEFT JOIN team_features ta ON ta.game_id = g.id AND ta.team = g.away_team
    LEFT JOIN bullpen_features bh ON bh.game_id = g.id AND bh.team = g.home_team
    LEFT JOIN bullpen_features ba ON ba.game_id = g.id AND ba.team = g.away_team
    LEFT JOIN pitcher_features pf_home
        ON pf_home.game_id = g.id AND pf_home.player_id = g.home_probable_pitcher_id
    LEFT JOIN pitcher_features pf_away
        ON pf_away.game_id = g.id AND pf_away.player_id = g.away_probable_pitcher_id
    LEFT JOIN lineup_ops_home loh ON loh.game_id = g.id
    LEFT JOIN lineup_ops_away loa ON loa.game_id = g.id
    LEFT JOIN team_runs_vs_arm_home trh ON trh.game_id = g.id
    LEFT JOIN team_runs_vs_arm_away tra ON tra.game_id = g.id
    LEFT JOIN pitcher_handedness ph ON ph.game_id = g.id
    WHERE g.home_score IS NOT NULL
      AND g.away_score IS NOT NULL
      AND g.home_ml_price IS NOT NULL
      AND g.away_ml_price IS NOT NULL
""")

with engine.connect() as conn:
    df = pd.read_sql(game_features_query, conn)

df["date"] = pd.to_datetime(df["date"])
df = df[(df["date"].dt.month >= 5) & (df["season_year"] >= 2022)].copy()

print(f"Loaded: {len(df)} games | Seasons: {sorted(df['season_year'].unique())}")
per_season = df.groupby("season_year").size()
print(f"Per season:\n{per_season.to_string()}")

# Target
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

# Market-implied probability (vig-removed)
raw_home = 1.0 / df["home_ml_price"]
raw_away = 1.0 / df["away_ml_price"]
df["home_implied_prob"] = raw_home / (raw_home + raw_away)

# Logit transform — natural space for logistic regression; avoids probability ceiling effects
df["logit_market"] = np.log(
    df["home_implied_prob"].clip(1e-6, 1 - 1e-6) /
    (1 - df["home_implied_prob"].clip(1e-6, 1 - 1e-6))
)

# Pitcher ERA trend: positive = pitcher getting worse recently vs season average
df["home_sp_era_trend"] = df["home_sp_last3_era"] - df["home_sp_era"]
df["away_sp_era_trend"] = df["away_sp_last3_era"] - df["away_sp_era"]

# Season win rate (rate vs raw counts — less correlated with schedule length)
total_home = (df["home_team_wins"] + df["home_team_losses"]).replace(0, np.nan)
total_away = (df["away_team_wins"] + df["away_team_losses"]).replace(0, np.nan)
df["home_season_win_rate"] = df["home_team_wins"] / total_home
df["away_season_win_rate"] = df["away_team_wins"] / total_away

# Momentum: recent win rate vs season win rate (positive = team trending up)
df["home_momentum"] = df["home_team_wins_last10"] / 10.0 - df["home_season_win_rate"]
df["away_momentum"] = df["away_team_wins_last10"] / 10.0 - df["away_season_win_rate"]

# Standard differentials
df["sp_era_diff"]     = df["home_sp_era"]    - df["away_sp_era"]
df["lineup_ops_diff"] = df["home_lineup_ops"] - df["away_lineup_ops"]

# ---------- Feature lists (must match feature_extractor output exactly) ----------
# Strategy: use ONLY logit_market (the market anchor) + features the market may underweight.
# Season-long ERA/WHIP/team records/run_diff are already baked into the line — drop them.
# Residual edge comes from: recent pitcher trend, team momentum, handedness matchup, bullpen.
categorical = ["home_throwing_hand", "away_throwing_hand"]
numerical = [
    "logit_market",
    "home_sp_era_trend",    "away_sp_era_trend",
    "home_momentum",        "away_momentum",
    "home_bullpen_era",     "away_bullpen_era",
    "home_avg_runs_vs_arm", "away_avg_runs_vs_arm",
    "home_avg_runs_lastx_total", "away_avg_runs_lastx_total",
]

X = df[categorical + numerical]
y = df["home_win"]

# ---------- Temporal train/test split ----------
season   = df["season_year"]
date_col = df["date"]

if args.mode == "eval":
    train_mask = (season < 2025) | ((season == 2025) & (date_col < EVAL_CUTOFF))
    test_mask  = (season == 2025) & (date_col >= EVAL_CUTOFF)
else:
    train_mask = season <= 2026
    test_mask  = pd.Series(False, index=df.index)

X_train_raw = X[train_mask].reset_index(drop=True)
y_train     = y[train_mask].reset_index(drop=True)
X_test_raw  = X[test_mask].reset_index(drop=True)
y_test      = y[test_mask].reset_index(drop=True)

print(f"\nMode: {args.mode} | Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

# ---------- Sample weights (365-day half-life) ----------
train_dates    = date_col[train_mask].reset_index(drop=True)
days_since     = (train_dates.max() - train_dates).dt.days
sample_weights = np.exp(-days_since / 365)

# ---------- Preprocessing ----------
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]), numerical),
])

# Three-way time-ordered split: fit (75%) | early-stop (15%) | platt calibration (10%)
n_calib = max(int(len(X_train_raw) * 0.10), 1)
n_es    = max(int(len(X_train_raw) * 0.15), 1)
X_fit_raw   = X_train_raw.iloc[:-(n_es + n_calib)]
y_fit       = y_train.iloc[:-(n_es + n_calib)]
w_fit       = sample_weights[:-(n_es + n_calib)]
X_es_raw    = X_train_raw.iloc[-(n_es + n_calib):-n_calib]
y_es        = y_train.iloc[-(n_es + n_calib):-n_calib]
X_calib_raw = X_train_raw.iloc[-n_calib:]
y_calib     = y_train.iloc[-n_calib:]

preprocessor.fit(X_train_raw)
X_fit_enc = preprocessor.transform(X_fit_raw)
X_es_enc  = preprocessor.transform(X_es_raw)

# ---------- Logistic Regression ----------
logreg = LogisticRegression(max_iter=5000, C=1.0, random_state=42)
logreg_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", logreg),
])
logreg_pipeline.fit(X_train_raw, y_train, model__sample_weight=sample_weights)

# ---------- XGBoost ----------
# eval_metric="auc" optimizes ranking directly; Platt calibration fixes log loss separately.
# More regularization (min_child_weight, reg_alpha) to reduce train/test gap.
xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=3,
    min_child_weight=10,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=0.5,
    reg_lambda=2.0,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
)
xgb_clf.fit(
    X_fit_enc, y_fit,
    sample_weight=w_fit,
    eval_set=[(X_es_enc, y_es)],
    verbose=False,
)
print(f"XGBoost best round: {xgb_clf.best_iteration} / 500")

# Uncalibrated pipeline for comparison
xgb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_clf),
])

# Platt calibration: fit a logistic regression on logit(raw_prob) from the held-out set.
X_calib_enc = preprocessor.transform(X_calib_raw)
raw_calib    = xgb_clf.predict_proba(X_calib_enc)[:, 1]
logit_calib  = np.log(raw_calib.clip(1e-9, 1 - 1e-9) / (1 - raw_calib.clip(1e-9, 1 - 1e-9)))
platt        = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
platt.fit(logit_calib.reshape(-1, 1), y_calib)

calibrated_xgb = PlattCalibratedPipeline(preprocessor, xgb_clf, platt)


# ---------- Evaluation ----------
def print_metrics(name, y_true, y_prob):
    pred  = (y_prob > 0.5).astype(int)
    acc   = accuracy_score(y_true, pred)
    auc   = roc_auc_score(y_true, y_prob)
    ll    = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    print(f"  {name:<35} acc={acc:.4f}  auc={auc:.4f}  logloss={ll:.4f}  brier={brier:.4f}")
    return acc, auc, ll, brier


print("\n--- Train set ---")
print_metrics("LogReg",           y_train, logreg_pipeline.predict_proba(X_train_raw)[:, 1])
print_metrics("XGBoost (raw)",    y_train, xgb_pipeline.predict_proba(X_train_raw)[:, 1])
print_metrics("XGBoost (calib)",  y_train, calibrated_xgb.predict_proba(X_train_raw)[:, 1])

lr_test_metrics    = None
xgb_test_metrics   = None
calib_test_metrics = None

if len(X_test_raw) > 0:
    lr_prob      = logreg_pipeline.predict_proba(X_test_raw)[:, 1]
    xgb_prob     = xgb_pipeline.predict_proba(X_test_raw)[:, 1]
    calib_prob   = calibrated_xgb.predict_proba(X_test_raw)[:, 1]
    implied_prob = df.loc[test_mask, "home_implied_prob"].reset_index(drop=True)
    logit_prob   = df.loc[test_mask, "logit_market"].reset_index(drop=True)
    # market in probability space for fair comparison
    market_prob  = implied_prob

    print("\n--- Test set (H2-2025 holdout) ---")
    lr_test_metrics    = print_metrics("LogReg",                       y_test, lr_prob)
    xgb_test_metrics   = print_metrics("XGBoost (raw)",                y_test, xgb_prob)
    calib_test_metrics = print_metrics("XGBoost (Platt calibrated)",   y_test, calib_prob)
    print_metrics("Market baseline (implied prob)",                     y_test, market_prob)

    # Direction agreement: calibrated model vs market
    model_home  = calib_prob > 0.5
    market_home = market_prob > 0.5
    agreement = (model_home == market_home).mean()
    print(f"\n  Calibrated XGBoost vs market direction agreement: {agreement:.3f}")

    # XGBoost feature importances
    cat_names  = list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical))
    feat_names = cat_names + numerical
    importances = xgb_clf.feature_importances_
    fi = sorted(zip(feat_names, importances), key=lambda x: -x[1])
    print("\n--- XGBoost Feature Importances (top 15) ---")
    for fname, imp in fi[:15]:
        print(f"  {fname:<40} {imp:.4f}")


# ---------- Save ----------
# Primary active model: calibrated XGBoost (best AUC + calibrated log loss)
joblib.dump(calibrated_xgb,  "app/models/moneyline_xgb.pkl")
joblib.dump(logreg_pipeline, "app/models/moneyline_logreg.pkl")
print("\nSaved: app/models/moneyline_xgb.pkl  (calibrated — use this)")
print("Saved: app/models/moneyline_logreg.pkl")

meta = {
    "mode": args.mode,
    "trained_date": str(pd.Timestamp.today().date()),
    "train_games": int(len(X_train_raw)),
    "test_games": int(len(X_test_raw)),
    "eval_cutoff": str(EVAL_CUTOFF.date()),
    "features_categorical": categorical,
    "features_numerical": numerical,
    "xgb_best_round": int(xgb_clf.best_iteration),
    "calibration": "platt_sigmoid",
}
if lr_test_metrics:
    acc, auc, ll, brier = lr_test_metrics
    meta["logreg_test"] = {"accuracy": acc, "roc_auc": auc, "log_loss": ll, "brier": brier}
if xgb_test_metrics:
    acc, auc, ll, brier = xgb_test_metrics
    meta["xgb_raw_test"] = {"accuracy": acc, "roc_auc": auc, "log_loss": ll, "brier": brier}
if calib_test_metrics:
    acc, auc, ll, brier = calib_test_metrics
    meta["xgb_calibrated_test"] = {"accuracy": acc, "roc_auc": auc, "log_loss": ll, "brier": brier}

with open("app/models/moneyline_model_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print("Saved: app/models/moneyline_model_metadata.json")
