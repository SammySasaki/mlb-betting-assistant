import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import joblib
import xgboost as xgb
from infra.db.init_db import engine


game_features_query = text("""
    WITH
    game_base AS (
        SELECT
            g.id AS game_id,
            g.date,
            g.season_year,
            g.home_team,
            g.away_team,
            g.home_probable_pitcher_id,
            g.away_probable_pitcher_id
        FROM games g
    ),
    ranked_home_ops AS (
        SELECT
            gb.game_id,
            gs.player_id,
            (
                SUM(gs.hits + gs.walks_batting) * 1.0 / NULLIF(SUM(gs.at_bats + gs.walks_batting), 0) +
                SUM(
                    (gs.hits - gs.doubles - gs.triples - gs.home_runs) + 
                    2 * gs.doubles + 
                    3 * gs.triples + 
                    4 * gs.home_runs
                )  * 1.0 / NULLIF(SUM(gs.at_bats), 0)
            ) AS ops,
            ROW_NUMBER() OVER (
                PARTITION BY gb.game_id
                ORDER BY (
                    SUM(gs.hits + gs.walks_batting) * 1.0 / NULLIF(SUM(NULLIF(gs.at_bats + gs.walks_batting, 0)), 0) +
                    SUM(gs.doubles + 2 * gs.triples + 3 * gs.home_runs) * 1.0 / NULLIF(SUM(NULLIF(gs.at_bats, 0)), 0)
                ) DESC NULLS LAST
            ) AS rn
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.home_team = gs.team AND g2.season_year= gb.season_year AND g2.date < gb.date
        GROUP BY gb.game_id, gs.player_id
        HAVING SUM(gs.at_bats) >= 50
    ),
    lineup_ops_home AS (
        SELECT game_id, AVG(ops) AS team_top5_avg_ops
        FROM ranked_home_ops
        WHERE rn <= 5
        GROUP BY game_id
    ),
    ranked_away_ops AS (
        SELECT
            gb.game_id,
            gs.player_id,
            (
                SUM(gs.hits + gs.walks_batting) * 1.0 / NULLIF(SUM(gs.at_bats + gs.walks_batting), 0) +
                SUM(
                    (gs.hits - gs.doubles - gs.triples - gs.home_runs) + 
                    2 * gs.doubles + 
                    3 * gs.triples + 
                    4 * gs.home_runs
                )  * 1.0 / NULLIF(SUM(gs.at_bats), 0)
            ) AS ops,
            ROW_NUMBER() OVER (
                PARTITION BY gb.game_id
                ORDER BY (
                    SUM(gs.hits + gs.walks_batting) * 1.0 / NULLIF(SUM(NULLIF(gs.at_bats + gs.walks_batting, 0)), 0) +
                    SUM(gs.doubles + 2 * gs.triples + 3 * gs.home_runs) * 1.0 / NULLIF(SUM(NULLIF(gs.at_bats, 0)), 0)
                ) DESC NULLS LAST
            ) AS rn
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.away_team = gs.team AND g2.season_year = gb.season_year AND g2.date < gb.date
        GROUP BY gb.game_id, gs.player_id
        HAVING SUM(gs.at_bats) >= 50
    ),
    lineup_ops_away AS (
        SELECT game_id, AVG(ops) AS team_top5_avg_ops
        FROM ranked_away_ops
        WHERE rn <= 5
        GROUP BY game_id
    ),  
    pitcher_handedness AS (
        SELECT
            g.id AS game_id,
            p_home.throwing_hand AS home_throwing_hand,
            p_away.throwing_hand AS away_throwing_hand
        FROM games g
        LEFT JOIN players p_home ON p_home.id = g.home_probable_pitcher_id
        LEFT JOIN players p_away ON p_away.id = g.away_probable_pitcher_id
    ),
    team_runs_vs_arm_home AS (
       SELECT
            gb.game_id,
            AVG(g.home_score) AS home_avg_runs_vs_arm
        FROM game_base gb
        JOIN pitcher_handedness ph ON ph.game_id = gb.game_id
        JOIN games g ON g.home_team = gb.home_team
        JOIN players p ON p.id = g.away_probable_pitcher_id
        WHERE
            g.date < gb.date AND
            g.season_year = gb.season_year AND
            ((ph.away_throwing_hand = 'L' AND p.throwing_hand = 'L') OR
            (ph.away_throwing_hand = 'R' AND p.throwing_hand = 'R'))
        GROUP BY gb.game_id
    ),
    team_runs_vs_arm_away AS (
        SELECT
            gb.game_id,
            AVG(g.away_score) AS away_avg_runs_vs_arm
        FROM game_base gb
        JOIN pitcher_handedness ph ON ph.game_id = gb.game_id
        JOIN games g ON g.away_team = gb.away_team
        JOIN players p ON p.id = g.home_probable_pitcher_id
        WHERE
            g.date < gb.date AND
            g.season_year = gb.season_year AND
            ((ph.home_throwing_hand = 'L' AND p.throwing_hand = 'L') OR
            (ph.home_throwing_hand = 'R' AND p.throwing_hand = 'R'))
        GROUP BY gb.game_id
    )
                           
    SELECT
        g.id AS game_id,
        g.date,
        g.season_year,
        g.home_team,
        g.away_team,
        g.venue,
        g.home_score,
        g.away_score,

        -- Home team features
        th.wins_season as home_wins,
        th.losses_season as home_losses,
        th.wins_last10 AS home_wins_last10,
        th.avg_runs_last10 AS home_avg_runs_last10,
        th.run_diff_season AS home_run_diff_season,
        bh.bullpen_era AS home_bullpen_era,
        loh.team_top5_avg_ops AS home_lineup_ops,
        trh.home_avg_runs_vs_arm AS home_avg_runs_vs_arm,

        -- Away team features
        ta.wins_season as away_wins,
        ta.losses_season as away_losses,
        ta.wins_last10 AS away_wins_last10,
        ta.avg_runs_last10 AS away_avg_runs_last10,
        ta.run_diff_season AS away_run_diff_season,
        ba.bullpen_era AS away_bullpen_era,
        loa.team_top5_avg_ops AS away_lineup_ops,
        tra.away_avg_runs_vs_arm AS away_avg_runs_vs_arm,

        -- Home pitcher features
        ph.sp_era AS home_sp_era,
        ph.sp_whip AS home_sp_whip,
        ph.last3_era AS home_sp_last3_era,

        -- Away pitcher features
        pa.sp_era AS away_sp_era,
        pa.sp_whip AS away_sp_whip,
        pa.last3_era AS away_sp_last3_era

    FROM games g
    LEFT JOIN team_features th
        ON th.game_id = g.id AND th.team = g.home_team
    LEFT JOIN team_features ta
        ON ta.game_id = g.id AND ta.team = g.away_team
    LEFT JOIN bullpen_features bh
        ON bh.game_id = g.id AND bh.team = g.home_team
    LEFT JOIN bullpen_features ba
        ON ba.game_id = g.id AND ba.team = g.away_team
    LEFT JOIN pitcher_features ph
        ON ph.game_id = g.id AND ph.player_id = g.home_probable_pitcher_id
    LEFT JOIN pitcher_features pa
        ON pa.game_id = g.id AND pa.player_id = g.away_probable_pitcher_id
    
    LEFT JOIN lineup_ops_home loh
        ON loh.game_id = g.id
    LEFT JOIN lineup_ops_away loa
        ON loa.game_id = g.id
    LEFT JOIN team_runs_vs_arm_home trh
        ON trh.game_id = g.id
    LEFT JOIN team_runs_vs_arm_away tra
        ON tra.game_id = g.id
    WHERE g.home_score IS NOT NULL
    AND g.away_score IS NOT NULL;
""")


# Step 1: Load data
with engine.connect() as conn:
    df = pd.read_sql(game_features_query, conn)

# ---------- Target ----------
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)

# ---------- Feature lists ----------
categorical = [
            #    "home_throwing_hand", "away_throwing_hand"
               ]
numerical = [
    "home_sp_era", "away_sp_era",
    "home_sp_whip", "away_sp_whip",
    "home_sp_last3_era",
    "away_sp_last3_era",
    "home_bullpen_era", "away_bullpen_era",
    "home_lineup_ops", "away_lineup_ops",
    "home_avg_runs_last10", "away_avg_runs_last10",
    "home_avg_runs_vs_arm", 
    "away_avg_runs_vs_arm", 
    "home_wins", "home_losses",
    "away_wins", "away_losses",
    "home_run_diff_season", "away_run_diff_season",
    "home_wins_last10", "away_wins_last10"
]

# ---------- Create feature differences ----------
df["sp_era_diff"] = df["home_sp_era"] - df["away_sp_era"]
df["lineup_ops_diff"] = df["home_lineup_ops"] - df["away_lineup_ops"]
df["run_diff_diff"] = df["home_run_diff_season"] - df["away_run_diff_season"]

df["date"] = pd.to_datetime(df["date"])
df = df[df["date"].dt.month >= 5]

numerical += ["sp_era_diff", "lineup_ops_diff", "run_diff_diff"]

# ---------- Features & target ----------
X = df[categorical + numerical]
y = df["home_win"]

# ---------- Train/val split ----------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# ---------- Preprocessing ----------
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[ 
        ("cat", categorical_transformer, categorical),
        ("num", numeric_transformer, numerical)
    ]
)

# ---------- Logistic Regression ----------
log_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=5000, random_state=42))
])

log_reg.fit(X_train, y_train)
y_pred_prob = log_reg.predict_proba(X_val)[:, 1]
y_pred = (y_pred_prob > 0.5).astype(int)

print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred_prob))
print("Log Loss:", log_loss(y_val, y_pred_prob))

# ---------- XGBoost ----------
X_train_enc = preprocessor.fit_transform(X_train)
X_val_enc = preprocessor.transform(X_val)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_enc, y_train)

y_pred_prob_xgb = xgb_model.predict_proba(X_val_enc)[:, 1]
y_pred_xgb = (y_pred_prob_xgb > 0.5).astype(int)

print("\nXGBoost:")
print("Accuracy:", accuracy_score(y_val, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_val, y_pred_prob_xgb))
print("Log Loss:", log_loss(y_val, y_pred_prob_xgb))

# ---------- Save models ----------
joblib.dump(log_reg, "app/models/moneyline_logreg2.pkl")
joblib.dump({"preprocessor": preprocessor, "model": xgb_model}, "app/models/moneyline_xgb2.pkl")

print("\nModels saved: moneyline_logreg.pkl, moneyline_xgb.pkl")