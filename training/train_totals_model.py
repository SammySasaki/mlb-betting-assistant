import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from infra.db.init_db import engine
from app.utils.utils import venue_orientations, venue_run_factors
from sklearn.impute import SimpleImputer


# Query game features
game_features_query = text("""
    WITH game_base AS (
        SELECT
            g.id AS game_id,
            g.date,
            g.start_hour_utc,
            g.season_year,
            g.home_team,
            g.away_team,
            g.venue,
            g.total_runs,
            w.temperature,
            w.wind_speed,
            w.wind_direction,
            w.precipitation
        FROM games g
        LEFT JOIN weather w ON g.id = w.game_id
        WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL AND w.temperature IS NOT NULL
    ),

    starting_pitchers AS (
        SELECT DISTINCT ON (pgs.game_id, pgs.team)
            pgs.game_id,
            g.date AS game_date,
            g.season_year,
            pgs.team,
            pgs.player_id
        FROM player_game_stats pgs
        JOIN games g ON g.id = pgs.game_id
        WHERE pgs.outs_pitched IS NOT NULL
        ORDER BY pgs.game_id, pgs.team, pgs.outs_pitched DESC
    ),

    pitcher_stats AS (
        SELECT
            sp.game_id,
            sp.team,
            sp.player_id,
            SUM(pgs.outs_pitched) / 3.0 AS total_ip,
            SUM(pgs.earned_runs) AS total_er,
            SUM(pgs.strikeouts) AS total_k,
            SUM(pgs.walks) AS total_walks,
            SUM(pgs.hits_allowed) AS total_hits_allowed
        FROM starting_pitchers sp
        JOIN games g ON g.date < sp.game_date AND g.season_year = sp.season_year
        JOIN player_game_stats pgs ON pgs.player_id = sp.player_id AND pgs.game_id = g.id
        GROUP BY sp.game_id, sp.team, sp.player_id
    ),
                           
    pitcher_last3 AS (
        SELECT
            sp.game_id,
            sp.team,
            sp.player_id,
            pgs.game_id AS past_game_id,
            pgs.outs_pitched,
            pgs.earned_runs,
            ROW_NUMBER() OVER (
                PARTITION BY sp.game_id, sp.player_id
                ORDER BY g.date DESC
            ) AS rn
        FROM starting_pitchers sp
        JOIN games g ON g.season_year = sp.season_year AND g.date < sp.game_date
        JOIN player_game_stats pgs ON pgs.player_id = sp.player_id AND pgs.game_id = g.id
        WHERE pgs.outs_pitched IS NOT NULL
    ),

    pitcher_last3_agg AS (
        SELECT
            game_id,
            team,
            player_id,
            SUM(earned_runs) * 9.0 / NULLIF(SUM(outs_pitched) / 3.0, 0) AS last3_era
        FROM pitcher_last3
        WHERE rn <= 3
        GROUP BY game_id, team, player_id
    ),
                        
    pitcher_pregame_stats AS (
        SELECT
            ps.game_id,
            ps.team,
            ps.player_id,
            ps.total_ip,
            ps.total_er * 9.0 / NULLIF(ps.total_ip, 0) AS sp_era,
            (ps.total_walks + ps.total_hits_allowed) / NULLIF(ps.total_ip, 0) AS sp_whip
        FROM pitcher_stats ps
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

    bullpen_stats_home AS (
        SELECT
            gb.game_id,
            SUM(gs.earned_runs) * 9.0 / NULLIF(SUM(gs.outs_pitched) / 3.0, 0) AS bullpen_era
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.home_team = gs.team AND g2.season_year = gb.season_year AND g2.date < gb.date
        LEFT JOIN (
            SELECT DISTINCT ON (pgs.game_id, pgs.team) pgs.game_id, pgs.team, pgs.player_id
            FROM player_game_stats pgs
            ORDER BY pgs.game_id, pgs.team, pgs.outs_pitched DESC
        ) starters ON starters.game_id = gs.game_id AND starters.player_id = gs.player_id
        WHERE starters.player_id IS NULL
        GROUP BY gb.game_id
    ),

    bullpen_stats_away AS (
        SELECT
            gb.game_id,
            SUM(gs.earned_runs) * 9.0 / NULLIF(SUM(gs.outs_pitched) / 3.0, 0) AS bullpen_era
        FROM player_game_stats gs
        JOIN games g2 ON gs.game_id = g2.id
        JOIN game_base gb ON gb.away_team = gs.team AND g2.season_year = gb.season_year AND g2.date < gb.date
        LEFT JOIN (
            SELECT DISTINCT ON (pgs.game_id, pgs.team) pgs.game_id, pgs.team, pgs.player_id
            FROM player_game_stats pgs
            ORDER BY pgs.game_id, pgs.team, pgs.outs_pitched DESC
        ) starters ON starters.game_id = gs.game_id AND starters.player_id = gs.player_id
        WHERE starters.player_id IS NULL
        GROUP BY gb.game_id
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
    ), 
    
    home_avg_runs_lastx_total AS (
        SELECT
            recent.game_id,
            AVG(recent.team_runs) AS home_avg_runs_lastx_total
        FROM (
            SELECT
                gb.game_id,
                CASE
                    WHEN g.home_team = gb.home_team THEN g.home_score
                    WHEN g.away_team = gb.home_team THEN g.away_score
                END AS team_runs,
                ROW_NUMBER() OVER (
                    PARTITION BY gb.game_id
                    ORDER BY g.date DESC
                ) AS rn
            FROM game_base gb
            JOIN games g ON (
                (g.home_team = gb.home_team OR g.away_team = gb.home_team)
                AND g.date < gb.date
                AND g.season_year = gb.season_year
            )
            WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ) recent
        WHERE recent.rn <= 10
        GROUP BY recent.game_id
    ),

    away_avg_runs_lastx_total AS (
        SELECT
            recent.game_id,
            AVG(recent.team_runs) AS away_avg_runs_lastx_total
        FROM (
            SELECT
                gb.game_id,
                CASE
                    WHEN g.home_team = gb.away_team THEN g.home_score
                    WHEN g.away_team = gb.away_team THEN g.away_score
                END AS team_runs,
                ROW_NUMBER() OVER (
                    PARTITION BY gb.game_id
                    ORDER BY g.date DESC
                ) AS rn
            FROM game_base gb
            JOIN games g ON (
                (g.home_team = gb.away_team OR g.away_team = gb.away_team)
                AND g.date < gb.date
                AND g.season_year = gb.season_year
            )
            WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
        ) recent
        WHERE recent.rn <= 10
        GROUP BY recent.game_id
    )
    
    SELECT
        gb.*,

        ph.sp_era AS home_sp_era,
        pa.sp_era AS away_sp_era,
        ph.sp_whip AS home_sp_whip,
        pa.sp_whip AS away_sp_whip,

        bh.bullpen_era AS home_bullpen_era,
        ba.bullpen_era AS away_bullpen_era,

        lh.team_top5_avg_ops AS home_lineup_ops,
        la.team_top5_avg_ops AS away_lineup_ops,

        trvah.home_avg_runs_vs_arm,
        trvaa.away_avg_runs_vs_arm,

        hph.home_throwing_hand,
        hph.away_throwing_hand,
                           
        hlastX.home_avg_runs_lastx_total,
        alastX.away_avg_runs_lastx_total,
                           
        ph3.last3_era AS home_sp_last3_era,
        pa3.last3_era AS away_sp_last3_era

    FROM game_base gb

    LEFT JOIN pitcher_pregame_stats ph ON ph.game_id = gb.game_id AND ph.team = gb.home_team
    LEFT JOIN pitcher_pregame_stats pa ON pa.game_id = gb.game_id AND pa.team = gb.away_team

    LEFT JOIN bullpen_stats_home bh ON bh.game_id = gb.game_id
    LEFT JOIN bullpen_stats_away ba ON ba.game_id = gb.game_id

    LEFT JOIN lineup_ops_home lh ON lh.game_id = gb.game_id
    LEFT JOIN lineup_ops_away la ON la.game_id = gb.game_id

    LEFT JOIN pitcher_handedness hph ON hph.game_id = gb.game_id

    LEFT JOIN team_runs_vs_arm_home trvah ON trvah.game_id = gb.game_id
    LEFT JOIN team_runs_vs_arm_away trvaa ON trvaa.game_id = gb.game_id
                           
    LEFT JOIN home_avg_runs_lastx_total hlastX ON hlastX.game_id = gb.game_id
    LEFT JOIN away_avg_runs_lastx_total alastX ON alastX.game_id = gb.game_id
                           
    LEFT JOIN pitcher_last3_agg ph3 ON ph3.game_id = gb.game_id AND ph3.team = gb.home_team
    LEFT JOIN pitcher_last3_agg pa3 ON pa3.game_id = gb.game_id AND pa3.team = gb.away_team
""")

# Step 1: Load data
with engine.connect() as conn:
    df = pd.read_sql(game_features_query, conn)

df["date"] = pd.to_datetime(df["date"])
print(df.columns.tolist())

numeric_feature_cols = [
    "temperature",
    "wind_flag",
    "venue_orientation_known",
    "home_avg_runs_vs_arm", 
    "away_avg_runs_vs_arm", 
    "away_avg_runs_lastx_total",
    "home_avg_runs_lastx_total",
    "home_sp_era", "away_sp_era",
    # "home_sp_whip", "away_sp_whip",
    # "home_bullpen_era", "away_bullpen_era",
    "home_lineup_ops", "away_lineup_ops",
    "venue_run_factor",
    "home_sp_last3_era",
    "away_sp_last3_era",
    # "home_sp_vs_away_lineup",
    # "away_sp_vs_home_lineup",
    # "home_offense_vs_away_bullpen",
    # "away_offense_vs_home_bullpen"
]

df["field_orientation"] = df["venue"].map(venue_orientations)
df["wind_direction"] = df["wind_direction"].astype(float)

df["wind_rel_angle"] = (df["wind_direction"] - df["field_orientation"]) % 360
def wind_flag_safe(angle):
    if pd.isna(angle):
        return 0        # treat missing orientation as neutral wind
    if (angle >= 315 or angle <= 45):
        return 1 
    elif 135 <= angle <= 225:
        return -1 
    else:
        return 0

df["wind_flag"] = df["wind_rel_angle"].apply(wind_flag_safe)
df["venue_orientation_known"] = df["field_orientation"].notna().astype(int)

df["venue_run_factor"] = df["venue"].map(venue_run_factors)

df["venue_run_factor"] = df["venue"].map(venue_run_factors).fillna(1.0)

# for putting weight on recent games
df["days_since_game"] = (pd.Timestamp.today().normalize() - df["date"]).dt.days
df["sample_weight"] = np.exp(-df["days_since_game"] / 120)


# interaction features:
df["home_sp_vs_away_lineup"] = df["home_sp_era"] - df["away_lineup_ops"]
df["away_sp_vs_home_lineup"] = df["away_sp_era"] - df["home_lineup_ops"]
df["home_offense_vs_away_bullpen"] = df["home_avg_runs_lastx_total"] - df["away_sp_last3_era"]
df["away_offense_vs_home_bullpen"] = df["away_avg_runs_lastx_total"] - df["home_sp_last3_era"]

required_columns = [
    "wind_direction", "venue", "date", "start_hour_utc"
] + numeric_feature_cols

imputer = SimpleImputer(strategy="median")
df[numeric_feature_cols] = imputer.fit_transform(df[numeric_feature_cols])
df = df.dropna(subset=required_columns)

# drop early season games
df = df[df["date"].dt.month >= 5]
print("Final shape:", df.shape)

X = df[numeric_feature_cols]
y = df["total_runs"]
y_log = np.log1p(y)

# Save feature columns for later use
feature_names = X.columns.tolist()
joblib.dump(feature_names, "app/models/feature_names.pkl")


# split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y_log, df["sample_weight"], test_size=0.1, random_state=42
)

X_train_np = np.array(X_train)
y_train_np = np.array(y_train, dtype=np.float32)
w_train_np = np.array(w_train, dtype=np.float32)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test, dtype=np.float32)

# constraints = {
#     "temperature": +1,
#     "wind_speed": 0,
#     "wind_flag": +1,
#     "home_avg_runs_vs_arm": +1,
#     "away_avg_runs_vs_arm": +1,
#     "home_avg_runs_lastx_total": +1,
#     "away_avg_runs_lastx_total": +1,
#     "home_sp_era": +1, "away_sp_era": +1,
#     "home_sp_whip": +1, "away_sp_whip": +1,
#     "home_bullpen_era": +1, "away_bullpen_era": +1,
#     "home_lineup_ops": +1, "away_lineup_ops": +1,
#     "venue_run_factor": +1,
#     "home_sp_vs_away_lineup": +1,
#     "away_sp_vs_home_lineup": +1,
#     "home_offense_vs_away_bullpen": +1,
#     "away_offense_vs_home_bullpen": +1,
#     "home_sp_last3_era": +1,
#     "away_sp_last3_era": +1
# }
# monotone = [constraints[c] for c in numeric_feature_cols]

model = XGBRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    # objective="count:poisson",
    # objective="reg:tweedie",
    # tweedie_variance_power=1.2,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    # monotone_constraints="(" + ",".join(map(str, monotone)) + ")"
) 

eval_set = [(X_train_np, y_train_np), (X_test_np, y_test_np)]

model.fit(
    X_train_np,
    y_train_np,
    sample_weight=w_train_np,
    eval_set=eval_set,
    eval_metric='mae',
    early_stopping_rounds=50,
    verbose=True
)

from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
ridge.fit(X_train, y_train)
print(ridge.score(X_test, y_test))

# Evaluation
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
# y_pred = ridge.predict(X_test)
errors = abs(y_test - y_pred)
# mae = errors.mean()
# std = errors.std()
# r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(np.expm1(y_test), y_pred)
r2 = r2_score(np.expm1(y_test), y_pred)

# Feature importance
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
df["total_runs"].hist(bins=15)
plt.close()

importances = model.feature_importances_
sorted_idx = np.argsort(importances)[-20:]
plt.figure(figsize=(8, 6))
plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
plt.title("Top 20 XGBoost Feature Importances")
plt.tight_layout()
plt.savefig("app/models/importance.png")
plt.close()

# Scatter plot of actual vs predicted (real scale)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)
plt.xlabel("Actual Total Runs")
plt.ylabel("Predicted Total Runs")
plt.title(f"Actual vs. Predicted Total Runs (R² = {r2:.3f})")
plt.tight_layout()
plt.savefig("app/models/ActualvPredicted.png")
plt.close()

print("Sample predictions:", np.round(y_pred[:10], 2))
print("Actual values:", np.round(y_test[:10], 2))

print(f"Validation MAE: {mae:.2f}")
print(f"Validation R²: {r2:.3f}")


joblib.dump(model, "app/models/mlb_xgb_model.pkl")
# joblib.dump(model, "app/models/mlb_ridge.pkl")