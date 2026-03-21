"""
app/services/season_stats_service.py

Builds and maintains the player_season_stats table.

Two entry points:
  - calculate_season_stats(season_year)  — full backfill for an entire season
  - update_for_date(target_date)         — incremental update for players who
                                           played on target_date (used nightly
                                           after ingestion)
"""

import logging
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.models import PlayerSeasonStats
from app.interfaces.iplayer_season_stats_repository import IPlayerSeasonStatsRepository
from app.services.stats_calculator import StatsCalculator

logger = logging.getLogger(__name__)

# SQL that aggregates a full season's raw stats for a set of players.
# :season_year and :player_ids are bound at call time.
_SEASON_AGG_SQL = text("""
    SELECT
        pgs.player_id,
        COUNT(DISTINCT pgs.game_id)                                        AS games,
        COALESCE(SUM(pgs.at_bats), 0)                                      AS at_bats,
        COALESCE(SUM(pgs.hits), 0)                                         AS hits,
        COALESCE(SUM(pgs.doubles), 0)                                      AS doubles,
        COALESCE(SUM(pgs.triples), 0)                                      AS triples,
        COALESCE(SUM(pgs.home_runs), 0)                                    AS home_runs,
        COALESCE(SUM(pgs.rbis), 0)                                         AS rbis,
        COALESCE(SUM(pgs.runs), 0)                                         AS runs,
        COALESCE(SUM(pgs.walks_batting), 0)                                AS walks_batting,
        COALESCE(SUM(CASE WHEN pgs.outs_pitched > 0 THEN 1 ELSE 0 END), 0) AS games_pitched,
        COALESCE(SUM(pgs.outs_pitched), 0)                                 AS outs_pitched,
        COALESCE(SUM(pgs.earned_runs), 0)                                  AS earned_runs,
        COALESCE(SUM(pgs.strikeouts), 0)                                   AS strikeouts,
        COALESCE(SUM(pgs.walks), 0)                                        AS walks_pitched,
        COALESCE(SUM(pgs.hits_allowed), 0)                                 AS hits_allowed
    FROM player_game_stats pgs
    JOIN games g ON g.id = pgs.game_id
    WHERE g.season_year = :season_year
      AND pgs.player_id = ANY(:player_ids)
    GROUP BY pgs.player_id
""")


class SeasonStatsService:

    def __init__(
        self,
        session: Session,
        season_stats_repo: IPlayerSeasonStatsRepository,
    ) -> None:
        self._session = session
        self._repo = season_stats_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_season_stats(self, season_year: int) -> dict:
        """Aggregate and upsert stats for every player in season_year.

        Intended for one-time backfills (see populate_season_stats.py).
        """
        player_ids = self._all_player_ids_for_season(season_year)
        if not player_ids:
            logger.info("No player_game_stats found for season %s", season_year)
            return {"season": season_year, "players_updated": 0}

        count = self._aggregate_and_upsert(season_year, player_ids)
        logger.info("Season %s backfill complete: %d players", season_year, count)
        return {"season": season_year, "players_updated": count}

    def update_for_date(self, target_date: date) -> dict:
        """Re-aggregate season stats for players who played on target_date.

        Called automatically by MLBIngestionService.ingest_day() after all
        game data has been committed.
        """
        season_year = target_date.year
        player_ids = self._player_ids_for_date(target_date)
        if not player_ids:
            return {"date": str(target_date), "players_updated": 0}

        count = self._aggregate_and_upsert(season_year, player_ids)
        logger.info(
            "Season stats updated for %s: %d players refreshed", target_date, count
        )
        return {"date": str(target_date), "players_updated": count}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_player_ids_for_season(self, season_year: int) -> list[int]:
        rows = self._session.execute(
            text("""
                SELECT DISTINCT pgs.player_id
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE g.season_year = :season_year
            """),
            {"season_year": season_year},
        ).fetchall()
        return [r[0] for r in rows]

    def _player_ids_for_date(self, target_date: date) -> list[int]:
        rows = self._session.execute(
            text("""
                SELECT DISTINCT pgs.player_id
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE g.date = :target_date
            """),
            {"target_date": target_date},
        ).fetchall()
        return [r[0] for r in rows]

    def _aggregate_and_upsert(self, season_year: int, player_ids: list[int]) -> int:
        rows = self._session.execute(
            _SEASON_AGG_SQL,
            {"season_year": season_year, "player_ids": player_ids},
        ).mappings().all()

        for row in rows:
            hits     = int(row["hits"])
            doubles  = int(row["doubles"])
            triples  = int(row["triples"])
            hr       = int(row["home_runs"])
            ab       = int(row["at_bats"])
            walks    = int(row["walks_batting"])
            op       = int(row["outs_pitched"])
            er       = int(row["earned_runs"])
            ha       = int(row["hits_allowed"])
            wp       = int(row["walks_pitched"])

            stats = PlayerSeasonStats(
                player_id   = row["player_id"],
                season_year = season_year,
                games         = int(row["games"]),
                at_bats       = ab,
                hits          = hits,
                doubles       = doubles,
                triples       = triples,
                home_runs     = hr,
                rbis          = int(row["rbis"]),
                runs          = int(row["runs"]),
                walks_batting = walks,
                avg = StatsCalculator.batting_avg(hits, ab),
                obp = StatsCalculator.obp(hits, walks, ab),
                slg = StatsCalculator.slg(hits, doubles, triples, hr, ab),
                ops = StatsCalculator.ops(hits, doubles, triples, hr, ab, walks),
                games_pitched = int(row["games_pitched"]),
                outs_pitched  = op,
                earned_runs   = er,
                strikeouts    = int(row["strikeouts"]),
                walks_pitched = wp,
                hits_allowed  = ha,
                era  = StatsCalculator.era(er, op),
                whip = StatsCalculator.whip(ha, wp, op),
            )
            self._repo.upsert(stats)

        return len(rows)
