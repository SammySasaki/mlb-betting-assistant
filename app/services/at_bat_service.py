import requests
from pybaseball import statcast_batter, playerid_lookup
import pandas as pd
from datetime import date


class AtBatService:

    def get_matchup_stats(
        self,
        player_id: int,
        position: str,
        opponent_id: int,
        limit: int=None,
        season: int=None,
        select_fields=None,
        aggregate=None
    ) -> dict:
        """
        Fetch batter-vs-pitcher stats for a season.
        
        Args:
            player_id: ID of the player
            position: P if pitcher, B if batter
            opponent_id: ID of the opponent
            limit: number of at-bats to query
            season: Year, if none then any year
            select_fields: list or str of requested fields ('hits', 'at_bats', 'avg', 'at_bats_detail', 'events')
            aggregate: 'sum' | 'count' | None
        Returns:
            Dict with aggregated stats or list of at-bats/events
        """
        if isinstance(select_fields, str):
            select_fields = [select_fields]

        # Pull Statcast data
        if season:
            start = f"{season}-03-01"
            end = f"{season}-11-30"
        else:
            start = "2022-03-01"
            end = date.today().strftime("%Y-%m-%d")

        if position == "B":
            batter_id = player_id
            pitcher_id = opponent_id
        elif position == "P":
            batter_id = opponent_id
            pitcher_id = player_id
        else:
            raise ValueError("position must be 'B' (batter) or 'P' (pitcher)")

        df = statcast_batter(start, end, batter_id)

        # Filter to only plate appearances against the specific pitcher
        df_vs_pitcher = df[df["pitcher"] == pitcher_id]
        df_vs_pitcher = df_vs_pitcher[[
            "game_date",
            "events"
        ]]
        df_vs_pitcher_abs = df_vs_pitcher[df_vs_pitcher["events"].notna()]

        # print(df_vs_pitcher_abs.head(5))

        if df_vs_pitcher.empty:
            return {}

        # Compute core stats
        num_at_bats = df_vs_pitcher["events"].notna().sum()
        hits = df_vs_pitcher_abs["events"].isin(["single", "double", "triple", "home_run"]).sum()
        avg = hits / num_at_bats if num_at_bats > 0 else 0.0

        # Build detailed list of ABs
        ab_details = df_vs_pitcher_abs[["game_date", "events"]].to_dict(orient="records")
        if limit:
            ab_details = ab_details[:limit]
        # Response builder
        results = {}

        if "hits" in select_fields:
            results["hits"] = hits if aggregate == "sum" else int(hits > 0)

        if "at_bats" in select_fields:
            results["at_bats"] = num_at_bats

        if "batting_average" in select_fields:
            results["avg"] = avg

        if "at_bats_detail" in select_fields:
            results["at_bats_detail"] = ab_details

        if "events" in select_fields:
            results["events"] = ab_details

        return results