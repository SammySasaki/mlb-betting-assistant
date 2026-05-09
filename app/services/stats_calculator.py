"""
app/services/stats_calculator.py

Pure baseball stat calculation functions.

No database dependency — can be used by the stats agent, the season stats
service, or any future browse/leaderboard feature without duplication.
"""


class StatsCalculator:

    @staticmethod
    def batting_avg(hits: int, at_bats: int) -> float:
        return round(hits / at_bats, 3) if at_bats > 0 else 0.0

    # We don't ingest Sac flies or HBP at the moment, results in +- 0.00X variation
    @staticmethod
    def obp(hits: int, walks: int, at_bats: int) -> float:
        denom = at_bats + walks
        return round((hits + walks) / denom, 3) if denom > 0 else 0.0

    @staticmethod
    def slg(hits: int, doubles: int, triples: int, home_runs: int, at_bats: int) -> float:
        singles = hits - doubles - triples - home_runs
        total_bases = singles + 2 * doubles + 3 * triples + 4 * home_runs
        return round(total_bases / at_bats, 3) if at_bats > 0 else 0.0

    @staticmethod
    def ops(hits: int, doubles: int, triples: int, home_runs: int, at_bats: int, walks: int) -> float:
        _obp = StatsCalculator.obp(hits, walks, at_bats)
        _slg = StatsCalculator.slg(hits, doubles, triples, home_runs, at_bats)
        return round(_obp + _slg, 3)

    @staticmethod
    def era(earned_runs: int, outs_pitched: int) -> float:
        ip = outs_pitched / 3
        return round((earned_runs * 9) / ip, 2) if ip > 0 else 0.0

    @staticmethod
    def whip(hits_allowed: int, walks: int, outs_pitched: int) -> float:
        ip = outs_pitched / 3
        return round((hits_allowed + walks) / ip, 3) if ip > 0 else 0.0

    @classmethod
    def from_spec(cls, calculate: str, row) -> dict:
        """Compute a derived stat from a SQLAlchemy Row, accessed by column name."""
        m = row._mapping
        if calculate == "AVG":
            return {"batting_average": cls.batting_avg(hits=m["hits"], at_bats=m["at_bats"])}
        elif calculate == "ERA":
            return {"ERA": cls.era(earned_runs=m["earned_runs"], outs_pitched=m["outs_pitched"])}
        elif calculate == "WHIP":
            return {"WHIP": cls.whip(hits_allowed=m["hits_allowed"], walks=m["walks"], outs_pitched=m["outs_pitched"])}
        elif calculate == "OBP":
            return {"OBP": cls.obp(hits=m["hits"], walks=m["walks_batting"], at_bats=m["at_bats"])}
        elif calculate == "SLG":
            return {"SLG": cls.slg(hits=m["hits"], doubles=m["doubles"], triples=m["triples"], home_runs=m["home_runs"], at_bats=m["at_bats"])}
        elif calculate == "OPS":
            return {"OPS": cls.ops(hits=m["hits"], doubles=m["doubles"], triples=m["triples"], home_runs=m["home_runs"], at_bats=m["at_bats"], walks=m["walks_batting"])}
        else:
            raise ValueError(f"Unsupported calculation: {calculate}")
