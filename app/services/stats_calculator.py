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
        """Drop-in replacement for StatsAgent.calculate_results().

        Fields in `row` are ordered alphabetically because the agent sorts
        the select list before building the query.
        """
        if calculate == "AVG":
            # Alphabetical: at_bats[0], hits[1]
            return {"batting_average": cls.batting_avg(hits=row[1], at_bats=row[0])}

        elif calculate == "ERA":
            # Alphabetical: earned_runs[0], outs_pitched[1]
            return {"ERA": cls.era(earned_runs=row[0], outs_pitched=row[1])}

        elif calculate == "WHIP":
            # Alphabetical: hits_allowed[0], outs_pitched[1], walks[2]
            return {"WHIP": cls.whip(hits_allowed=row[0], walks=row[2], outs_pitched=row[1])}

        elif calculate == "OBP":
            # Alphabetical: at_bats[0], hits[1], walks_batting[2]
            return {"OBP": cls.obp(hits=row[1], walks=row[2], at_bats=row[0])}

        elif calculate == "SLG":
            # Alphabetical: at_bats[0], doubles[1], hits[2], home_runs[3], triples[4]
            return {"SLG": cls.slg(hits=row[2], doubles=row[1], triples=row[4], home_runs=row[3], at_bats=row[0])}

        elif calculate == "OPS":
            # Alphabetical: at_bats[0], doubles[1], hits[2], home_runs[3], triples[4], walks_batting[5]
            return {"OPS": cls.ops(hits=row[2], doubles=row[1], triples=row[4], home_runs=row[3], at_bats=row[0], walks=row[5])}

        else:
            raise ValueError(f"Unsupported calculation: {calculate}")
