import re, json


# Venue orientation mapping (degrees from home plate to center field)  
venue_orientations = {
    "Fenway Park": 120,
    "Wrigley Field": 115,
    "Yankee Stadium": 124,
    "Dodger Stadium": 112,
    "Oracle Park": 110,
    "Petco Park": 105,
    "Busch Stadium": 122,
    "Daikin Park": 120,  
    "Minute Maid Park": 120,  
    "PNC Park": 135,
    "Coors Field": 120,
    "Citi Field": 125,
    "T-Mobile Park": 120,
    "Globe Life Field": 125,
    "Rate Field": 135,  
    "Guaranteed Rate Field": 135,  
    "Target Field": 120,
    "Rogers Centre": 110,
    "Truist Park": 120,
    "Comerica Park": 120,
    "Great American Ball Park": 115,
    "Chase Field": 135,
    "Nationals Park": 110,
    "George M. Steinbrenner Field": 130,  
    "Tropicana Field": 130,  
    "Sutter Health Park": 120,         
    "Oakland Coliseum": 120,      
    "loanDepot park": 110,
    "Kauffman Stadium": 120,
    "Progressive Field": 110,
    "Oriole Park at Camden Yards": 110,
    "American Family Field": 135,
    "Angel Stadium": 120,
    "Citizens Bank Park": 115
}

venue_run_factors = {
    "Fenway Park": 1.08,
    "Wrigley Field": 1.01,
    "Yankee Stadium": 1.06,
    "Dodger Stadium": 1.01,
    "Oracle Park": 0.92,
    "Petco Park": 0.91,
    "Busch Stadium": 0.95,
    "Daikin Park": 1.04, 
    "PNC Park": 0.96,
    "Coors Field": 1.34,  # Highest in MLB
    "Citi Field": 0.95,
    "T-Mobile Park": 0.91,
    "Globe Life Field": 0.97,
    "Rate Field": 1.03,
    "Target Field": 0.99,
    "Rogers Centre": 1.05,
    "Truist Park": 1.01,
    "Comerica Park": 0.94,
    "Great American Ball Park": 1.12,
    "Chase Field": 1.04,
    "Nationals Park": 1.00,
    "George M. Steinbrenner Field": 1.00,  # Spring training — assume average
    "Sutter Health Park": 1.00,  # Minor league — assume average
    "loanDepot park": 0.95,
    "Kauffman Stadium": 0.93,
    "Progressive Field": 0.97,
    "Oriole Park at Camden Yards": 1.01,
    "American Family Field": 1.06,
    "Angel Stadium": 0.96,
    "Citizens Bank Park": 1.03
}

venue_utc_offsets = {
    "Fenway Park": -4,  # Boston, MA (EDT)
    "Yankee Stadium": -4,
    "Dodger Stadium": -7,  # Los Angeles, CA (PDT)
    "Wrigley Field": -5,   # Chicago, IL (CDT)
    "Truist Park": -4,     # Atlanta, GA (EDT)
    "Oriole Park at Camden Yards": -4,
    "George M. Steinbrenner Field": -4,
    "Tropicana Field": -4,
    "Rogers Centre": -4,   # Toronto, ON (EDT)
    "Comerica Park": -4,
    "Kauffman Stadium": -5,
    "Angel Stadium": -7,
    "Daikin Park": -5,
    "Minute Maid Park": -5,
    "Target Field": -5,
    "Sutter Health Park": -7,
    "Oakland Coliseum": -7,
    "T-Mobile Park": -7,
    "Progressive Field": -4,
    "Globe Life Field": -5,
    "Rate Field": -5,
    "Guaranteed Rate Field": -5,
    "PNC Park": -4,
    "Citizens Bank Park": -4,
    "Petco Park": -7,
    "Oracle Park": -7,
    "American Family Field": -5,
    "Busch Stadium": -5,
    "Nationals Park": -4,
    "Great American Ball Park": -4,
    "loanDepot park": -4,
    "Chase Field": -7,      # Phoenix does not observe DST
    "Coors Field": -6,      # Denver, CO (MDT)
    "Citi Field": -4
}

def _extract_json(raw_text: str) -> dict:
    # Strip markdown or text around JSON
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in LLM response")
    return json.loads(match.group(0))