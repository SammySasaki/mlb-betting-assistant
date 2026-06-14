package scraper

// Article is the normalized news item returned by every scraper.
// All scrapers must produce this shape — the HTTP handler encodes a slice of these as JSON.
type Article struct {
	Source      string `json:"source"`       // "rotowire" | "mlb_com" | "espn"
	Headline    string `json:"headline"`
	Body        string `json:"body"`
	URL         string `json:"url"`
	PublishedAt string `json:"published_at"` // RFC3339 / ISO8601, empty string if unknown
	Category    string `json:"category"`     // "INJURY" | "LINEUP" | "TRANSACTION" | "GENERAL"
	PlayerName  string `json:"player_name"`  // extracted from Rotowire's "{Player} News: {headline}" format
}

// ScrapeResult is the envelope the HTTP handler sends back to Python.
type ScrapeResult struct {
	Articles       []Article `json:"articles"`
	SourcesScraped int       `json:"sources_scraped"`
	TotalArticles  int       `json:"total_articles"`
	ScrapeDurationMs int64   `json:"scrape_duration_ms"`
}
