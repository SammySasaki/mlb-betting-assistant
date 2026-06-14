package scraper

import (
	"log"
	"strings"
	"time"

	"github.com/mmcdole/gofeed"
)

func ScrapeMLB() []Article {
	fp := gofeed.NewParser()
	feed, err := fp.ParseURL("https://www.mlb.com/feeds/news/rss.xml")
	if err != nil {
		log.Printf("mlb scrape error: %v", err)
		return nil
	}

	log.Printf("mlb: fetched %d items", len(feed.Items))

	articles := []Article{}
	for _, item := range feed.Items {
		publishedAt := ""
		if item.PublishedParsed != nil {
			publishedAt = item.PublishedParsed.Format(time.RFC3339)
		}
		articles = append(articles, Article{
			Source:      "mlb_com",
			Headline:    item.Title,
			Body:        item.Description,
			URL:         item.Link,
			PublishedAt: publishedAt,
			Category:    categorize(item.Title + " " + item.Description),
		})
	}
	return articles
}

func categorize(text string) string {
	t := strings.ToLower(text)
	switch {
	case containsAny(t, "scratch", "injured", "injury", "il ", " il,", "disabled list", "placed on", "day-to-day", "dtd"):
		return "INJURY"
	case containsAny(t, "lineup", "starting lineup", "batting order"):
		return "LINEUP"
	case containsAny(t, "trade", "dfa", "designat", "sign", "waiver", "release", "acquire"):
		return "TRANSACTION"
	default:
		return "GENERAL"
	}
}

func containsAny(text string, keywords ...string) bool {
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			return true
		}
	}
	return false
}
