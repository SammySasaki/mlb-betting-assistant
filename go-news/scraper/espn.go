package scraper

import (
	"log"
	"time"

	"github.com/mmcdole/gofeed"
)

func ScrapeESPN() []Article {
	fp := gofeed.NewParser()
	feed, err := fp.ParseURL("https://www.espn.com/espn/rss/mlb/news")
	if err != nil {
		log.Printf("espn scrape error: %v", err)
		return nil
	}

	log.Printf("espn: fetched %d items", len(feed.Items))

	articles := []Article{}
	for _, item := range feed.Items {
		publishedAt := ""
		if item.PublishedParsed != nil {
			publishedAt = item.PublishedParsed.Format(time.RFC3339)
		}
		articles = append(articles, Article{
			Source:      "espn",
			Headline:    item.Title,
			Body:        item.Description,
			URL:         item.Link,
			PublishedAt: publishedAt,
			Category:    categorize(item.Title + " " + item.Description),
		})
	}
	return articles
}
