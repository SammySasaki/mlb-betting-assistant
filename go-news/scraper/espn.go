package scraper

import (
	"log"
	"net/http"
	"time"

	"github.com/mmcdole/gofeed"
)

func ScrapeESPN() []Article {
	req, err := http.NewRequest("GET", "https://www.espn.com/espn/rss/mlb/news", nil)
	if err != nil {
		log.Printf("espn request error: %v", err)
		return nil
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("espn fetch error: %v", err)
		return nil
	}
	defer resp.Body.Close()

	fp := gofeed.NewParser()
	feed, err := fp.Parse(resp.Body)
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
