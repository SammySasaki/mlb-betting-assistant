package scraper

import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/PuerkitoBio/goquery"
)

const rotowireBase = "https://www.rotowire.com"

func ScrapeRotowire() []Article {
	req, err := http.NewRequest("GET", rotowireBase+"/baseball/news.php", nil)
	if err != nil {
		log.Printf("rotowire request error: %v", err)
		return nil
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		log.Printf("rotowire fetch error: %v", err)
		return nil
	}
	defer resp.Body.Close()

	doc, err := goquery.NewDocumentFromReader(resp.Body)
	if err != nil {
		log.Printf("rotowire parse error: %v", err)
		return nil
	}

	articles := []Article{}

	doc.Find(".news-update").Each(func(i int, s *goquery.Selection) {
		headline := strings.TrimSpace(s.Find("a.news-update__headline").Text())
		if headline == "" {
			return
		}

		playerName := strings.TrimSpace(s.Find("a.news-update__player-link").Text())
		body := strings.TrimSpace(s.Find(".news-update__news").Text())

		publishedAt := ""
		if ts := strings.TrimSpace(s.Find(".news-update__timestamp").Text()); ts != "" {
			if t, err := time.Parse("January 2, 2006", ts); err == nil {
				publishedAt = t.Format(time.RFC3339)
			}
		}

		href, _ := s.Find("a.news-update__headline").Attr("href")
		url := href
		if url != "" && !strings.HasPrefix(url, "http") {
			url = rotowireBase + url
		}
		if url == "" {
			return
		}

		if playerName != "" {
			headline = playerName + ": " + headline
		}

		articles = append(articles, Article{
			Source:      "rotowire",
			Headline:    headline,
			Body:        body,
			URL:         url,
			PublishedAt: publishedAt,
			Category:    categorize(headline + " " + body),
			PlayerName:  playerName,
		})
	})

	log.Printf("rotowire: fetched %d items", len(articles))
	return articles
}
