package main

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/sports-bet/go-news/scraper"
)

func main() {
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/scrape", scrapeHandler)

	log.Println("go-news listening on :8092")
	if err := http.ListenAndServe(":8092", nil); err != nil {
		log.Fatal(err)
	}
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

// scrapeHandler fans out to all three scrapers concurrently using goroutines,
// collects results via a channel, then encodes the combined list as JSON.
//
// Goroutine pattern used here:
//  1. Create a buffered channel large enough to hold one result per goroutine.
//  2. Launch each scraper in its own goroutine; each sends its []Article into the channel.
//  3. Use sync.WaitGroup so we know when all goroutines are done.
//  4. Close the channel once the WaitGroup is done (in a separate goroutine).
//  5. Range over the closed channel to collect all results.
func scrapeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	// Buffered channel — one slot per scraper so goroutines never block on send.
	results := make(chan []scraper.Article, 3)

	var wg sync.WaitGroup
	wg.Add(3)

	go func() {
		defer wg.Done()
		results <- scraper.ScrapeRotowire()
	}()
	go func() {
		defer wg.Done()
		results <- scraper.ScrapeMLB()
	}()
	go func() {
		defer wg.Done()
		results <- scraper.ScrapeESPN()
	}()

	// Close the channel once all goroutines finish so the range below terminates.
	go func() {
		wg.Wait()
		close(results)
	}()

	var all []scraper.Article
	sourcesScraped := 0
	for batch := range results {
		if batch != nil {
			all = append(all, batch...)
			sourcesScraped++
		}
	}

	resp := scraper.ScrapeResult{
		Articles:         all,
		SourcesScraped:   sourcesScraped,
		TotalArticles:    len(all),
		ScrapeDurationMs: time.Since(start).Milliseconds(),
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("encode error: %v", err)
	}
}
