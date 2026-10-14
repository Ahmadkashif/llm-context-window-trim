// Command bench drives N concurrent streaming chat requests at a base URL and reports time to first
// token (TTFT) and total latency percentiles. Run it once against the upstream directly and once
// through ctxtrim to measure what the middleware adds.
//
//	bench -url http://127.0.0.1:18491 -c 16 -n 20 -history-words 3000
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

func pct(d []time.Duration, p float64) time.Duration {
	if len(d) == 0 {
		return 0
	}
	i := int(float64(len(d)-1) * p)
	return d[i]
}

func main() {
	url := flag.String("url", "http://127.0.0.1:18491", "base URL")
	c := flag.Int("c", 4, "concurrent streams")
	n := flag.Int("n", 20, "requests per stream")
	hw := flag.Int("history-words", 3000, "words of conversation history in each request")
	label := flag.String("label", "", "label printed with the result")
	flag.Parse()

	// History: turns of ~150 words each, then a final question.
	var msgs []map[string]string
	msgs = append(msgs, map[string]string{"role": "system", "content": "You are a concise assistant."})
	for w, i := 0, 0; w < *hw; w, i = w+300, i+1 {
		msgs = append(msgs,
			map[string]string{"role": "user", "content": fmt.Sprintf("Turn %d: %s", i, strings.Repeat("please explain the scheduler ", 37))},
			map[string]string{"role": "assistant", "content": fmt.Sprintf("Answer %d: %s", i, strings.Repeat("the scheduler batches requests ", 37))})
	}
	msgs = append(msgs, map[string]string{"role": "user", "content": "Summarise in one line."})
	body, _ := json.Marshal(map[string]any{"model": "fast", "messages": msgs, "max_tokens": 256, "stream": true})

	client := &http.Client{Transport: &http.Transport{MaxIdleConnsPerHost: *c * 2}, Timeout: 120 * time.Second}
	var mu sync.Mutex
	var ttfts, totals []time.Duration
	errs := 0
	var wg sync.WaitGroup
	start := time.Now()
	for s := 0; s < *c; s++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < *n; i++ {
				t0 := time.Now()
				resp, err := client.Post(*url+"/v1/chat/completions", "application/json", strings.NewReader(string(body)))
				if err != nil {
					mu.Lock()
					errs++
					mu.Unlock()
					continue
				}
				var ttft time.Duration
				ok := resp.StatusCode == 200
				sc := bufio.NewScanner(resp.Body)
				for sc.Scan() {
					line := sc.Text()
					if ttft == 0 && strings.HasPrefix(line, "data: ") && strings.Contains(line, `"content"`) && !strings.Contains(line, `"content":""`) {
						ttft = time.Since(t0)
					}
				}
				resp.Body.Close()
				total := time.Since(t0)
				mu.Lock()
				if ok && ttft > 0 {
					ttfts = append(ttfts, ttft)
					totals = append(totals, total)
				} else {
					errs++
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	wall := time.Since(start)
	sort.Slice(ttfts, func(i, j int) bool { return ttfts[i] < ttfts[j] })
	sort.Slice(totals, func(i, j int) bool { return totals[i] < totals[j] })
	ms := func(d time.Duration) string { return fmt.Sprintf("%.1f", float64(d.Microseconds())/1000) }
	fmt.Printf("label=%s c=%d ok=%d errors=%d body_bytes=%d wall_s=%.1f ttft_p50_ms=%s ttft_p95_ms=%s total_p50_ms=%s total_p95_ms=%s\n",
		*label, *c, len(ttfts), errs, len(body), wall.Seconds(), ms(pct(ttfts, .5)), ms(pct(ttfts, .95)), ms(pct(totals, .5)), ms(pct(totals, .95)))
	if errs > 0 {
		os.Exit(1)
	}
}
