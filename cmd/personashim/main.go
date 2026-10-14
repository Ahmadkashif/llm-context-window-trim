// Command personashim stands in for a chat-portal "key shim": the hop between a chat UI and the API
// that adds a persona system prompt. It shows the in-process use of the trim middleware: the persona
// is injected FIRST and the trim runs AFTER it, so the persona is counted and always kept.
//
// Portal traffic usually does not pass through the public edge route, so an edge-only trim would
// miss it; that is why the trim must also run here.
//
// Environment:
//
//	SHIM_LISTEN       127.0.0.1:18492
//	SHIM_UPSTREAM     base URL of the API server (or of ctxtrim)
//	SHIM_PERSONA      persona system prompt text
//	CTXTRIM_WINDOW_TOKENS / CTXTRIM_DEFAULT_MAX_TOKENS / CTXTRIM_RESERVE_TOKENS / CTXTRIM_ON_OVERSIZE as in ctxtrim
package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/Ahmadkashif/llm-context-window-trim/proxy"
	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

func envStr(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envInt(k string, d int) int {
	if n, err := strconv.Atoi(os.Getenv(k)); err == nil {
		return n
	}
	return d
}

// injectPersona prepends the persona as the first system message of chat/completions requests.
func injectPersona(persona string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" || persona == "" {
			next.ServeHTTP(w, r)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 4<<20))
		if err != nil {
			proxy.WriteError(w, 413, "invalid_request_error", "request_too_large", "request body too large")
			return
		}
		var req map[string]json.RawMessage
		var msgs []json.RawMessage
		if json.Unmarshal(body, &req) == nil && json.Unmarshal(req["messages"], &msgs) == nil {
			pm, _ := json.Marshal(map[string]string{"role": "system", "content": persona})
			msgs = append([]json.RawMessage{pm}, msgs...)
			req["messages"], _ = json.Marshal(msgs)
			body, _ = json.Marshal(req)
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.ContentLength = int64(len(body))
		r.Header.Set("Content-Length", strconv.Itoa(len(body)))
		next.ServeHTTP(w, r)
	})
}

func main() {
	up, err := proxy.NewUpstreamProxy([]string{envStr("SHIM_UPSTREAM", "http://127.0.0.1:18491")})
	if err != nil {
		log.Fatal(err)
	}
	window := envInt("CTXTRIM_WINDOW_TOKENS", 0)
	opt := proxy.Options{Logger: slog.New(slog.NewJSONHandler(os.Stdout, nil)), Trim: trim.Config{
		WindowTokens: window, DefaultMaxTokens: envInt("CTXTRIM_DEFAULT_MAX_TOKENS", 1024), SetMaxTokens: true,
		ReserveTokens: envInt("CTXTRIM_RESERVE_TOKENS", window/32), OnOversize: trim.Oversize(envStr("CTXTRIM_ON_OVERSIZE", "reject")),
	}}
	h := injectPersona(envStr("SHIM_PERSONA", ""), proxy.Middleware(up, opt, nil))
	addr := envStr("SHIM_LISTEN", "127.0.0.1:18492")
	log.Printf("personashim on %s (window %d)", addr, window)
	log.Fatal((&http.Server{Addr: addr, Handler: h, ReadHeaderTimeout: 10 * time.Second}).ListenAndServe())
}
