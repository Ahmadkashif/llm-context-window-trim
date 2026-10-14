// Package proxy wraps trim.Fit as an HTTP middleware for OpenAI-compatible gateways.
//
// Two ways to use it:
//   - as a sidecar (cmd/ctxtrim): an edge proxy routes POST /v1/chat/completions to it, and it forwards
//     to one or more upstreams;
//   - in-process (cmd/personashim): another Go handler calls Middleware or Apply on the body it is
//     about to forward, after it has added its own system prompt.
package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

// Response headers set on trimmed requests (and on the upstream's response).
const (
	HeaderDropped   = "X-Context-Dropped-Messages"
	HeaderEstimated = "X-Context-Estimated-Tokens"
	HeaderTruncated = "X-Context-Truncated"
)

// Options configure the middleware.
type Options struct {
	Trim         trim.Config
	MaxBodyBytes int64 // default 4 MiB
	Logger       *slog.Logger
}

// Stats are process-wide counters, exposed by cmd/ctxtrim on /metrics.
type Stats struct {
	Requests, Trimmed, DroppedMessages, Rejected, Truncated, BadRequests atomic.Int64
	TrimNanos                                                            atomic.Int64
}

// WriteError writes an OpenAI-style error body.
func WriteError(w http.ResponseWriter, status int, typ, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
		"message": msg, "type": typ, "code": code, "param": "messages"}})
}

// Middleware trims POST bodies for chat/completions and passes everything else through unchanged.
func Middleware(next http.Handler, opt Options, st *Stats) http.Handler {
	if opt.MaxBodyBytes <= 0 {
		opt.MaxBodyBytes = 4 << 20
	}
	if opt.Logger == nil {
		opt.Logger = slog.Default()
	}
	if st == nil {
		st = &Stats{}
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
			next.ServeHTTP(w, r)
			return
		}
		st.Requests.Add(1)
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, opt.MaxBodyBytes))
		if err != nil {
			st.BadRequests.Add(1)
			WriteError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request_too_large", "request body too large")
			return
		}
		start := time.Now()
		res, err := trim.Fit(body, opt.Trim)
		st.TrimNanos.Add(int64(time.Since(start)))
		var tl *trim.TooLargeError
		switch {
		case errors.As(err, &tl):
			st.Rejected.Add(1)
			opt.Logger.Info("context too large", "kept_tokens", tl.Kept, "budget", tl.Budget)
			WriteError(w, http.StatusBadRequest, "invalid_request_error", "context_length_exceeded", tl.Error())
			return
		case errors.Is(err, trim.ErrBadRequest):
			// Not something we understand: let the upstream produce its own error.
			st.BadRequests.Add(1)
			res.Body = body
		case err != nil:
			WriteError(w, http.StatusInternalServerError, "server_error", "trim_failed", "context trim failed")
			return
		}
		if res.Changed {
			st.Trimmed.Add(1)
			st.DroppedMessages.Add(int64(res.DroppedMessages))
			if res.Truncated {
				st.Truncated.Add(1)
			}
			w.Header().Set(HeaderDropped, strconv.Itoa(res.DroppedMessages))
			w.Header().Set(HeaderEstimated, strconv.Itoa(res.EstimatedAfter))
			if res.Truncated {
				w.Header().Set(HeaderTruncated, "1")
			}
			if res.DroppedMessages > 0 || res.Truncated {
				opt.Logger.Info("context trimmed", "dropped", res.DroppedMessages, "truncated", res.Truncated,
					"est_before", res.EstimatedBefore, "est_after", res.EstimatedAfter, "budget", res.PromptBudget)
			}
		}
		r.Body = io.NopCloser(bytes.NewReader(res.Body))
		r.ContentLength = int64(len(res.Body))
		r.Header.Set("Content-Length", strconv.Itoa(len(res.Body)))
		next.ServeHTTP(w, r)
	})
}

// NewUpstreamProxy returns a round-robin reverse proxy over upstream base URLs. Streaming responses
// are flushed immediately. A failed upstream answers 502; there is no passive ejection (a client's
// bad request must never take an upstream out of rotation).
func NewUpstreamProxy(upstreams []string) (http.Handler, error) {
	if len(upstreams) == 0 {
		return nil, fmt.Errorf("no upstreams")
	}
	targets := make([]*url.URL, len(upstreams))
	for i, u := range upstreams {
		t, err := url.Parse(u)
		if err != nil || t.Scheme == "" || t.Host == "" {
			return nil, fmt.Errorf("bad upstream %q", u)
		}
		targets[i] = t
	}
	var next atomic.Uint64
	rp := &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			t := targets[next.Add(1)%uint64(len(targets))]
			pr.SetURL(t)
			pr.Out.Host = t.Host
		},
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			WriteError(w, http.StatusBadGateway, "server_error", "upstream_unavailable", "upstream unavailable")
		},
	}
	return rp, nil
}
