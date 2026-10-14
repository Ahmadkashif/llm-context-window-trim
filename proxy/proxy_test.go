package proxy

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

// fakeUpstream rejects prompts whose estimate (5 letters/token, like cmd/fakellm) exceeds limit.
func fakeUpstream(t *testing.T, limit int, hits *atomic.Int64) *httptest.Server {
	est := trim.ClassEstimator{LetterRun: 5}
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		if r.URL.Path != "/v1/chat/completions" {
			fmt.Fprint(w, `{"object":"list","data":[]}`)
			return
		}
		var req struct {
			Messages  []struct{ Role, Content string }
			MaxTokens int  `json:"max_tokens"`
			Stream    bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad json", 400)
			return
		}
		p := 3
		for _, m := range req.Messages {
			p += 6 + est.Text(m.Content)
		}
		if p+req.MaxTokens > limit {
			w.WriteHeader(400)
			fmt.Fprintf(w, `{"message":"This model's maximum context length is %d tokens. However, you requested %d tokens"}`, limit, p+req.MaxTokens)
			return
		}
		first := req.Messages[0].Content
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			for i := 0; i < 3; i++ {
				fmt.Fprintf(w, "data: {\"n\":%d}\n\n", i)
				w.(http.Flusher).Flush()
			}
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"messages": len(req.Messages), "first": first, "prompt": p})
	}))
}

func longChat(turns int, persona bool) map[string]any {
	var msgs []map[string]string
	if persona {
		msgs = append(msgs, map[string]string{"role": "system", "content": "base system"})
	}
	for i := 0; i < turns; i++ {
		msgs = append(msgs,
			map[string]string{"role": "user", "content": fmt.Sprintf("q%d %s", i, strings.Repeat("lorem ipsum dolor ", 60))},
			map[string]string{"role": "assistant", "content": fmt.Sprintf("a%d %s", i, strings.Repeat("sit amet consectetur ", 60))})
	}
	msgs = append(msgs, map[string]string{"role": "user", "content": "final question"})
	return map[string]any{"model": "fast", "messages": msgs, "max_tokens": 512}
}

func post(t *testing.T, url string, body any) (*http.Response, []byte) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp, out
}

func TestBeforeAfterOverflow(t *testing.T) {
	var hits atomic.Int64
	up := fakeUpstream(t, 8192, &hits)
	defer up.Close()

	// Before: straight to the upstream, a long chat overflows.
	resp, out := post(t, up.URL, longChat(60, false))
	if resp.StatusCode != 400 || !strings.Contains(string(out), "maximum context length") {
		t.Fatalf("expected the upstream to overflow first: %d %s", resp.StatusCode, out)
	}

	// After: through the middleware it succeeds.
	rp, _ := NewUpstreamProxy([]string{up.URL})
	st := &Stats{}
	mw := httptest.NewServer(Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192, ReserveTokens: 256}}, st))
	defer mw.Close()
	resp, out = post(t, mw.URL, longChat(60, false))
	if resp.StatusCode != 200 {
		t.Fatalf("expected 200 through the middleware: %d %s", resp.StatusCode, out)
	}
	if resp.Header.Get(HeaderDropped) == "" || resp.Header.Get(HeaderDropped) == "0" {
		t.Fatalf("missing %s header", HeaderDropped)
	}
	if st.Trimmed.Load() != 1 || st.DroppedMessages.Load() == 0 {
		t.Fatalf("stats not updated: %+v", st)
	}
}

func TestOversizeTurnIs400WithoutTouchingUpstream(t *testing.T) {
	var hits atomic.Int64
	up := fakeUpstream(t, 8192, &hits)
	defer up.Close()
	rp, _ := NewUpstreamProxy([]string{up.URL})
	mw := httptest.NewServer(Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192}}, nil))
	defer mw.Close()
	huge := map[string]any{"messages": []map[string]string{{"role": "user", "content": strings.Repeat("word ", 30000)}}, "max_tokens": 100}
	resp, out := post(t, mw.URL, huge)
	if resp.StatusCode != 400 || !strings.Contains(string(out), "context_length_exceeded") {
		t.Fatalf("expected 400 context_length_exceeded: %d %s", resp.StatusCode, out)
	}
	if hits.Load() != 0 {
		t.Fatalf("upstream should not be called, got %d hits", hits.Load())
	}
}

func TestStreamingPassesThrough(t *testing.T) {
	var hits atomic.Int64
	up := fakeUpstream(t, 8192, &hits)
	defer up.Close()
	rp, _ := NewUpstreamProxy([]string{up.URL})
	mw := httptest.NewServer(Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192}}, nil))
	defer mw.Close()
	body := longChat(60, false)
	body["stream"] = true
	b, _ := json.Marshal(body)
	resp, err := http.Post(mw.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(b)))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	n := 0
	for sc.Scan() {
		if strings.HasPrefix(sc.Text(), "data: ") {
			n++
		}
	}
	if resp.StatusCode != 200 || n != 4 || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		t.Fatalf("stream broken: status %d events %d ct %q", resp.StatusCode, n, resp.Header.Get("Content-Type"))
	}
}

func TestOtherRoutesUntouched(t *testing.T) {
	var hits atomic.Int64
	up := fakeUpstream(t, 8192, &hits)
	defer up.Close()
	rp, _ := NewUpstreamProxy([]string{up.URL})
	mw := httptest.NewServer(Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192}}, nil))
	defer mw.Close()
	resp, err := http.Get(mw.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 || hits.Load() != 1 {
		t.Fatalf("GET /v1/models not proxied: %d", resp.StatusCode)
	}
}

func TestRoundRobin(t *testing.T) {
	var a, b atomic.Int64
	ua, ub := fakeUpstream(t, 8192, &a), fakeUpstream(t, 8192, &b)
	defer ua.Close()
	defer ub.Close()
	rp, _ := NewUpstreamProxy([]string{ua.URL, ub.URL})
	mw := httptest.NewServer(Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192}}, nil))
	defer mw.Close()
	for i := 0; i < 10; i++ {
		post(t, mw.URL, longChat(1, false))
	}
	if a.Load() != 5 || b.Load() != 5 {
		t.Fatalf("round robin uneven: %d/%d", a.Load(), b.Load())
	}
}

// injectPersona mimics a portal shim that adds its own system prompt before the trim runs.
func injectPersona(persona string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		msgs := req["messages"].([]any)
		req["messages"] = append([]any{map[string]any{"role": "system", "content": persona}}, msgs...)
		b, _ := json.Marshal(req)
		r.Body = io.NopCloser(strings.NewReader(string(b)))
		r.ContentLength = int64(len(b))
		next.ServeHTTP(w, r)
	})
}

func TestPersonaInjectedBeforeTrimIsKept(t *testing.T) {
	var hits atomic.Int64
	up := fakeUpstream(t, 8192, &hits)
	defer up.Close()
	rp, _ := NewUpstreamProxy([]string{up.URL})
	shim := httptest.NewServer(injectPersona("PERSONA", Middleware(rp, Options{Trim: trim.Config{WindowTokens: 8192, ReserveTokens: 256}}, nil)))
	defer shim.Close()
	resp, out := post(t, shim.URL, longChat(60, true))
	if resp.StatusCode != 200 {
		t.Fatalf("shim path failed: %d %s", resp.StatusCode, out)
	}
	var got struct{ First string }
	_ = json.Unmarshal(out, &got)
	if got.First != "PERSONA" {
		t.Fatalf("persona not first after trim: %q", got.First)
	}
}
