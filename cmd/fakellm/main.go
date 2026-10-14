// Command fakellm is a small OpenAI-compatible chat server for tests and demos. It behaves like an
// inference server with a hard context limit (like vLLM's --max-model-len): when prompt tokens plus
// the requested completion exceed the limit it answers 400 with a vLLM-style message.
//
// Token counting uses its own approximation (trim.ClassEstimator with 5 letters per token plus a
// 6-token chat-template cost per message). It is not the real tokenizer; the demo checks the requests
// fakellm received against the real Qwen3 tokenizer separately (tools/realcount.py).
//
// Environment:
//
//	FAKE_LISTEN          127.0.0.1:18491
//	FAKE_MAX_MODEL_LEN   8192
//	FAKE_MODEL           fast
//	FAKE_TTFT_MS         time to first token (default 50)
//	FAKE_TOKEN_MS        delay per streamed token (default 5)
//	FAKE_REPLY_TOKENS    words in each reply, capped by max_tokens (default 48)
//	FAKE_LOG             if set, append every accepted request as one JSON line
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

type msg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type chatReq struct {
	Model               string `json:"model"`
	Messages            []msg  `json:"messages"`
	MaxTokens           *int   `json:"max_tokens"`
	MaxCompletionTokens *int   `json:"max_completion_tokens"`
	Stream              bool   `json:"stream"`
}

var (
	maxLen     = envInt("FAKE_MAX_MODEL_LEN", 8192)
	model      = envStr("FAKE_MODEL", "fast")
	ttft       = time.Duration(envInt("FAKE_TTFT_MS", 50)) * time.Millisecond
	perTok     = time.Duration(envInt("FAKE_TOKEN_MS", 5)) * time.Millisecond
	replyWords = envInt("FAKE_REPLY_TOKENS", 48)
	counter    = trim.ClassEstimator{LetterRun: envInt("FAKE_LETTER_RUN", 5)}
	logMu      sync.Mutex
	logFile    *os.File
	seq        atomic.Int64
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

func text(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct{ Type, Text string }
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return string(raw)
}

func promptTokens(ms []msg) int {
	n := 3
	for _, m := range ms {
		n += 6 + counter.Text(text(m.Content))
	}
	return n
}

func apiError(w http.ResponseWriter, status int, m string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "error", "message": m, "type": "BadRequestError", "param": nil, "code": status})
}

func chat(w http.ResponseWriter, r *http.Request) {
	var req chatReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Messages) == 0 {
		apiError(w, 400, "invalid request: messages required")
		return
	}
	p := promptTokens(req.Messages)
	completion := maxLen - p
	if req.MaxCompletionTokens != nil {
		completion = *req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		completion = *req.MaxTokens
	}
	if p >= maxLen || p+completion > maxLen {
		apiError(w, 400, fmt.Sprintf("This model's maximum context length is %d tokens. However, you requested %d tokens (%d in the messages, %d in the completion). Please reduce the length of the messages or completion.", maxLen, p+completion, p, completion))
		return
	}
	n := seq.Add(1)
	if logFile != nil {
		logMu.Lock()
		_ = json.NewEncoder(logFile).Encode(map[string]any{"seq": n, "prompt_tokens_fake": p, "max_tokens": completion, "messages": req.Messages})
		logMu.Unlock()
	}
	words := replyWords
	if completion < words {
		words = completion
	}
	reply := make([]string, words)
	for i := range reply {
		reply[i] = fmt.Sprintf("w%d", i%10) // short tokens: "w3" etc.
	}
	reply[0] = fmt.Sprintf("Reply %d with %d prompt messages.", n, len(req.Messages))
	id := fmt.Sprintf("chatcmpl-fake-%d", n)
	created := time.Now().Unix()
	time.Sleep(ttft)
	if !req.Stream {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": id, "object": "chat.completion", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": strings.Join(reply, " ")}}},
			"usage": map[string]int{"prompt_tokens": p, "completion_tokens": words, "total_tokens": p + words},
		})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	fl, _ := w.(http.Flusher)
	send := func(delta map[string]any, finish any) {
		b, _ := json.Marshal(map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}})
		fmt.Fprintf(w, "data: %s\n\n", b)
		if fl != nil {
			fl.Flush()
		}
	}
	send(map[string]any{"role": "assistant", "content": ""}, nil)
	for i, wd := range reply {
		if i > 0 {
			wd = " " + wd
			time.Sleep(perTok)
		}
		send(map[string]any{"content": wd}, nil)
	}
	send(map[string]any{}, "stop")
	fmt.Fprint(w, "data: [DONE]\n\n")
	if fl != nil {
		fl.Flush()
	}
}

func main() {
	if p := os.Getenv("FAKE_LOG"); p != "" {
		f, err := os.OpenFile(p, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			log.Fatal(err)
		}
		logFile = f
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{
			map[string]any{"id": model, "object": "model", "owned_by": "fakellm", "max_model_len": maxLen}}})
	})
	mux.HandleFunc("POST /v1/chat/completions", chat)
	addr := envStr("FAKE_LISTEN", "127.0.0.1:18491")
	log.Printf("fakellm on %s (max_model_len %d, model %s)", addr, maxLen, model)
	log.Fatal((&http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}).ListenAndServe())
}
