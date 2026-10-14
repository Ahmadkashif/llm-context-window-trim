package trim

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
)

type m struct {
	Role    string `json:"role"`
	Content any    `json:"content,omitempty"`
}

func body(t *testing.T, extra map[string]any, msgs ...m) []byte {
	t.Helper()
	req := map[string]any{"model": "fast", "messages": msgs}
	for k, v := range extra {
		req[k] = v
	}
	b, err := json.Marshal(req)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func decode(t *testing.T, b []byte) (map[string]json.RawMessage, []m) {
	t.Helper()
	var req map[string]json.RawMessage
	if err := json.Unmarshal(b, &req); err != nil {
		t.Fatal(err)
	}
	var msgs []m
	if err := json.Unmarshal(req["messages"], &msgs); err != nil {
		t.Fatal(err)
	}
	return req, msgs
}

func words(n int) string { return strings.TrimSpace(strings.Repeat("lorem ", n)) }

// conversation builds system + n user/assistant turns + a final user turn.
func conversation(n, wordsPerMsg int) []m {
	out := []m{{Role: "system", Content: "You are a helpful assistant."}}
	for i := 0; i < n; i++ {
		out = append(out, m{Role: "user", Content: fmt.Sprintf("question %d %s", i, words(wordsPerMsg))},
			m{Role: "assistant", Content: fmt.Sprintf("answer %d %s", i, words(wordsPerMsg))})
	}
	return append(out, m{Role: "user", Content: "latest question"})
}

func TestFitsUntouchedReturnsOriginalBytes(t *testing.T) {
	b := body(t, map[string]any{"max_tokens": 100}, conversation(2, 10)...)
	res, err := Fit(b, Config{WindowTokens: 8192})
	if err != nil {
		t.Fatal(err)
	}
	if res.Changed || string(res.Body) != string(b) || res.DroppedMessages != 0 {
		t.Fatalf("expected untouched body, got changed=%v dropped=%d", res.Changed, res.DroppedMessages)
	}
}

func TestDropsOldestTurnsKeepsSystemAndLatest(t *testing.T) {
	conv := conversation(40, 200) // far beyond 2048
	b := body(t, map[string]any{"max_tokens": 256, "temperature": 0.2}, conv...)
	cfg := Config{WindowTokens: 2048}
	res, err := Fit(b, cfg)
	if err != nil {
		t.Fatal(err)
	}
	req, msgs := decode(t, res.Body)
	if msgs[0].Role != "system" || msgs[0].Content != "You are a helpful assistant." {
		t.Fatalf("system prompt lost: %+v", msgs[0])
	}
	if last := msgs[len(msgs)-1]; last.Role != "user" || last.Content != "latest question" {
		t.Fatalf("latest user turn lost: %+v", last)
	}
	if msgs[1].Role != "user" {
		t.Fatalf("first kept turn must start with a user message, got %s", msgs[1].Role)
	}
	// The newest history must be the part kept.
	if !strings.HasPrefix(msgs[len(msgs)-2].Content.(string), "answer 39") {
		t.Fatalf("newest assistant answer not kept: %v", msgs[len(msgs)-2].Content)
	}
	if res.EstimatedAfter > res.PromptBudget {
		t.Fatalf("still over budget: %d > %d", res.EstimatedAfter, res.PromptBudget)
	}
	if res.DroppedMessages == 0 || res.DroppedMessages%2 != 0 {
		t.Fatalf("expected whole turns dropped, got %d messages", res.DroppedMessages)
	}
	if string(req["temperature"]) != "0.2" {
		t.Fatalf("unrelated fields must be preserved, got temperature=%s", req["temperature"])
	}
}

func TestKeepsMidConversationSystemMessages(t *testing.T) {
	conv := conversation(30, 200)
	persona := m{Role: "system", Content: "PERSONA: answer as the house assistant."}
	conv = append(conv[:5], append([]m{persona}, conv[5:]...)...)
	res, err := Fit(body(t, map[string]any{"max_tokens": 256}, conv...), Config{WindowTokens: 2048})
	if err != nil {
		t.Fatal(err)
	}
	_, msgs := decode(t, res.Body)
	found := 0
	for _, x := range msgs {
		if x.Role == "system" {
			found++
		}
	}
	if found != 2 {
		t.Fatalf("expected both system messages kept, found %d", found)
	}
}

func TestNeverSplitsToolResultFromItsTurn(t *testing.T) {
	conv := []m{{Role: "system", Content: "sys"}}
	for i := 0; i < 20; i++ {
		conv = append(conv,
			m{Role: "user", Content: fmt.Sprintf("q%d %s", i, words(150))},
			m{Role: "assistant", Content: "calling tool"},
			m{Role: "tool", Content: fmt.Sprintf("tool result %d %s", i, words(150))},
			m{Role: "assistant", Content: fmt.Sprintf("a%d", i)})
	}
	conv = append(conv, m{Role: "user", Content: "now"})
	res, err := Fit(body(t, map[string]any{"max_tokens": 128}, conv...), Config{WindowTokens: 2048})
	if err != nil {
		t.Fatal(err)
	}
	_, msgs := decode(t, res.Body)
	if msgs[1].Role != "user" {
		t.Fatalf("history must restart at a user message, got %s", msgs[1].Role)
	}
	if res.DroppedMessages%4 != 0 {
		t.Fatalf("dropped %d messages, not whole 4-message turns", res.DroppedMessages)
	}
}

func TestRejectsWhenLatestTurnAloneTooBig(t *testing.T) {
	conv := []m{{Role: "system", Content: "sys"}, {Role: "user", Content: words(5000)}}
	_, err := Fit(body(t, map[string]any{"max_tokens": 512}, conv...), Config{WindowTokens: 2048})
	var tl *TooLargeError
	if !errors.As(err, &tl) {
		t.Fatalf("expected TooLargeError, got %v", err)
	}
	if !strings.Contains(tl.Error(), "context window 2048") {
		t.Fatalf("error should name the window: %s", tl.Error())
	}
}

func TestTruncateModeKeepsTailOfLatestTurn(t *testing.T) {
	long := "START " + words(5000) + " THE-END"
	conv := []m{{Role: "system", Content: "sys"}, {Role: "user", Content: long}}
	res, err := Fit(body(t, map[string]any{"max_tokens": 512}, conv...), Config{WindowTokens: 2048, OnOversize: TruncateLatest})
	if err != nil {
		t.Fatal(err)
	}
	_, msgs := decode(t, res.Body)
	c := msgs[1].Content.(string)
	if !res.Truncated || !strings.HasPrefix(c, TruncationMarker) || !strings.HasSuffix(c, "THE-END") || strings.Contains(c, "START") {
		t.Fatalf("expected tail kept with marker, truncated=%v len=%d", res.Truncated, len(c))
	}
	if res.EstimatedAfter > res.PromptBudget {
		t.Fatalf("over budget after truncation: %d > %d", res.EstimatedAfter, res.PromptBudget)
	}
}

func TestDefaultMaxTokensIsWrittenWhenMissing(t *testing.T) {
	b := body(t, nil, conversation(1, 5)...)
	res, err := Fit(b, Config{WindowTokens: 8192, DefaultMaxTokens: 700, SetMaxTokens: true})
	if err != nil {
		t.Fatal(err)
	}
	req, _ := decode(t, res.Body)
	if string(req["max_tokens"]) != "700" {
		t.Fatalf("max_tokens = %s, want 700", req["max_tokens"])
	}
	// max_completion_tokens is honoured and not overwritten.
	b = body(t, map[string]any{"max_completion_tokens": 300}, conversation(1, 5)...)
	res, _ = Fit(b, Config{WindowTokens: 8192, DefaultMaxTokens: 700, SetMaxTokens: true})
	if res.MaxTokens != 300 || res.Changed {
		t.Fatalf("max_completion_tokens not honoured: %+v", res)
	}
}

func TestRequestedMaxTokensLargerThanWindowRejected(t *testing.T) {
	_, err := Fit(body(t, map[string]any{"max_tokens": 9000}, conversation(1, 5)...), Config{WindowTokens: 8192})
	var tl *TooLargeError
	if !errors.As(err, &tl) {
		t.Fatalf("expected TooLargeError, got %v", err)
	}
}

func TestMultipartContentCounted(t *testing.T) {
	parts := []map[string]any{{"type": "text", "text": words(10)}, {"type": "image_url", "image_url": map[string]string{"url": "data:,"}}}
	b := body(t, map[string]any{"max_tokens": 10}, m{Role: "user", Content: parts})
	res, err := Fit(b, Config{WindowTokens: 8192, NonTextPartCost: 1000})
	if err != nil {
		t.Fatal(err)
	}
	if res.EstimatedBefore < 1000 {
		t.Fatalf("image part not counted: %d", res.EstimatedBefore)
	}
}

func TestBadRequests(t *testing.T) {
	for _, b := range []string{`not json`, `{"messages":[]}`, `{"messages":[{"content":"x"}]}`} {
		if _, err := Fit([]byte(b), Config{WindowTokens: 100}); !errors.Is(err, ErrBadRequest) {
			t.Fatalf("%s: expected ErrBadRequest, got %v", b, err)
		}
	}
}

func TestDisabledWhenNoWindow(t *testing.T) {
	b := []byte(`whatever`)
	res, err := Fit(b, Config{})
	if err != nil || string(res.Body) != "whatever" {
		t.Fatalf("window 0 must pass through: %v", err)
	}
}

func TestEstimatorClasses(t *testing.T) {
	e := ClassEstimator{}
	cases := map[string]int{
		"":           0,
		"hello":      2, // 5 letters / 3
		"hi there":   3, // 1 + ceil(5/3), single space merged into the word
		"12345":      5, // digit per token
		"a  b":       3, // double space is a token
		"{\"k\": 1}": 8, // { " k " : (space before digit) 1 }
		"日本語":        5, // 3 × 1.5, rounded up
		"🚀":          2,
		"привет":     6,
	}
	for s, want := range cases {
		if got := e.Text(s); got != want {
			t.Errorf("Text(%q) = %d, want %d", s, got, want)
		}
	}
}

func FuzzFitNeverExceedsBudgetOrDropsPinned(f *testing.F) {
	f.Add(10, 50, 2048, 256)
	f.Add(1, 3000, 1024, 100)
	f.Fuzz(func(t *testing.T, turns, wpm, window, maxTok int) {
		if turns < 0 || turns > 60 || wpm < 0 || wpm > 400 || window < 64 || window > 32768 || maxTok < 1 || maxTok > window {
			t.Skip()
		}
		conv := conversation(turns, wpm)
		res, err := Fit(body(t, map[string]any{"max_tokens": maxTok}, conv...), Config{WindowTokens: window})
		if err != nil {
			var tl *TooLargeError
			if !errors.As(err, &tl) {
				t.Fatalf("unexpected error %v", err)
			}
			return
		}
		if res.EstimatedAfter > res.PromptBudget {
			t.Fatalf("over budget %d > %d", res.EstimatedAfter, res.PromptBudget)
		}
		_, msgs := decode(t, res.Body)
		if msgs[0].Role != "system" || msgs[len(msgs)-1].Content != "latest question" {
			t.Fatal("pinned message dropped")
		}
	})
}

func benchmarkFit(b *testing.B, turns int) {
	conv := conversation(turns, 150)
	req, _ := json.Marshal(map[string]any{"model": "fast", "messages": conv, "max_tokens": 256})
	b.SetBytes(int64(len(req)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Fit(req, Config{WindowTokens: 8192}); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkFit10Turns(b *testing.B)  { benchmarkFit(b, 10) }
func BenchmarkFit40Turns(b *testing.B)  { benchmarkFit(b, 40) }
func BenchmarkFit160Turns(b *testing.B) { benchmarkFit(b, 160) }
