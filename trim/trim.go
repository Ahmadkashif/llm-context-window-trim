// Package trim fits an OpenAI-style chat/completions request into a fixed context window by dropping
// the oldest conversation turns. It is stateless: everything it needs is in the request body.
//
// What is always kept:
//   - every system and developer message (including a persona prompt injected by a proxy), in place;
//   - the latest user message and everything after it.
//
// What is dropped, oldest first: whole turns before the latest user message. A turn starts at a user
// message and runs up to the next user message, so an assistant reply and its tool results are never
// separated from the question that produced them.
package trim

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"unicode/utf8"
)

// Oversize decides what happens when the kept messages alone do not fit.
type Oversize string

const (
	// Reject returns ErrTooLarge; the proxy answers 400 context_length_exceeded.
	Reject Oversize = "reject"
	// TruncateLatest keeps the end of the latest user message's text and drops its beginning.
	TruncateLatest Oversize = "truncate"
)

// Config holds the per-model budget. WindowTokens normally comes from the hardware profile
// (the same number the inference server is started with, e.g. vLLM --max-model-len).
type Config struct {
	WindowTokens     int      // total context of the served model (prompt + completion)
	DefaultMaxTokens int      // completion budget when the request names none
	SetMaxTokens     bool     // write DefaultMaxTokens into requests that name none
	ReserveTokens    int      // extra safety margin kept free
	PerMessageTokens int      // chat-template cost of one message (role markers etc.)
	PerRequestTokens int      // fixed cost of a request (generation prompt, BOS)
	NonTextPartCost  int      // cost assumed for an image or other non-text content part
	OnOversize       Oversize // Reject (default) or TruncateLatest
	Estimator        Estimator
}

// Defaults fills unset fields with conservative values.
func (c Config) Defaults() Config {
	if c.DefaultMaxTokens <= 0 {
		c.DefaultMaxTokens = 1024
	}
	if c.PerMessageTokens <= 0 {
		c.PerMessageTokens = 8
	}
	if c.PerRequestTokens <= 0 {
		c.PerRequestTokens = 16
	}
	if c.NonTextPartCost <= 0 {
		c.NonTextPartCost = 1024
	}
	if c.ReserveTokens < 0 {
		c.ReserveTokens = 0
	}
	if c.OnOversize == "" {
		c.OnOversize = Reject
	}
	if c.Estimator == nil {
		c.Estimator = ClassEstimator{}
	}
	return c
}

// Result describes what Fit did.
type Result struct {
	Body            []byte // request body to forward (the original bytes when nothing changed)
	Changed         bool
	DroppedMessages int
	Truncated       bool
	EstimatedBefore int // estimated prompt tokens before trimming
	EstimatedAfter  int // estimated prompt tokens after trimming
	MaxTokens       int // completion budget the estimate assumed
	PromptBudget    int // WindowTokens - MaxTokens - ReserveTokens
}

// ErrBadRequest means the body is not a chat/completions request this package understands.
var ErrBadRequest = errors.New("bad request")

// TooLargeError means the messages that must be kept do not fit.
type TooLargeError struct {
	Kept, Budget, Window, MaxTokens int
}

func (e *TooLargeError) Error() string {
	return fmt.Sprintf("the system prompt and latest message need about %d tokens but only %d fit "+
		"(context window %d, %d reserved for the reply); shorten the message or lower max_tokens",
		e.Kept, e.Budget, e.Window, e.MaxTokens)
}

type message struct {
	raw    json.RawMessage
	role   string
	tokens int
}

type parsedMessage struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	Name      string          `json:"name"`
	ToolCalls json.RawMessage `json:"tool_calls"`
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Fit trims body so that estimated prompt tokens + max_tokens + reserve <= WindowTokens.
func Fit(body []byte, cfg Config) (Result, error) {
	cfg = cfg.Defaults()
	if cfg.WindowTokens <= 0 {
		return Result{Body: body}, nil // disabled
	}
	var req map[string]json.RawMessage
	if err := json.Unmarshal(body, &req); err != nil {
		return Result{}, fmt.Errorf("%w: body is not a JSON object", ErrBadRequest)
	}
	var raws []json.RawMessage
	if err := json.Unmarshal(req["messages"], &raws); err != nil || len(raws) == 0 {
		return Result{}, fmt.Errorf("%w: messages must be a non-empty array", ErrBadRequest)
	}
	msgs := make([]message, len(raws))
	for i, r := range raws {
		var pm parsedMessage
		if err := json.Unmarshal(r, &pm); err != nil || pm.Role == "" {
			return Result{}, fmt.Errorf("%w: messages[%d] has no role", ErrBadRequest, i)
		}
		msgs[i] = message{raw: r, role: pm.Role, tokens: cost(pm, cfg)}
	}

	maxTokens, named := requestedMaxTokens(req)
	if !named {
		maxTokens = cfg.DefaultMaxTokens
	}
	budget := cfg.WindowTokens - maxTokens - cfg.ReserveTokens
	res := Result{MaxTokens: maxTokens, PromptBudget: budget}

	total := cfg.PerRequestTokens
	for _, m := range msgs {
		total += m.tokens
	}
	res.EstimatedBefore = total

	// Which messages are pinned, and how the rest group into turns.
	lastUser := -1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].role == "user" {
			lastUser = i
			break
		}
	}
	pinned := make([]bool, len(msgs))
	pinnedTokens := cfg.PerRequestTokens
	for i, m := range msgs {
		if m.role == "system" || m.role == "developer" || lastUser < 0 || i >= lastUser {
			pinned[i] = true
			pinnedTokens += m.tokens
		}
	}

	drop := make([]bool, len(msgs))
	if total > budget {
		// Drop whole turns, oldest first, until it fits.
		cur := total
		for start := 0; start < lastUser && cur > budget; {
			end := start + 1
			for end < lastUser && msgs[end].role != "user" {
				end++
			}
			for i := start; i < end; i++ {
				if !pinned[i] {
					drop[i] = true
					cur -= msgs[i].tokens
					res.DroppedMessages++
				}
			}
			start = end
		}
		total = cur
	}

	if total > budget {
		if cfg.OnOversize != TruncateLatest || lastUser < 0 {
			return res, &TooLargeError{Kept: pinnedTokens, Budget: budget, Window: cfg.WindowTokens, MaxTokens: maxTokens}
		}
		allowed := msgs[lastUser].tokens - (total - budget)
		nr, ok := truncateMessage(msgs[lastUser].raw, allowed, cfg)
		if !ok {
			return res, &TooLargeError{Kept: pinnedTokens, Budget: budget, Window: cfg.WindowTokens, MaxTokens: maxTokens}
		}
		var pm parsedMessage
		_ = json.Unmarshal(nr, &pm)
		newTokens := cost(pm, cfg)
		total += newTokens - msgs[lastUser].tokens
		msgs[lastUser].raw, msgs[lastUser].tokens = nr, newTokens
		res.Truncated = true
	}
	res.EstimatedAfter = total

	setMax := !named && cfg.SetMaxTokens
	if res.DroppedMessages == 0 && !res.Truncated && !setMax {
		res.Body = body
		return res, nil
	}
	// Re-assemble the body from the already-validated raw pieces (no second JSON validation pass).
	var mb bytes.Buffer
	mb.Grow(len(body))
	mb.WriteByte('[')
	first := true
	for i, m := range msgs {
		if drop[i] {
			continue
		}
		if !first {
			mb.WriteByte(',')
		}
		first = false
		mb.Write(m.raw)
	}
	mb.WriteByte(']')
	req["messages"] = mb.Bytes()
	if setMax {
		req["max_tokens"] = json.RawMessage(fmt.Sprint(maxTokens))
	}
	out := writeObject(req, len(body))
	res.Body, res.Changed = out, true
	return res, nil
}

// writeObject serialises a map of raw JSON values without re-validating them. Keys are sorted so
// the output is deterministic.
func writeObject(obj map[string]json.RawMessage, sizeHint int) []byte {
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b bytes.Buffer
	b.Grow(sizeHint)
	b.WriteByte('{')
	for i, k := range keys {
		if i > 0 {
			b.WriteByte(',')
		}
		kb, _ := json.Marshal(k)
		b.Write(kb)
		b.WriteByte(':')
		b.Write(obj[k])
	}
	b.WriteByte('}')
	return b.Bytes()
}

// requestedMaxTokens reads max_completion_tokens (newer clients) or max_tokens.
func requestedMaxTokens(req map[string]json.RawMessage) (int, bool) {
	for _, k := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := req[k]; ok && !bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
			var n int
			if json.Unmarshal(v, &n) == nil && n > 0 {
				return n, true
			}
		}
	}
	return 0, false
}

func cost(pm parsedMessage, cfg Config) int {
	n := cfg.PerMessageTokens + cfg.Estimator.Text(pm.Role) + cfg.Estimator.Text(pm.Name)
	if len(pm.ToolCalls) > 0 && string(pm.ToolCalls) != "null" {
		n += cfg.Estimator.Text(string(pm.ToolCalls))
	}
	if len(pm.Content) == 0 {
		return n
	}
	var s string
	if json.Unmarshal(pm.Content, &s) == nil {
		return n + cfg.Estimator.Text(s)
	}
	var parts []contentPart
	if json.Unmarshal(pm.Content, &parts) == nil {
		for _, p := range parts {
			if p.Type == "text" {
				n += cfg.Estimator.Text(p.Text)
			} else {
				n += cfg.NonTextPartCost
			}
		}
		return n
	}
	return n + cfg.Estimator.Text(string(pm.Content))
}

// TruncationMarker is prepended to a message whose beginning was cut.
const TruncationMarker = "[earlier part of this message removed to fit the context window]\n"

// truncateMessage keeps the tail of a string-content message so that the message costs <= allowed.
func truncateMessage(raw json.RawMessage, allowed int, cfg Config) (json.RawMessage, bool) {
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return nil, false
	}
	var pm parsedMessage
	_ = json.Unmarshal(raw, &pm)
	var text string
	if json.Unmarshal(pm.Content, &text) != nil {
		return nil, false // multi-part content is not truncated
	}
	fixed := cfg.PerMessageTokens + cfg.Estimator.Text(pm.Role) + cfg.Estimator.Text(pm.Name) + cfg.Estimator.Text(TruncationMarker)
	room := allowed - fixed
	if room <= 0 {
		return nil, false
	}
	// Binary search the smallest cut (in bytes, on rune boundaries) whose tail fits.
	lo, hi := 0, len(text)
	for lo < hi {
		mid := (lo + hi) / 2
		for mid < len(text) && !utf8.RuneStart(text[mid]) {
			mid++
		}
		if cfg.Estimator.Text(text[mid:]) <= room {
			hi = mid
		} else {
			lo = mid + 1
		}
	}
	for lo < len(text) && !utf8.RuneStart(text[lo]) {
		lo++
	}
	if lo >= len(text) {
		return nil, false
	}
	nc, _ := json.Marshal(TruncationMarker + text[lo:])
	obj["content"] = nc
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, false
	}
	return out, true
}
