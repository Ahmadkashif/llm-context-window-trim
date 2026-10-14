# llm-context-window-trim

A small, stateless Go middleware that stops long chats from overflowing a self-hosted model's
context window.

Self-hosted inference servers have a hard limit. vLLM started with `--max-model-len 8192` answers
`400 This model's maximum context length is 8192 tokens…` as soon as the history plus `max_tokens`
passes 8192. Chat UIs resend the whole conversation on every turn, so a long chat breaks at some point.
This middleware sits in front of an OpenAI-compatible API. It drops the oldest turns until the request
fits, and keeps:

- every `system` / `developer` message, including a persona prompt that a proxy injected;
- the latest user message and everything after it.

It drops whole turns: a user message plus everything up to the next user message. An assistant reply
and its tool results never lose the question that produced them. If the kept messages alone are too
big, it answers `400 context_length_exceeded` with a clear message (default), or with
`CTXTRIM_ON_OVERSIZE=truncate` it keeps the end of the latest message.

Summaries, server-side memory and retrieval are deliberately out of scope. They are the obvious next
steps, but they need state.

## Two places it runs

```
API clients ──> edge proxy (TLS, auth checks) ──POST /v1/chat/completions──> ctxtrim ──> API replicas ──> inference
                                     └──────── every other /v1/* route ──────────────────> API replicas

chat UI ──> portal shim (adds persona, per-user key) ──[trim in-process]──> API replicas ──> inference
```

A chat portal often does not go through the public edge route. A trim that runs only at the edge would
miss exactly the long conversations it is meant for. So the trim is a library (`trim`, `proxy`), and the
same library runs both as a sidecar (`cmd/ctxtrim`) and inside a shim (`cmd/personashim`, a stand-in for
a portal key shim). The shim injects the persona first and trims second, so the persona is counted and
never dropped.

## How tokens are counted (and the margin)

There is no tokenizer in the request path. `trim.ClassEstimator` gives a deliberately pessimistic count
based on character classes:

| Characters | Cost |
|---|---|
| ASCII digit, punctuation or symbol | 1 token each (Qwen/Llama-family tokenizers split digits one by one) |
| run of ASCII letters | ceil(len / 3) |
| whitespace | 1 token for a run of 2 or more; a single space is free before a letter and costs 1 before anything else |
| non-ASCII rune | 1 (2-byte), 1.5 (3-byte, e.g. CJK and Devanagari), 2 (4-byte, emoji) |
| per message / per request | +8 / +16 (the real Qwen chat-template cost is 5 + role) |

The budget is `window − max_tokens − reserve`. `reserve` defaults to window/32 (256 for 8192). When a
request names no `max_tokens`, ctxtrim writes `CTXTRIM_DEFAULT_MAX_TOKENS` (1024) into it. Without that,
vLLM would default the completion to "whatever is left", and that is exactly the case that overflows.

Measured against the real Qwen3 tokenizer (`tools/calibrate.py`, `evidence/04-calibration.txt`, 79
samples of English prose, Python and Go code, JSON, 10 languages, emoji, numbers and hex):

- **78/79 samples are over-estimated, as intended.** The one miss is a wall of random hex, 2.0% under,
  which the 256-token reserve absorbs. For comparison, a naive `bytes / 3` rule was conservative on only
  69/79 samples, and was 67% short on digit-heavy text.
- **Over-estimation is the price.** The median is +72% and the maximum +217% (on CJK). In the
  end-to-end demo, the fullest request after trimming used **74% of the window by the real tokenizer**.
  Some usable context is given away for safety. If you need it back, count exactly with the inference
  server's own tokenizer endpoint (vLLM has `POST /tokenize`). That costs an extra round trip.

## Run it

Requirements: Go ≥ 1.26 and Python ≥ 3.10 (for the SDK demo and the tokenizer checks). Docker with
Compose v2 is optional, for the edge demo.

```bash
scripts/run-evidence.sh          # tests, before/after demo, real-tokenizer check, calibration, bench → evidence/
scripts/run-compose-demo.sh --down   # same demo through Caddy -> ctxtrim -> 2 replicas in Docker, then removes it
```

`run-evidence.sh` creates `.venv/` (openai, tokenizers) and downloads the Qwen3-4B `tokenizer.json`
(Apache-2.0, pinned by revision and SHA-256) into `.cache/`. Nothing else is installed. Everything
binds 127.0.0.1 on ports 18480–18497. Use `SKIP_PY=1` for a Go-only run.

Using ctxtrim in front of your own API:

```bash
go build -o ctxtrim ./cmd/ctxtrim
CTXTRIM_UPSTREAMS=http://127.0.0.1:4000 CTXTRIM_WINDOW_TOKENS=8192 ./ctxtrim   # listens on 127.0.0.1:18490
```

Then point only `POST /v1/chat/completions` at it from your edge (see `deploy/Caddyfile`). Settings:
`CTXTRIM_DEFAULT_MAX_TOKENS`, `CTXTRIM_SET_MAX_TOKENS`, `CTXTRIM_RESERVE_TOKENS` and
`CTXTRIM_ON_OVERSIZE=reject|truncate`. ctxtrim exposes `/healthz` and Prometheus counters on
`/metrics`. Trimmed responses carry `X-Context-Dropped-Messages` and `X-Context-Estimated-Tokens`.
The middleware never logs prompt content, only counts.

## What was verified (commit `1350fdb`, files in `evidence/`)

| Check | Result |
|---|---|
| `go vet`, `go test -race` (unit + HTTP integration), 20 s fuzz of "never over budget, never drops pinned messages" | **PASS** (`01-go-test.txt`) |
| Before: a fresh 40-turn household conversation through the official OpenAI Python SDK, straight to the 8192-token server | **fails at turn 10** with `400 maximum context length` (`02-overflow-demo.txt`) |
| After, API path (through ctxtrim) | **40/40 turns OK**. The client-side history reached **31,359 real Qwen3 tokens** (3.8× the window); up to 66 messages dropped per request |
| After, portal path (persona shim, in-process trim) | **40/40 turns OK**; persona was the first message in **40/40** forwarded requests |
| Every request the inference server received after trimming, re-counted with the real Qwen3 tokenizer | **0 of 80 over the window**; fullest was 74.4% (`03-real-tokenizer-check.txt`) |
| Estimator vs real tokenizer | 78/79 conservative; worst miss −2.0% (`04-calibration.txt`) |
| Overhead at 4/8/16 concurrent streams, direct vs through ctxtrim (bare processes) | TTFT p50 +3.9 / +4.9 / +0.4 ms; p95 +4.4 / +6.9 / +13.4 ms; 0 errors in 1,120 requests (`05-bench.txt`) |
| Same through the Docker stack, Caddy → ctxtrim → 2 replicas vs Caddy straight to the replicas | TTFT p50 within ±1 ms; p95 +0.3 to +3.9 ms; 0 errors (`06-compose-demo.txt`) |
| Trim CPU cost (Go micro-benchmark) | 1.9 ms per 19 KB request that fits, 8.8 ms per 77 KB request that is trimmed. `encoding/json` is the bulk of it (`07-go-microbench.txt`) |
| Containers | all non-root, read-only, `cap_drop: ALL`. The edge needs `NET_BIND_SERVICE` back because the official Caddy binary has a file capability (`06-compose-demo.txt`) |

For scale: on a single 8 GB GPU serving a 4B model, time to first token at 16 streams is in the
seconds, so the middleware's few milliseconds are well under 1%.

## What was NOT run

- **No real inference server.** The upstream is `cmd/fakellm`, which enforces the limit with its own
  approximate tokenizer. That approximation counts about 3–6% above Qwen3 on the demo text. So the
  "before" run fails at turn 10, when the real request would have needed 8,160 of 8,192 tokens; a real
  server would fail one turn later. The "after" claim does not depend on the fake: every forwarded
  request was re-counted with the real tokenizer.
- **No real API gateway** (for example LiteLLM) between ctxtrim and the model, and no real chat UI. The
  portal path is a stand-in shim.
- **Tokenizers other than Qwen3** were not calibrated. Run `tools/calibrate.py --tokenizer <yours>` before
  relying on the margin for another model family.
- **Images and tool schemas** are only roughly costed: a fixed 1,024 tokens per non-text part, and tool
  definitions in the request (`tools`) are not counted. Requests that use them need a larger reserve.

## Layout

```
trim/            the algorithm + estimator (no I/O)        proxy/   HTTP middleware + round-robin upstream proxy
cmd/ctxtrim      sidecar binary                             cmd/personashim  portal-style shim using the library
cmd/fakellm      fake OpenAI-compatible server, hard limit  cmd/bench        streaming load generator
cmd/estimate     estimator CLI used by tools/calibrate.py   deploy/          Caddyfile + compose demo stack
demo/            OpenAI-SDK before/after demo               tools/           calibrate.py, realcount.py
```

## License

MIT, see `LICENSE`. The Qwen3 tokenizer is downloaded at run time and is not redistributed here (Apache-2.0).
