// Command ctxtrim is a stateless context-window trimming proxy for OpenAI-compatible chat APIs.
//
// Put it between an edge proxy and the API servers for POST /v1/chat/completions only:
//
//	client -> edge (TLS, auth pre-checks) -> ctxtrim -> API server(s) -> inference server
//
// Configuration is by environment (flags override):
//
//	CTXTRIM_LISTEN              127.0.0.1:18490
//	CTXTRIM_UPSTREAMS           comma-separated base URLs, e.g. http://api-1:4000,http://api-2:4000
//	CTXTRIM_WINDOW_TOKENS       context window of the served model (e.g. vLLM --max-model-len)
//	CTXTRIM_DEFAULT_MAX_TOKENS  completion budget when a request names none (default 1024)
//	CTXTRIM_SET_MAX_TOKENS      write that default into such requests (default true)
//	CTXTRIM_RESERVE_TOKENS      extra safety margin (default window/32)
//	CTXTRIM_ON_OVERSIZE         reject (default) | truncate
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/Ahmadkashif/llm-context-window-trim/proxy"
	"github.com/Ahmadkashif/llm-context-window-trim/trim"
)

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func main() {
	listen := flag.String("listen", env("CTXTRIM_LISTEN", "127.0.0.1:18490"), "listen address")
	ups := flag.String("upstreams", env("CTXTRIM_UPSTREAMS", ""), "comma-separated upstream base URLs")
	window := flag.Int("window", envInt("CTXTRIM_WINDOW_TOKENS", 0), "context window tokens (0 = pass-through)")
	defMax := flag.Int("default-max-tokens", envInt("CTXTRIM_DEFAULT_MAX_TOKENS", 1024), "completion budget when the request names none")
	setMax := flag.Bool("set-max-tokens", env("CTXTRIM_SET_MAX_TOKENS", "true") == "true", "write the default max_tokens into requests that name none")
	reserve := flag.Int("reserve", envInt("CTXTRIM_RESERVE_TOKENS", -1), "safety margin tokens (-1 = window/32)")
	over := flag.String("on-oversize", env("CTXTRIM_ON_OVERSIZE", "reject"), "reject | truncate")
	flag.Parse()

	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if *reserve < 0 {
		*reserve = *window / 32
	}
	if *over != string(trim.Reject) && *over != string(trim.TruncateLatest) {
		log.Error("bad -on-oversize", "value", *over)
		os.Exit(2)
	}
	up, err := proxy.NewUpstreamProxy(splitList(*ups))
	if err != nil {
		log.Error("upstreams", "err", err)
		os.Exit(2)
	}
	st := &proxy.Stats{}
	opt := proxy.Options{Logger: log, Trim: trim.Config{
		WindowTokens: *window, DefaultMaxTokens: *defMax, SetMaxTokens: *setMax,
		ReserveTokens: *reserve, OnOversize: trim.Oversize(*over),
	}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		for _, m := range []struct {
			name string
			v    int64
		}{
			{"ctxtrim_requests_total", st.Requests.Load()},
			{"ctxtrim_trimmed_requests_total", st.Trimmed.Load()},
			{"ctxtrim_dropped_messages_total", st.DroppedMessages.Load()},
			{"ctxtrim_truncated_requests_total", st.Truncated.Load()},
			{"ctxtrim_rejected_requests_total", st.Rejected.Load()},
			{"ctxtrim_unparsed_requests_total", st.BadRequests.Load()},
		} {
			fmt.Fprintf(w, "# TYPE %s counter\n%s %d\n", m.name, m.name, m.v)
		}
		fmt.Fprintf(w, "# TYPE ctxtrim_trim_seconds_total counter\nctxtrim_trim_seconds_total %g\n", float64(st.TrimNanos.Load())/1e9)
	})
	mux.Handle("/", proxy.Middleware(up, opt, st))

	srv := &http.Server{Addr: *listen, Handler: mux, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 60 * time.Second, IdleTimeout: 120 * time.Second}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()
	log.Info("ctxtrim listening", "addr", *listen, "upstreams", *ups, "window", *window, "default_max_tokens", *defMax, "reserve", *reserve, "on_oversize", *over)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Error("serve", "err", err)
		os.Exit(1)
	}
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, strings.TrimRight(p, "/"))
		}
	}
	return out
}
