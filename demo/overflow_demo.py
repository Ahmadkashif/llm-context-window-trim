#!/usr/bin/env python3
"""Before/after demo: grow a fresh multi-turn conversation past an 8192-token window with the
official OpenAI Python SDK, against three base URLs:

  before   -> the inference server directly (expected: 400 once the history no longer fits)
  api      -> through ctxtrim (the edge/sidecar path)            (expected: every turn succeeds)
  portal   -> through personashim (persona injected, in-process trim) (expected: every turn succeeds)

Usage: python3 -I demo/overflow_demo.py --before URL --api URL --portal URL [--turns 40]
Prints one line per turn and a JSON summary on the last line.
"""
import argparse, json, random, sys

from openai import OpenAI, BadRequestError

TOPICS = ["solar panels", "sourdough bread", "bicycle maintenance", "tax filing", "home networking",
          "garden irrigation", "car insurance", "learning Spanish", "kitchen renovation", "dog training"]
VERBS = ["compare", "explain", "list the trade-offs of", "estimate the cost of", "plan", "troubleshoot"]
FILLER = ("The household has two adults and three children, a budget of {n} per month, and the house was "
          "built in {y}. Last week we measured {m} units, and the previous reading was {p}. ")


def user_message(rnd, turn):
    topic, verb = rnd.choice(TOPICS), rnd.choice(VERBS)
    body = "".join(FILLER.format(n=rnd.randint(100, 5000), y=rnd.randint(1950, 2024), m=rnd.randint(10, 999),
                                 p=rnd.randint(10, 999)) for _ in range(rnd.randint(8, 14)))
    return f"Turn {turn}: please {verb} {topic}. Context: {body} What should we do first?"


REAL = None  # set from --tokenizer: counts a history the way a Qwen chat template renders it


def real_tokens(history):
    if REAL is None:
        return None
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in history) + "<|im_start|>assistant\n"
    return len(REAL.encode(text, add_special_tokens=False).ids)


def run(base_url, turns, seed, label):
    client = OpenAI(base_url=base_url + "/v1", api_key="sk-demo-not-a-real-key", max_retries=0, timeout=60)
    rnd = random.Random(seed)
    history = [{"role": "system", "content": "You are a practical household assistant. Answer briefly."}]
    first_fail, results, sent_chars = None, [], 0
    for t in range(1, turns + 1):
        history.append({"role": "user", "content": user_message(rnd, t)})
        sent_chars = sum(len(m["content"]) for m in history)
        try:
            raw = client.chat.completions.with_raw_response.create(model="fast", messages=history, max_tokens=512)
            resp = raw.parse()
            dropped = raw.headers.get("x-context-dropped-messages", "0")
            reply = resp.choices[0].message.content
            history.append({"role": "assistant", "content": reply + " " + "Noted. " * 40})
            results.append({"turn": t, "status": 200, "history_messages": len(history), "dropped": int(dropped)})
            print(f"[{label}] turn {t:2d}: 200  history={len(history):3d} msgs, {sent_chars:6d} chars sent, dropped by trim={dropped}")
        except BadRequestError as e:
            msg = str(e.body.get("message") if isinstance(e.body, dict) else e)[:160]
            results.append({"turn": t, "status": 400, "error": msg})
            rt = real_tokens(history)
            print(f"[{label}] turn {t:2d}: 400  history={len(history):3d} msgs, {sent_chars:6d} chars sent"
                  f"{'' if rt is None else f', real tokens {rt} + 512 max_tokens'}: {msg}")
            if first_fail is None:
                first_fail = t
            history.pop()  # the client gives up on this turn
            break
    ok = sum(1 for r in results if r["status"] == 200)
    if REAL is not None:
        print(f"[{label}] full client-side history at the end: {real_tokens(history)} real Qwen3 tokens")
    return {"real_tokens_history_at_end": real_tokens(history),
            "label": label, "base_url": base_url, "turns_attempted": len(results), "turns_ok": ok,
            "first_failure_turn": first_fail, "history_chars_at_end": sent_chars,
            "max_dropped": max([r.get("dropped", 0) for r in results] or [0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--api", required=True)
    ap.add_argument("--portal", required=True)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--tokenizer", help="tokenizer.json of the served model, to report real token counts")
    a = ap.parse_args()
    global REAL
    if a.tokenizer:
        from tokenizers import Tokenizer
        REAL = Tokenizer.from_file(a.tokenizer)
    summary = [run(a.before, a.turns, a.seed, "before"), run(a.api, a.turns, a.seed, "api"), run(a.portal, a.turns, a.seed, "portal")]
    print(json.dumps(summary))
    before, api, portal = summary
    passed = before["first_failure_turn"] is not None and api["turns_ok"] == a.turns and portal["turns_ok"] == a.turns
    print("RESULT:", "PASS" if passed else "FAIL",
          f"(before fails at turn {before['first_failure_turn']}; api {api['turns_ok']}/{a.turns}; portal {portal['turns_ok']}/{a.turns})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
