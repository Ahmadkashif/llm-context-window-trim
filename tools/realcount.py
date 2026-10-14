#!/usr/bin/env python3
"""Check, with the model's real tokenizer, that every request the inference server received fits.

fakellm (FAKE_LOG=...) writes one JSON line per accepted request. For each, this script counts the
prompt the way a Qwen-style chat template would build it:
    <|im_start|>{role}\\n{content}<|im_end|>\\n   per message, plus <|im_start|>assistant\\n
and checks prompt + max_tokens <= window.

Usage: python3 -I tools/realcount.py --tokenizer tokenizer.json --log fake.jsonl [--window 8192]
"""
import argparse, json, sys

from tokenizers import Tokenizer


def content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return json.dumps(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--window", type=int, default=8192)
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    n, worst, over = 0, 0.0, 0
    worst_line = None
    for line in open(a.log, encoding="utf-8"):
        r = json.loads(line)
        prompt = "".join(f"<|im_start|>{m['role']}\n{content_text(m.get('content'))}<|im_end|>\n" for m in r["messages"])
        prompt += "<|im_start|>assistant\n"
        real = len(tok.encode(prompt, add_special_tokens=False).ids)
        need = real + r["max_tokens"]
        util = need / a.window
        n += 1
        if need > a.window:
            over += 1
        if util > worst:
            worst, worst_line = util, (r["seq"], real, r["max_tokens"], r["prompt_tokens_fake"])
    print(f"requests checked: {n}")
    print(f"over the {a.window}-token window by the REAL tokenizer: {over}")
    if worst_line:
        s, real, mt, fake = worst_line
        print(f"fullest request: seq {s}: real prompt {real} + max_tokens {mt} = {real + mt} "
              f"({worst * 100:.1f}% of window; fakellm's own count was {fake})")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
