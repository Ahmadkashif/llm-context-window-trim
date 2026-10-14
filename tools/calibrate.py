#!/usr/bin/env python3
"""Measure how conservative the byte-based token estimate is against a real tokenizer.

Usage:
  go build -o bin/estimate ./cmd/estimate
  python3 -I tools/calibrate.py --tokenizer path/to/tokenizer.json --estimator bin/estimate [--letter-run 3]

The corpus is built locally from text that ships with the Python and Go installs on the machine
(stdlib docstrings, stdlib source code), generated JSON, and a fixed set of multilingual and emoji
samples written for this repo. Nothing is downloaded except the tokenizer file you pass in.

For each sample it prints bytes, real tokens, bytes/token and the shipped estimator's estimate
(trim.ClassEstimator, run through cmd/estimate). The estimate is conservative when estimate >= real.
For comparison it also reports how a naive "bytes / 3" rule would have done.
"""
import argparse, json, os, pydoc, random, statistics, subprocess, sys, importlib

from tokenizers import Tokenizer

MULTILINGUAL = {
    "german": "Die Anfrage wurde abgelehnt, weil der Kontext zu lang war. Bitte kürze den Verlauf und versuche es erneut. Größere Modelle brauchen mehr Speicher für den Schlüssel-Wert-Cache.",
    "french": "La requête a été refusée parce que le contexte était trop long. Raccourcissez l'historique et réessayez. Les modèles plus grands ont besoin de davantage de mémoire.",
    "spanish": "La solicitud fue rechazada porque el contexto era demasiado largo. Acorta el historial e inténtalo de nuevo; los modelos grandes necesitan más memoria.",
    "urdu": "درخواست مسترد کر دی گئی کیونکہ سیاق و سباق بہت طویل تھا۔ براہ کرم گفتگو کو مختصر کریں اور دوبارہ کوشش کریں۔",
    "arabic": "تم رفض الطلب لأن السياق كان طويلاً جداً. يرجى تقصير السجل والمحاولة مرة أخرى.",
    "hindi": "अनुरोध अस्वीकार कर दिया गया क्योंकि संदर्भ बहुत लंबा था। कृपया इतिहास छोटा करें और फिर से प्रयास करें।",
    "russian": "Запрос был отклонён, потому что контекст оказался слишком длинным. Сократите историю и попробуйте снова.",
    "chinese": "请求被拒绝，因为上下文太长。请缩短对话历史后重试。更大的模型需要更多的显存来存放键值缓存。",
    "japanese": "コンテキストが長すぎるため、リクエストは拒否されました。履歴を短くしてもう一度お試しください。",
    "korean": "컨텍스트가 너무 길어서 요청이 거부되었습니다. 기록을 줄이고 다시 시도하세요.",
    "emoji": "ok 👍 done ✅ ship it 🚀🚀 thanks 🙏 party 🎉🎉🎉 weather ☀️🌧️ flags 🇵🇰🇬🇧 family 👨‍👩‍👧",
    "numbers": " ".join(str(random.Random(7).randint(0, 10**9)) for _ in range(60)),
    "hex": " ".join(f"{random.Random(i).getrandbits(64):016x}" for i in range(40)),
}


def corpus():
    out = []
    # English prose: stdlib module docstrings.
    for mod in ["json", "http.client", "asyncio", "argparse", "subprocess", "email", "logging",
                "collections", "functools", "itertools", "pathlib", "datetime", "decimal", "unittest"]:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        text = pydoc.getdoc(m)
        for name in dir(m):
            obj = getattr(m, name, None)
            d = pydoc.getdoc(obj) if obj is not None and not name.startswith("_") else ""
            if len(d) > 200:
                text += "\n\n" + d
        for i in range(0, min(len(text), 12000), 3000):
            chunk = text[i:i + 3000]
            if len(chunk) > 500:
                out.append(("prose-en", f"{mod}#{i}", chunk))
    # Python code.
    import inspect
    for mod in ["json.decoder", "http.client", "argparse", "pathlib", "textwrap"]:
        src = inspect.getsource(importlib.import_module(mod))
        for i in range(0, min(len(src), 9000), 3000):
            out.append(("code-py", f"{mod}#{i}", src[i:i + 3000]))
    # Go code from GOROOT, if present.
    try:
        goroot = subprocess.run(["go", "env", "GOROOT"], capture_output=True, text=True, check=True).stdout.strip()
        for rel in ["src/net/http/server.go", "src/encoding/json/decode.go", "src/strings/strings.go"]:
            p = os.path.join(goroot, rel)
            if os.path.exists(p):
                src = open(p, encoding="utf-8").read()
                for i in range(0, 9000, 3000):
                    out.append(("code-go", f"{rel}#{i}", src[i:i + 3000]))
    except Exception:
        pass
    # JSON.
    rnd = random.Random(42)
    for i in range(5):
        obj = [{"id": rnd.randint(1, 10**6), "name": f"item-{rnd.randint(1, 999)}", "price": round(rnd.random() * 100, 2),
                "tags": [rnd.choice(["red", "green", "blue", "large", "small"]) for _ in range(3)],
                "active": rnd.random() > 0.5} for _ in range(25)]
        out.append(("json", f"json#{i}", json.dumps(obj, indent=2)))
    for k, v in MULTILINGUAL.items():
        out.append(("multilingual" if k not in ("emoji", "numbers", "hex") else k, k, (v + " ") * 6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--estimator", required=True, help="path to the built cmd/estimate binary")
    ap.add_argument("--letter-run", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    samples = corpus()
    ests = json.loads(subprocess.run([a.estimator, "-letter-run", str(a.letter_run)], input=json.dumps([s[2] for s in samples]),
                                     capture_output=True, text=True, check=True).stdout)
    rows = []
    for (cat, name, text), est in zip(samples, ests):
        b = len(text.encode("utf-8"))
        t = len(tok.encode(text, add_special_tokens=False).ids)
        rows.append({"category": cat, "sample": name, "bytes": b, "tokens": t, "bytes_per_token": round(b / t, 3),
                     "estimate": est, "conservative": est >= t, "naive_bytes3_ok": b / 3 >= t})
    # Per-message chat-template overhead (Qwen-style <|im_start|>role\n...<|im_end|>\n).
    over = len(tok.encode("<|im_start|>user\n<|im_end|>\n", add_special_tokens=False).ids)
    if a.json:
        print(json.dumps({"rows": rows, "template_overhead_tokens_per_message": over}, indent=1))
        return
    print(f"{'category':14} {'sample':40} {'bytes':>6} {'tokens':>6} {'B/tok':>6} {'est':>6} ok")
    for r in rows:
        print(f"{r['category']:14} {r['sample'][:40]:40} {r['bytes']:6} {r['tokens']:6} {r['bytes_per_token']:6} {r['estimate']:6} {'yes' if r['conservative'] else 'NO'}")
    cats = sorted({r["category"] for r in rows})
    print("\nper category: min / median bytes-per-token")
    for c in cats:
        v = [r["bytes_per_token"] for r in rows if r["category"] == c]
        print(f"  {c:14} min {min(v):5.2f}  median {statistics.median(v):5.2f}  n={len(v)}")
    allv = [r["bytes_per_token"] for r in rows]
    bad = [r for r in rows if not r["conservative"]]
    tot_b = sum(r["bytes"] for r in rows); tot_t = sum(r["tokens"] for r in rows)
    print(f"\nall: n={len(rows)} min {min(allv):.2f} median {statistics.median(allv):.2f} overall {tot_b / tot_t:.2f} bytes/token")
    print(f"ClassEstimator(letter_run={a.letter_run}): {len(rows) - len(bad)}/{len(rows)} samples conservative (estimate >= real)")
    print(f"naive bytes/3 rule, for comparison: {sum(r['naive_bytes3_ok'] for r in rows)}/{len(rows)} conservative")
    for r in bad:
        print(f"  UNDER-estimate: {r['category']} {r['sample']} real {r['tokens']} est {r['estimate']} ({(r['tokens'] - r['estimate']) / r['tokens'] * 100:.1f}% short)")
    over_est = [(r["estimate"] - r["tokens"]) / r["tokens"] * 100 for r in rows if r["conservative"]]
    print(f"over-estimate on conservative samples: median {statistics.median(over_est):.0f}%, max {max(over_est):.0f}%")
    print(f"chat-template overhead per message: {over} tokens (+ role name)")


if __name__ == "__main__":
    sys.exit(main())
