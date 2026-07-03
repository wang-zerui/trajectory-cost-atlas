#!/usr/bin/env python3
"""Recompute strong-model cost for all open trajectory datasets under ONE
unified Opus 4.8 list price, straight from raw token counts.

Motivation / fixes over the original pipeline:
  * The original `analyze_opus48.py` mixes recorded per-call cost (Pi's stored
    rate is ~7x below Opus 4.8 list, cache_write billed at $0) with list-price
    numbers baked into a hardcoded PNG/HTML. This script derives every dollar
    figure from token counts at a single, explicit price table so the whole
    dashboard is reproducible and internally consistent.
  * cache_read is priced as input at $0.50/MTok (list). We keep it a separate
    line so the "long session -> cache read dominates" effect is visible per
    dataset instead of being pooled away.

Token sources (all raw, no re-tokenization):
  Pi          : usage.{input,output,cacheRead,cacheWrite} per assistant message
  KernelBook  : result.modelUsage[opus].{input,output,cache_read,cache_creation}
  OpenHands   : static verified totals from analyze_opus48.static_openhands_verification
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "hf_trajectories"

# Unified Claude Opus 4.8 list price, USD per 1M tokens.
PRICE = {"fresh": 5.0, "cache_write": 6.25, "cache_read": 0.50, "decode": 25.0}
OPUS = "claude-opus-4-8"


def cost(tokens: dict[str, int]) -> dict[str, float]:
    return {k: tokens.get(k, 0) * PRICE[k] / 1_000_000 for k in PRICE}


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def summarize(name: str, tokens: dict[str, int], meta: dict[str, Any]) -> dict[str, Any]:
    c = cost(tokens)
    ctot = sum(c.values())
    ttot = sum(tokens.get(k, 0) for k in PRICE)
    return {
        "dataset": name,
        **meta,
        "tokens": {k: tokens.get(k, 0) for k in PRICE},
        "token_total": ttot,
        "cost_usd": c,
        "cost_total_usd": ctot,
        "token_share_pct": {k: pct(tokens.get(k, 0), ttot) for k in PRICE},
        "cost_share_pct": {k: pct(c[k], ctot) for k in PRICE},
    }


def load_pi() -> dict[str, Any]:
    files = sorted(glob.glob(str(DATA / "claude_opus_4_8_pi_traces" / "*.jsonl")))
    tok: Counter[str] = Counter()
    calls = 0
    for f in files:
        for line in open(f, encoding="utf-8"):
            e = json.loads(line)
            if e.get("type") != "message":
                continue
            m = e.get("message") or {}
            if m.get("role") != "assistant":
                continue
            u = m.get("usage") or {}
            calls += 1
            tok["fresh"] += int(u.get("input") or 0)
            tok["decode"] += int(u.get("output") or 0)
            tok["cache_read"] += int(u.get("cacheRead") or 0)
            tok["cache_write"] += int(u.get("cacheWrite") or 0)
    return summarize("Pi", dict(tok), {"sessions": len(files), "calls": calls})


def load_kernelbook() -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = DATA / "kernelbook_opus4_8_multiturn" / "batch_0.parquet"
    rows = pq.read_table(path).to_pylist()
    tok: Counter[str] = Counter()
    turns = 0
    for row in rows:
        raw = json.loads(row.get("trace") or "[]")
        for event in raw:
            if event.get("type") != "result":
                continue
            for model, usage in (event.get("modelUsage") or {}).items():
                if model != OPUS:
                    continue  # weak Haiku aux usage = free by assumption
                tok["fresh"] += int(usage.get("inputTokens") or 0)
                tok["decode"] += int(usage.get("outputTokens") or 0)
                tok["cache_read"] += int(usage.get("cacheReadInputTokens") or 0)
                tok["cache_write"] += int(usage.get("cacheCreationInputTokens") or 0)
        turns += int((row.get("metadata") or {}).get("num_turns") or 0)
    return summarize("KernelBook", dict(tok), {"rows": len(rows), "turns": turns})


def load_openhands() -> dict[str, Any]:
    # Static verified totals (archive not stored in repo). prompt_tokens includes
    # cache; fresh = prompt - cache_read - cache_write.
    prompt = 153_343_721
    cr = 150_242_743
    cw = 3_087_464
    fresh = prompt - cr - cw
    tok = {"fresh": fresh, "cache_read": cr, "cache_write": cw, "decode": 1_208_650}
    return summarize("OpenHands", tok, {"instances": 16, "calls": 1_225})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=HERE / "results" / "recomputed_cost.json")
    args = ap.parse_args()

    datasets = [load_pi(), load_kernelbook(), load_openhands()]

    pooled_tok: Counter[str] = Counter()
    for d in datasets:
        for k in PRICE:
            pooled_tok[k] += d["tokens"][k]
    pooled = summarize("POOLED", dict(pooled_tok), {"note": "sum of 3 datasets, not workload-weighted"})

    report = {
        "price_per_mtok": PRICE,
        "assumption": "weak-model inference is free; strong-side (Opus 4.8) tokens only",
        "datasets": datasets,
        "pooled": pooled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Console table
    print(f"{'dataset':<12} {'total$':>10} {'fresh%':>7} {'cWrite%':>8} {'cRead%':>8} {'decode%':>8}")
    for d in datasets + [pooled]:
        s = d["cost_share_pct"]
        print(f"{d['dataset']:<12} {d['cost_total_usd']:>10.2f} "
              f"{s['fresh']:>6.1f}% {s['cache_write']:>7.1f}% {s['cache_read']:>7.1f}% {s['decode']:>7.1f}%")
    print(f"\npooled token total: {pooled['token_total']:,}")
    print(f"pooled cost total : ${pooled['cost_total_usd']:.2f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
