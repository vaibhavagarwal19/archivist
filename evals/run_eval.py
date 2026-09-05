"""Evaluation harness for the agent's routing and retrieval.

Two suites, run independently:

  intent     Does the LLM classifier pick the right intent, and does it beat
             the keyword heuristic it falls back to? The heuristic is the
             baseline: any claim that the model adds value has to clear it.

  retrieval  For a question with a known source document, does that document
             appear in the top-k chunks? Reported as hit@k and MRR.

Usage:
    python -m evals.run_eval                 # both suites
    python -m evals.run_eval --suite intent
    python -m evals.run_eval --suite retrieval
    python -m evals.run_eval --baseline-only # no API calls, heuristic only
    python -m evals.run_eval --json results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
INTENT_SET = EVAL_DIR / "dataset.jsonl"
RETRIEVAL_SET = EVAL_DIR / "retrieval_dataset.jsonl"
CORPUS = Path(__file__).resolve().parents[1] / "samples"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Intent suite
# ---------------------------------------------------------------------------
async def eval_intent(cases: list[dict], baseline_only: bool = False) -> dict:
    from app.services.langgraph_agent import (
        INTENT_SYSTEM_PROMPT,
        IntentClassification,
        LangGraphAgentService,
    )

    heuristic = LangGraphAgentService._heuristic_intent
    service = None if baseline_only else LangGraphAgentService()

    system_prompt = INTENT_SYSTEM_PROMPT

    rows, latencies, fallbacks = [], [], 0
    for case in cases:
        expected = case["intent"]
        baseline = heuristic(case["query"])

        predicted, latency = None, 0.0
        if service is not None:
            started = time.time()
            try:
                result = await service._invoke_json_model(
                    IntentClassification, system_prompt, f"User query:\n{case['query']}"
                )
                predicted = result.intent
            except Exception:
                # Production behaviour: fall back to the heuristic.
                predicted = baseline
                fallbacks += 1
            latency = time.time() - started
            latencies.append(latency)

        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": expected,
                "baseline": baseline,
                "predicted": predicted,
                "baseline_correct": baseline == expected,
                "model_correct": predicted == expected if predicted else None,
                "latency_s": round(latency, 3),
            }
        )

    total = len(rows)
    baseline_acc = sum(r["baseline_correct"] for r in rows) / total
    model_acc = (
        sum(bool(r["model_correct"]) for r in rows) / total if service is not None else None
    )

    per_intent = defaultdict(lambda: {"n": 0, "baseline": 0, "model": 0})
    for r in rows:
        bucket = per_intent[r["expected"]]
        bucket["n"] += 1
        bucket["baseline"] += r["baseline_correct"]
        bucket["model"] += bool(r["model_correct"])

    return {
        "suite": "intent",
        "cases": total,
        "baseline_accuracy": baseline_acc,
        "model_accuracy": model_acc,
        "fallbacks": fallbacks,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_max": round(max(latencies), 3) if latencies else None,
        "per_intent": {k: dict(v) for k, v in sorted(per_intent.items())},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Retrieval suite
# ---------------------------------------------------------------------------
def eval_retrieval(cases: list[dict], k: int = 4) -> dict:
    """Index the sample corpus into a throwaway index and measure hit@k / MRR."""
    from app.services.vector_store import VectorStoreService
    from app.utils.text import chunk_text

    tmp = Path(tempfile.mkdtemp(prefix="eval-index-"))
    try:
        store = VectorStoreService()
        store.index_path = tmp / "index"
        store.index_path.parent.mkdir(parents=True, exist_ok=True)

        for doc_id, path in enumerate(sorted(CORPUS.glob("*.txt")), start=1):
            chunks = chunk_text(path.read_text())
            store.add_documents(
                chunks,
                [{"document_id": doc_id, "file_name": path.name, "owner_id": 1} for _ in chunks],
            )

        rows, reciprocal_ranks = [], []
        for case in cases:
            results = store.search_chunks(case["query"], k=k, owner_id=1)
            names = [r["file_name"] for r in results]
            expected = set(case["expected_documents"])

            rank = next((i + 1 for i, n in enumerate(names) if n in expected), 0)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            rows.append(
                {
                    "id": case["id"],
                    "query": case["query"],
                    "expected": sorted(expected),
                    "retrieved": names,
                    "hit": bool(rank),
                    "rank": rank or None,
                }
            )

        hits = sum(r["hit"] for r in rows)
        return {
            "suite": "retrieval",
            "cases": len(rows),
            "k": k,
            f"hit_at_{k}": hits / len(rows),
            "mrr": round(sum(reciprocal_ranks) / len(rows), 3),
            "rows": rows,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def render(report: dict) -> None:
    if report["suite"] == "intent":
        print(f"\n{'=' * 62}\nINTENT ROUTING  ({report['cases']} cases)\n{'=' * 62}")
        base = report["baseline_accuracy"]
        model = report["model_accuracy"]
        print(f"  keyword baseline : {base:.1%}")
        if model is not None:
            delta = model - base
            print(f"  LLM classifier   : {model:.1%}   ({delta:+.1%} vs baseline)")
            print(f"  heuristic fallbacks used: {report['fallbacks']}")
            print(f"  latency p50 {report['latency_p50']}s | max {report['latency_max']}s")
        print(f"\n  {'intent':<18} {'n':>3}  {'baseline':>9}  {'model':>7}")
        for intent, s in report["per_intent"].items():
            b = s["baseline"] / s["n"]
            m = s["model"] / s["n"]
            print(f"  {intent:<18} {s['n']:>3}  {b:>8.0%}  {m:>6.0%}")
        wrong = [r for r in report["rows"] if r["model_correct"] is False]
        if wrong:
            print(f"\n  misrouted by the model ({len(wrong)}):")
            for r in wrong:
                print(f"    {r['id']}  expected {r['expected']:<17} got {r['predicted']:<17} {r['query'][:44]}")

    else:
        k = report["k"]
        print(f"\n{'=' * 62}\nRETRIEVAL  ({report['cases']} cases, k={k})\n{'=' * 62}")
        print(f"  hit@{k} : {report[f'hit_at_{k}']:.1%}")
        print(f"  MRR   : {report['mrr']}")
        misses = [r for r in report["rows"] if not r["hit"]]
        if misses:
            print(f"\n  missed ({len(misses)}):")
            for r in misses:
                print(f"    {r['id']}  expected {r['expected']}  got {r['retrieved']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["intent", "retrieval", "all"], default="all")
    parser.add_argument("--baseline-only", action="store_true", help="Skip LLM calls")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--json", type=Path, help="Write full results to this file")
    args = parser.parse_args()

    reports = []
    if args.suite in ("intent", "all"):
        reports.append(asyncio.run(eval_intent(load(INTENT_SET), args.baseline_only)))
    if args.suite in ("retrieval", "all"):
        reports.append(eval_retrieval(load(RETRIEVAL_SET), k=args.k))

    for report in reports:
        render(report)

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
