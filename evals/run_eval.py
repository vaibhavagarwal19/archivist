"""Evaluation harness for the agent's routing and retrieval.

Suites, run independently:

  intent          Does the LLM classifier pick the right intent, and does it beat
                  the keyword heuristic it falls back to? The heuristic is the
                  baseline: any claim that the model adds value has to clear it.
                  Run on two sets, reported separately:
                    dev      dataset.jsonl          the prompt was tuned against it
                    heldout  heldout_dataset.jsonl  never used to tune anything

  retrieval       The original 15-case suite over 5 small sample documents.
                  Easy by construction; kept as a smoke test.

  retrieval_hard  47 questions over a 26-document synthetic corpus containing
                  near-duplicate documents, paraphrased questions and questions
                  that need two documents. Reports hit@1, hit@k, MRR and fact
                  recall (does a retrieved chunk actually contain the answer).

Usage:
    python -m evals.run_eval                       # every suite
    python -m evals.run_eval --suite intent
    python -m evals.run_eval --suite retrieval_hard
    python -m evals.run_eval --baseline-only       # no API calls, heuristic only
    python -m evals.run_eval --check               # exit 1 if below thresholds.json
    python -m evals.run_eval --json results.json

The held-out set exists to measure generalisation. Do not edit the intent prompt
while looking at its misses; if you do, move those cases to the dev set and
write fresh held-out ones.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

EVAL_DIR = Path(__file__).parent
INTENT_SET = EVAL_DIR / "dataset.jsonl"
HELDOUT_SET = EVAL_DIR / "heldout_dataset.jsonl"
RETRIEVAL_SET = EVAL_DIR / "retrieval_dataset.jsonl"
HARD_SET = EVAL_DIR / "retrieval_hard_dataset.jsonl"
THRESHOLDS = EVAL_DIR / "thresholds.json"
CORPUS = Path(__file__).resolve().parents[1] / "samples"
HARD_CORPUS = EVAL_DIR / "corpus"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _normalise(text: str) -> str:
    """Collapse whitespace so chunk joins and line breaks never hide a match."""
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Intent suite
# ---------------------------------------------------------------------------
async def eval_intent(cases: list[dict], label: str, baseline_only: bool = False) -> dict:
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
        "key": f"intent_{label}",
        "label": label,
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
# Retrieval suites
# ---------------------------------------------------------------------------
def _build_index(corpus_dir: Path, tmp: Path):
    """Index every .txt file in `corpus_dir` into a throwaway FAISS index."""
    from app.services.vector_store import VectorStoreService
    from app.utils.text import chunk_text

    store = VectorStoreService()
    store.index_path = tmp / "index"
    store.index_path.parent.mkdir(parents=True, exist_ok=True)

    for doc_id, path in enumerate(sorted(corpus_dir.glob("*.txt")), start=1):
        chunks = chunk_text(path.read_text())
        store.add_documents(
            chunks,
            [{"document_id": doc_id, "file_name": path.name, "owner_id": 1} for _ in chunks],
        )
    return store


def eval_retrieval(cases: list[dict], k: int = 4) -> dict:
    """Index the sample corpus into a throwaway index and measure hit@k / MRR."""
    tmp = Path(tempfile.mkdtemp(prefix="eval-index-"))
    try:
        store = _build_index(CORPUS, tmp)

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
            "key": "retrieval",
            "cases": len(rows),
            "k": k,
            f"hit_at_{k}": hits / len(rows),
            "mrr": round(sum(reciprocal_ranks) / len(rows), 3),
            "rows": rows,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def eval_retrieval_hard(cases: list[dict], k: int = 4) -> dict:
    """Score retrieval on the synthetic corpus at document and fact level.

    hit@1       the top chunk comes from an expected document
    hit@k       an expected document is in the top k (for multi_doc cases, ALL
                expected documents must be, since the answer needs both)
    MRR         reciprocal rank of the first expected document
    fact recall every phrase in `answer_contains` appears in some retrieved
                chunk. Stricter than hit@k: the right file can come back with
                the wrong paragraph.
    """
    tmp = Path(tempfile.mkdtemp(prefix="eval-hard-index-"))
    try:
        store = _build_index(HARD_CORPUS, tmp)

        rows = []
        for case in cases:
            results = store.search_chunks(case["query"], k=k, owner_id=1)
            names = [r["file_name"] for r in results]
            expected = case["expected_documents"]
            multi = case["category"] == "multi_doc"

            rank = next((i + 1 for i, n in enumerate(names) if n in expected), 0)
            doc_hit = all(e in names for e in expected) if multi else bool(rank)
            retrieved_text = _normalise(" ".join(r["content"] for r in results))
            missing_facts = [
                f for f in case["answer_contains"] if _normalise(f) not in retrieved_text
            ]
            rows.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "query": case["query"],
                    "expected": expected,
                    "retrieved": names,
                    "hit_1": bool(names) and names[0] in expected,
                    "hit_k": doc_hit,
                    "rank": rank or None,
                    "reciprocal_rank": 1 / rank if rank else 0.0,
                    "fact_hit": not missing_facts,
                    "missing_facts": missing_facts,
                }
            )

        def summarise(subset: list[dict]) -> dict:
            n = len(subset)
            return {
                "n": n,
                "hit_at_1": sum(r["hit_1"] for r in subset) / n,
                f"hit_at_{k}": sum(r["hit_k"] for r in subset) / n,
                "mrr": round(sum(r["reciprocal_rank"] for r in subset) / n, 3),
                f"fact_recall_at_{k}": sum(r["fact_hit"] for r in subset) / n,
            }

        by_category: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_category[r["category"]].append(r)

        return {
            "suite": "retrieval_hard",
            "key": "retrieval_hard",
            "cases": len(rows),
            "k": k,
            "documents": len(list(HARD_CORPUS.glob("*.txt"))),
            **summarise(rows),
            "per_category": {c: summarise(rs) for c, rs in sorted(by_category.items())},
            "rows": rows,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------
def check_thresholds(reports: list[dict], thresholds: dict) -> list[str]:
    """Return one message per metric that fell below its recorded minimum.

    A metric that was not measured this run (e.g. model accuracy under
    --baseline-only) is skipped rather than failed.
    """
    failures = []
    for report in reports:
        for metric, minimum in thresholds.get(report["key"], {}).items():
            value = report.get(metric)
            if value is None:
                continue
            if value + 1e-9 < minimum:
                failures.append(f"{report['key']}.{metric} = {value:.3f} < required {minimum:.3f}")
    return failures


# ---------------------------------------------------------------------------
def render(report: dict) -> None:
    if report["suite"] == "intent":
        label = report["label"]
        print(f"\n{'=' * 62}\nINTENT ROUTING [{label}]  ({report['cases']} cases)\n{'=' * 62}")
        base = report["baseline_accuracy"]
        model = report["model_accuracy"]
        print(f"  keyword baseline : {base:.1%}")
        if model is not None:
            delta = model - base
            print(f"  LLM classifier   : {model:.1%}   ({delta:+.1%} vs baseline)")
            print(f"  heuristic fallbacks used: {report['fallbacks']}")
            print(f"  latency p50 {report['latency_p50']}s | max {report['latency_max']}s")
        show_model = model is not None
        header = f"\n  {'intent':<18} {'n':>3}  {'baseline':>9}"
        print(header + (f"  {'model':>7}" if show_model else ""))
        for intent, s in report["per_intent"].items():
            line = f"  {intent:<18} {s['n']:>3}  {s['baseline'] / s['n']:>8.0%}"
            if show_model:
                line += f"  {s['model'] / s['n']:>6.0%}"
            print(line)
        wrong = [r for r in report["rows"] if r["model_correct"] is False]
        if wrong:
            print(f"\n  misrouted by the model ({len(wrong)}):")
            for r in wrong:
                print(f"    {r['id']}  expected {r['expected']:<17} got {r['predicted']:<17} {r['query'][:44]}")

    elif report["suite"] == "retrieval":
        k = report["k"]
        print(f"\n{'=' * 62}\nRETRIEVAL  ({report['cases']} cases, k={k})\n{'=' * 62}")
        print(f"  hit@{k} : {report[f'hit_at_{k}']:.1%}")
        print(f"  MRR   : {report['mrr']}")
        misses = [r for r in report["rows"] if not r["hit"]]
        if misses:
            print(f"\n  missed ({len(misses)}):")
            for r in misses:
                print(f"    {r['id']}  expected {r['expected']}  got {r['retrieved']}")

    else:
        k = report["k"]
        print(
            f"\n{'=' * 62}\nRETRIEVAL HARD  ({report['cases']} cases, "
            f"{report['documents']} documents, k={k})\n{'=' * 62}"
        )
        print(f"  hit@1        : {report['hit_at_1']:.1%}")
        print(f"  hit@{k}        : {report[f'hit_at_{k}']:.1%}")
        print(f"  MRR          : {report['mrr']}")
        print(f"  fact recall@{k}: {report[f'fact_recall_at_{k}']:.1%}")
        print(f"\n  {'category':<16} {'n':>3}  {'hit@1':>6}  {f'hit@{k}':>6}  {'MRR':>5}  {'fact':>6}")
        for category, s in report["per_category"].items():
            print(
                f"  {category:<16} {s['n']:>3}  {s['hit_at_1']:>6.0%}  {s[f'hit_at_{k}']:>6.0%}"
                f"  {s['mrr']:>5.2f}  {s[f'fact_recall_at_{k}']:>6.0%}"
            )
        misses = [r for r in report["rows"] if not r["hit_k"] or not r["fact_hit"]]
        if misses:
            print(f"\n  missed ({len(misses)}):")
            for r in misses:
                kind = "doc" if not r["hit_k"] else "fact"
                print(f"    {r['id']} [{kind}]  expected {r['expected']}  got {r['retrieved']}")
                if r["missing_facts"]:
                    print(f"        missing: {r['missing_facts']}")


def render_generalisation(reports: list[dict]) -> None:
    """Show the dev vs held-out gap, which is the honest accuracy number."""
    by_label = {r["label"]: r for r in reports if r["suite"] == "intent"}
    dev, held = by_label.get("dev"), by_label.get("heldout")
    if not dev or not held or dev["model_accuracy"] is None or held["model_accuracy"] is None:
        return
    gap = dev["model_accuracy"] - held["model_accuracy"]
    print(f"\n{'=' * 62}\nGENERALISATION\n{'=' * 62}")
    print(f"  dev (tuned against) : {dev['model_accuracy']:.1%}  ({dev['cases']} cases)")
    print(f"  held-out            : {held['model_accuracy']:.1%}  ({held['cases']} cases)")
    print(f"  gap                 : {gap:+.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--suite", choices=["intent", "retrieval", "retrieval_hard", "all"], default="all"
    )
    parser.add_argument("--baseline-only", action="store_true", help="Skip LLM calls")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--json", type=Path, help="Write full results to this file")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any metric is below its minimum in thresholds.json",
    )
    args = parser.parse_args()

    reports = []
    if args.suite in ("intent", "all"):
        reports.append(asyncio.run(eval_intent(load(INTENT_SET), "dev", args.baseline_only)))
        reports.append(asyncio.run(eval_intent(load(HELDOUT_SET), "heldout", args.baseline_only)))
    if args.suite in ("retrieval", "all"):
        reports.append(eval_retrieval(load(RETRIEVAL_SET), k=args.k))
    if args.suite in ("retrieval_hard", "all"):
        reports.append(eval_retrieval_hard(load(HARD_SET), k=args.k))

    for report in reports:
        render(report)
    render_generalisation(reports)

    if args.json:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f"\nwrote {args.json}")

    if args.check:
        failures = check_thresholds(reports, json.loads(THRESHOLDS.read_text()))
        if failures:
            print("\nREGRESSION CHECK FAILED")
            for message in failures:
                print(f"  - {message}")
            sys.exit(1)
        print("\nregression check passed")


if __name__ == "__main__":
    main()
