# Evaluation

Two suites measure the parts of the agent where "it looks right" is not
evidence: which intent the router picks, and whether retrieval surfaces the
document that actually contains the answer.

```bash
python -m evals.run_eval                  # both suites
python -m evals.run_eval --suite intent
python -m evals.run_eval --baseline-only  # heuristic only, no API calls
python -m evals.run_eval --json out.json  # full per-case results
```

## Why a baseline

Every LLM decision in this agent has a keyword-heuristic fallback. That
fallback is also the honest baseline: if the model cannot beat the keywords,
the model is not earning its latency or its cost. The suite reports both on
every run, so the comparison is never assumed.

## Intent routing — 54 labelled cases

| Iteration | Baseline | LLM | Notes |
|---|---|---|---|
| Original prompt | 37.0% | — | Model output failed schema validation on every call; the agent silently ran on keywords alone. |
| Schema-constrained decoding | 37.0% | 88.9% | Sending the JSON Schema with the request fixed validation. |
| Company-vs-public distinction | 37.0% | 92.6% | `general_chat` 50% → 100%; `reporting` regressed to 67%. |
| Ordered decision rules | **37.0%** | **96.3%** | Regression recovered; 0 fallbacks. |

Final per-intent accuracy:

| Intent | n | Baseline | Model |
|---|---|---|---|
| general_chat | 8 | 100% | 100% |
| greeting | 6 | 33% | 83% |
| knowledge_lookup | 12 | 25% | 92% |
| reporting | 9 | 33% | 100% |
| summarization | 9 | 22% | 100% |
| workflow | 10 | 20% | 100% |

Latency p50 1.9–3.7s per classification depending on prompt length.

**Two cases still miss, and both are ambiguous labels rather than model
failures:** "morning! anything for me today?" (greeting or request?) and
"Remind me of the benefits enrollment deadline" ("remind me" reads as an
action). They are kept in the set as a standing reminder that the ceiling here
is not 100%.

### What the eval caught that testing would not

The first full run scored 88.9%, and **four of the six errors were the same
failure**: general questions such as "What time zone is UTC+5:30?" routed to
`knowledge_lookup`, so the agent searched company documents for world
knowledge. No unit test would have flagged that — every component behaved
correctly. Fixing it, then watching `reporting` regress on the next run and
recovering it, is the loop this harness exists for.

## Retrieval — 15 labelled cases

| Metric | Character chunking | Sentence-aware chunking |
|---|---|---|
| hit@4 | 100% | 100% |
| MRR | 0.900 | **0.933** |

Each case names the document(s) that genuinely contain the answer; cases whose
answer legitimately appears in two documents accept either.

## Adding cases

`dataset.jsonl` — one JSON object per line: `id`, `query`, `intent`.
`retrieval_dataset.jsonl` — `id`, `query`, `expected_documents` (a list).

Write queries the way a colleague would actually phrase them. A set that
mirrors the keyword heuristics inflates the baseline; one written to defeat
them deflates it. Neither tells you anything useful.
