# Evaluation

Two suites measure the parts of the agent where "it looks right" is not
evidence: which intent the router picks, and whether retrieval surfaces the
document that actually contains the answer.

```bash
python -m evals.run_eval                  # every suite
python -m evals.run_eval --suite intent
python -m evals.run_eval --baseline-only  # heuristic only, no API calls
python -m evals.run_eval --json out.json  # full per-case results
```

## Suites

| Suite | Data | What it measures |
|---|---|---|
| `intent` (dev) | `dataset.jsonl`, 54 cases | Intent routing. The prompt was tuned against this set. |
| `intent` (held-out) | `heldout_dataset.jsonl`, 42 cases | Intent routing on cases never used for tuning. |
| `retrieval` | `retrieval_dataset.jsonl`, 15 cases over 5 documents | Smoke test. Easy by construction. |
| `retrieval_hard` | `retrieval_hard_dataset.jsonl`, 47 cases over 26 documents | Retrieval under near-duplicates, paraphrase and multi-document questions. |

```bash
python -m evals.run_eval --suite retrieval_hard
python -m evals.run_eval --baseline-only --check   # what CI runs: no API key needed
python -m evals.run_eval --suite intent --check    # needs GROQ_API_KEY
```

`--check` exits non-zero if any metric falls below its minimum in
`thresholds.json`. CI runs it on every push for everything that needs no API
key. Metrics not measured in a run (model accuracy under `--baseline-only`) are
skipped, not failed.

## Dev vs held-out: why two intent sets

The intent prompt was revised four times while watching results on the 54-case
set (table below). A score on cases you tuned against measures how well you fit
them, not how well the router generalises. The held-out set exists to measure
the second thing: the run prints both and the gap between them.

**Rule:** do not edit the intent prompt while looking at held-out misses. If you
do, move those cases into the dev set and write fresh held-out ones.

| Set | Cases | Keyword baseline | LLM classifier | Fallbacks |
|---|---|---|---|---|
| Dev (tuned against) | 54 | 37.0% | 96.3% | 0 |
| Held-out | 42 | 31.0% | 100.0% | 0 |

No sign of overfitting: the held-out score is not lower than the dev score. Read
the 100% with care, though. 42 of 42 only bounds the true accuracy below at
about 92% (95% confidence), and the held-out cases were written after the
prompt, so they are clean, unambiguous requests. The dev set scores lower
because it contains the two deliberately ambiguous cases. The held-out set
shows the router handles ordinary phrasing it has not seen; it does not show
how it handles borderline requests. Harder held-out cases (a question that could
be public knowledge or a company policy, a request that is both a report and an
action, multi-turn references) are the next addition.

## Hard retrieval suite

The corpus is 26 short synthetic documents for a fictional company, generated
for this suite. It deliberately contains families of near-duplicates, where the
right answer depends on picking the correct document:

- leave policy for the US, India and the UK
- Sev-1, Sev-2 and security incident runbooks
- expense policy v2 (current) and v1 (superseded)
- two vendor contracts with different renewal and termination terms

Four question categories: `near_duplicate` (18), `paraphrase` with little word
overlap (12), `single_doc` (11) and `multi_doc` where the answer needs two
documents (6).

Metrics: **hit@1**, **hit@k**, **MRR**, and **fact recall**, which asks whether
a retrieved chunk actually contains the answer phrase. Fact recall is stricter
than hit@k: the right file can come back with the wrong paragraph. For
`multi_doc` cases, hit@k requires every expected document.

Current results (fastembed `bge-small-en-v1.5`, sentence-aware chunking, k=4):

| Category | n | hit@1 | hit@4 | MRR | Fact recall |
|---|---|---|---|---|---|
| near_duplicate | 18 | 83% | 100% | 0.92 | 100% |
| paraphrase | 12 | 83% | 92% | 0.88 | 92% |
| single_doc | 11 | 91% | 100% | 0.95 | 100% |
| multi_doc | 6 | 100% | 100% | 1.00 | 100% |
| **All** | **47** | **87.2%** | **97.9%** | **0.926** | **97.9%** |

The one miss is `pp-02` ("How long can the person holding the pager stay silent
before somebody else is alerted?"): the question shares almost no words with
"the secondary is paged if the primary does not acknowledge within 10 minutes",
and incident runbooks outrank the on-call document.

**Limits of this suite, stated plainly:**

- hit@4 is already near its ceiling, so the headroom is in hit@1 and the
  paraphrase category. That is where hybrid search or reranking would show up.
- The corpus and questions were written together by the same author, so they
  are not adversarial. They test retrieval on clean, short documents; they say
  nothing about long, messy or scanned ones.
- 47 cases is enough to see large changes, not small ones. One case is 2.1
  points.

`tests/test_eval_data.py` checks the data offline: every expected phrase really
appears in its labelled document, each multi-document case needs every document,
and the held-out set shares no query with the dev set.

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
