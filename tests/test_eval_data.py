"""Offline integrity checks on the evaluation data.

An eval is only evidence if its labels are right. These tests catch the quiet
ways a dataset goes wrong - a held-out case copied into the dev set, an
"expected phrase" that never appears in the document it points at, a
multi-document question that one document could answer alone - without calling
a model or touching the network.
"""
from __future__ import annotations

import json

import pytest

from evals.run_eval import (
    CORPUS,
    HARD_CORPUS,
    HARD_SET,
    HELDOUT_SET,
    INTENT_SET,
    RETRIEVAL_SET,
    THRESHOLDS,
    _normalise,
    check_thresholds,
    load,
)

INTENTS = {"greeting", "summarization", "reporting", "workflow", "knowledge_lookup", "general_chat"}
CATEGORIES = {"near_duplicate", "paraphrase", "single_doc", "multi_doc"}


def _doc_text(directory, name: str) -> str:
    return _normalise((directory / name).read_text())


@pytest.mark.parametrize("path", [INTENT_SET, HELDOUT_SET], ids=["dev", "heldout"])
def test_intent_sets_are_well_formed(path):
    cases = load(path)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    for case in cases:
        assert case["intent"] in INTENTS, case
        assert case["query"].strip(), case


def test_heldout_does_not_leak_into_dev():
    dev = {_normalise(c["query"]).lower() for c in load(INTENT_SET)}
    leaked = [c["id"] for c in load(HELDOUT_SET) if _normalise(c["query"]).lower() in dev]
    assert not leaked, f"held-out cases also appear in the dev set: {leaked}"


def test_heldout_covers_every_intent():
    counts = {intent: 0 for intent in INTENTS}
    for case in load(HELDOUT_SET):
        counts[case["intent"]] += 1
    thin = {intent: n for intent, n in counts.items() if n < 5}
    assert not thin, f"too few held-out cases to say anything about: {thin}"


def test_easy_retrieval_set_points_at_real_documents():
    names = {p.name for p in CORPUS.glob("*.txt")}
    for case in load(RETRIEVAL_SET):
        assert set(case["expected_documents"]) <= names, case["id"]


def test_hard_set_is_well_formed():
    cases = load(HARD_SET)
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert {c["category"] for c in cases} == CATEGORIES
    for case in cases:
        assert case["answer_contains"], f"{case['id']} has no answer phrase to score"


def test_hard_set_points_at_real_documents():
    names = {p.name for p in HARD_CORPUS.glob("*.txt")}
    for case in load(HARD_SET):
        assert set(case["expected_documents"]) <= names, case["id"]


def test_expected_phrases_appear_in_the_expected_documents():
    """A phrase that is not in the labelled document makes the case unwinnable."""
    for case in load(HARD_SET):
        texts = [_doc_text(HARD_CORPUS, name) for name in case["expected_documents"]]
        for phrase in case["answer_contains"]:
            assert any(_normalise(phrase) in text for text in texts), (
                f"{case['id']}: {phrase!r} is not in {case['expected_documents']}"
            )


def test_single_document_phrases_come_from_that_document():
    for case in load(HARD_SET):
        if case["category"] == "multi_doc":
            continue
        (name,) = case["expected_documents"]
        text = _doc_text(HARD_CORPUS, name)
        for phrase in case["answer_contains"]:
            assert _normalise(phrase) in text, f"{case['id']}: {phrase!r} not in {name}"


def test_multi_document_cases_really_need_every_document():
    """Each expected document must be the only source of at least one phrase."""
    for case in load(HARD_SET):
        if case["category"] != "multi_doc":
            continue
        assert len(case["expected_documents"]) >= 2, case["id"]
        texts = {n: _doc_text(HARD_CORPUS, n) for n in case["expected_documents"]}
        for name in case["expected_documents"]:
            sole = [
                p
                for p in case["answer_contains"]
                if _normalise(p) in texts[name]
                and not any(_normalise(p) in t for n, t in texts.items() if n != name)
            ]
            assert sole, f"{case['id']}: {name} contributes nothing the other documents lack"


def test_phrases_do_not_trivially_match_unrelated_documents():
    """If a phrase appears in many documents, fact recall stops meaning anything."""
    texts = {p.name: _normalise(p.read_text()) for p in HARD_CORPUS.glob("*.txt")}
    for case in load(HARD_SET):
        for phrase in case["answer_contains"]:
            holders = [n for n, t in texts.items() if _normalise(phrase) in t]
            assert len(holders) <= 2, f"{case['id']}: {phrase!r} appears in {holders}"


def test_check_thresholds_flags_a_regression():
    reports = [{"key": "retrieval_hard", "hit_at_1": 0.70}]
    failures = check_thresholds(reports, {"retrieval_hard": {"hit_at_1": 0.85}})
    assert len(failures) == 1 and "hit_at_1" in failures[0]


def test_check_thresholds_passes_when_at_or_above_minimum():
    reports = [{"key": "retrieval_hard", "hit_at_1": 0.85}]
    assert check_thresholds(reports, {"retrieval_hard": {"hit_at_1": 0.85}}) == []


def test_check_thresholds_skips_unmeasured_metrics():
    """Model accuracy is None under --baseline-only and must not fail the check."""
    reports = [{"key": "intent_dev", "baseline_accuracy": 0.37, "model_accuracy": None}]
    thresholds = {"intent_dev": {"baseline_accuracy": 0.30, "model_accuracy": 0.90}}
    assert check_thresholds(reports, thresholds) == []


def test_threshold_file_is_sane():
    thresholds = json.loads(THRESHOLDS.read_text())
    assert set(thresholds) <= {"intent_dev", "intent_heldout", "retrieval", "retrieval_hard"}
    for metrics in thresholds.values():
        for metric, minimum in metrics.items():
            assert 0.0 <= minimum <= 1.0, f"{metric} minimum {minimum} is not a fraction"
