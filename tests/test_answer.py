import json

import pytest

from app.answer import NOT_FOUND, AnswerError, answer_question, cited_pmids
from app.retrieve import Passage

PASSAGES = [
    Passage("111", "Tau PET study", 2021, "J1", "Tau PET predicts decline.", 0.9),
    Passage("222", "Amyloid PET study", 2019, "J2", "Amyloid appears early.", 0.8),
]


class FakeLLM:
    """Returns prepared replies in order and records the prompts it received."""

    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete_json(self, system, user):
        self.prompts.append(user)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def test_cited_pmids_handles_groups_and_duplicates():
    text = "A [111]. B [222, 111]. C [333; 444]. Not a citation [abc]."
    assert cited_pmids(text) == ["111", "222", "333", "444"]


def test_valid_answer_returns_only_cited_sources():
    llm = FakeLLM({"found": True, "answer": "Tau PET predicts decline [111].", "citations": ["111"]})
    result = answer_question("q", PASSAGES, llm)
    assert result.found and result.attempts == 1
    assert [p.pmid for p in result.sources] == ["111"]


def test_citation_list_is_rebuilt_from_inline_citations():
    llm = FakeLLM({"found": True, "answer": "X [111]. Y [222].", "citations": []})
    assert [p.pmid for p in answer_question("q", PASSAGES, llm).sources] == ["111", "222"]


def test_invented_pmid_is_retried_with_feedback_then_accepted():
    llm = FakeLLM(
        {"found": True, "answer": "Fact [999].", "citations": ["999"]},
        {"found": True, "answer": "Fact [222].", "citations": ["222"]},
    )
    result = answer_question("q", PASSAGES, llm)
    assert result.attempts == 2
    assert "999" in llm.prompts[1] and "rejected" in llm.prompts[1]


def test_invented_pmid_twice_raises():
    bad = {"found": True, "answer": "Fact [999].", "citations": ["999"]}
    with pytest.raises(AnswerError, match="999"):
        answer_question("q", PASSAGES, FakeLLM(bad, bad))


def test_answer_without_any_citation_is_rejected():
    bad = {"found": True, "answer": "Uncited claim.", "citations": []}
    with pytest.raises(AnswerError, match="no \\[PMID\\]"):
        answer_question("q", PASSAGES, FakeLLM(bad, bad))


def test_broken_json_is_retried():
    llm = FakeLLM("not json", {"found": True, "answer": "A [111].", "citations": ["111"]})
    assert answer_question("q", PASSAGES, llm).attempts == 2


def test_not_found_is_a_valid_answer():
    llm = FakeLLM({"found": False, "answer": "", "citations": []})
    result = answer_question("q", PASSAGES, llm)
    assert not result.found and result.answer == NOT_FOUND and result.sources == []


def test_no_passages_does_not_call_the_llm():
    llm = FakeLLM()
    assert answer_question("q", [], llm).found is False
    assert llm.prompts == []


def test_uncited_summary_sentence_is_rejected_then_fixed():
    llm = FakeLLM(
        {"found": True, "answer": "Tau wins. Tau PET predicts decline [111].", "citations": ["111"]},
        {"found": True, "answer": "Tau PET predicts decline [111].", "citations": ["111"]},
    )
    result = answer_question("q", PASSAGES, llm)
    assert result.attempts == 2
    assert "Tau wins." in llm.prompts[1]


def test_uncited_sentences_splits_after_citations():
    from app.answer import uncited_sentences

    assert uncited_sentences("A is true [111]. B is true [222]. C has none.") == ["C has none."]
    assert uncited_sentences("Values were 0.5 [111]. E.g. x [222].") == []
