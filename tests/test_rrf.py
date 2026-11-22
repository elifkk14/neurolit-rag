import pytest

from app.retrieve import retrieve, rrf


def test_rrf_rewards_articles_ranked_high_in_both_lists():
    vector = [("a", "ta", 0.9), ("b", "tb", 0.8), ("c", "tc", 0.7)]
    keyword = [("c", "tc-kw", 5.0), ("b", "tb-kw", 4.0), ("d", "td", 3.0)]
    fused = rrf(vector, keyword, k=60)
    # b is 2nd in both lists -> 2/62; c is 3rd + 1st -> 1/63 + 1/61; a only 1st in one -> 1/61
    assert [pmid for pmid, _, _ in fused] == ["c", "b", "a", "d"]
    assert fused[1][2] == pytest.approx(2 / 62)


def test_rrf_keeps_text_from_first_list():
    fused = rrf([("x", "from-vector", 1.0)], [("x", "from-keyword", 1.0)])
    assert fused[0][1] == "from-vector"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="mode"):
        retrieve(conn=None, question="q", mode="magic")


def test_eval_metrics():
    from eval.run import first_relevant_rank, metrics

    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3
    assert first_relevant_rank(["a"], {"z"}) is None
    m = metrics([1, 3, None, 8])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.75
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3 + 1 / 8) / 4)
