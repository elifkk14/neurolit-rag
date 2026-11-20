from ingest.chunk import chunk_abstract, split_sentences


def test_does_not_split_on_abbreviations_or_decimals():
    text = "Effects were seen, e.g. in the hippocampus. Values had p < 0.05 in all. Done."
    assert split_sentences(text) == [
        "Effects were seen, e.g. in the hippocampus.",
        "Values had p < 0.05 in all.",
        "Done.",
    ]


def test_short_abstract_is_one_chunk_with_title():
    chunks = chunk_abstract("Title here", "First sentence. Second sentence.")
    assert chunks == ["Title here\nFirst sentence. Second sentence."]


def test_long_abstract_is_split_with_overlap_and_within_budget():
    sentences = [f"Sentence {i} has exactly six words." for i in range(30)]  # 180 words
    chunks = chunk_abstract("T", " ".join(sentences), max_words=61)  # 60-word budget = 10 sentences

    assert len(chunks) > 1
    for c in chunks:
        assert c.startswith("T\n")
        assert len(c.split()) <= 61
    # last sentence of chunk 1 is repeated as the first sentence of chunk 2
    first_body, second_body = chunks[0].split("\n")[1], chunks[1].split("\n")[1]
    assert second_body.startswith(first_body.split(". ")[-1].rstrip("."))
    # nothing lost
    joined = " ".join(c.split("\n")[1] for c in chunks)
    assert all(s in joined for s in sentences)
