from pathlib import Path

from ingest.pubmed import parse_articles

FIXTURE = Path(__file__).parent / "fixtures" / "efetch_sample.xml"


def _parsed():
    return {a.pmid: a for a in parse_articles(FIXTURE.read_text())}


def test_records_without_abstract_are_skipped():
    assert set(_parsed()) == {"11111111", "22222222"}


def test_structured_abstract_keeps_labels_and_nested_text():
    a = _parsed()["11111111"]
    assert a.title == "Amyloid PET in early Alzheimer disease."
    assert a.abstract == "BACKGROUND: Amyloid accumulates early.\nRESULTS: Tau PET tracked 18F decline."
    assert a.year == 2021
    assert a.journal == "Journal of Test Neuroimaging"


def test_medline_date_falls_back_to_first_four_digits():
    assert _parsed()["22222222"].year == 2019


def test_parse_years():
    from ingest.pubmed import parse_years

    assert parse_years("2015-2017") == [2015, 2016, 2017]
    assert parse_years("2020") == [2020]
