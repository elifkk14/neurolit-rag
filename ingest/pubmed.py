"""Download PubMed abstracts with the NCBI E-utilities API.

Two-step flow:
  1. esearch  -> run the query on the NCBI server and keep the result list there
                 (usehistory=y). We get back a WebEnv + query_key, not the PMIDs.
  2. efetch   -> fetch the records in batches using that WebEnv/query_key.

Sampling: PubMed returns newest records first, so "the first 5000 hits" would be almost
all from the last two years. Instead we run one search per publication year and take
the same number of records from each year (stratified sample).

NCBI rules (https://www.ncbi.nlm.nih.gov/books/NBK25497/):
  * max 3 requests/second without an API key, 10/second with one
  * send `tool` and `email` so NCBI can contact you instead of blocking you

Usage:
  python -m ingest.pubmed                       # 5000 records, 2015-2026
  python -m ingest.pubmed --max 24 --years 2020-2021 --out data/raw/sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

log = logging.getLogger("ingest.pubmed")

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "neurolit-rag"
BATCH_SIZE = 200  # efetch accepts up to 10k, but smaller batches fail less often

DEFAULT_QUERY = (
    '"alzheimer disease"[MeSH Terms] '
    'AND ("positron-emission tomography"[MeSH Terms] OR "magnetic resonance imaging"[MeSH Terms]) '
    "AND hasabstract "
    "AND english[Language]"
)


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    year: int | None
    journal: str


# ---------------------------------------------------------------- HTTP


class EutilsClient:
    """Small wrapper that adds the NCBI parameters, rate limit and retries."""

    def __init__(self, email: str | None, api_key: str | None, session: requests.Session | None = None):
        self.email = email
        self.api_key = api_key
        self.session = session or requests.Session()
        # 3 req/s without key -> wait 0.34 s between calls; 10 req/s with key -> 0.11 s
        self.min_interval = 0.11 if api_key else 0.34
        self._last_call = 0.0

    def _params(self, extra: dict) -> dict:
        params = {"db": "pubmed", "tool": TOOL_NAME, **extra}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def get(self, endpoint: str, params: dict, retries: int = 4) -> requests.Response:
        for attempt in range(1, retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

            try:
                resp = self.session.get(f"{EUTILS}/{endpoint}", params=self._params(params), timeout=30)
            except requests.RequestException as exc:
                log.warning("%s failed (%s), attempt %d/%d", endpoint, exc, attempt, retries)
            else:
                # 429 = too many requests, 5xx = NCBI side problem -> retry; anything else is final
                if resp.status_code == 200:
                    return resp
                if resp.status_code != 429 and resp.status_code < 500:
                    resp.raise_for_status()
                log.warning("%s returned %d, attempt %d/%d", endpoint, resp.status_code, attempt, retries)
            time.sleep(2**attempt)  # 2, 4, 8, 16 s
        raise RuntimeError(f"{endpoint} failed after {retries} attempts")

    def search(self, query: str) -> tuple[int, str, str]:
        """Return (total hit count, WebEnv, query_key)."""
        params = {"term": query, "usehistory": "y", "retmax": 0, "retmode": "json"}
        resp = self.get("esearch.fcgi", params)
        result = resp.json()["esearchresult"]
        return int(result["count"]), result["webenv"], result["querykey"]

    def fetch_batch(self, webenv: str, query_key: str, start: int, size: int) -> str:
        resp = self.get(
            "efetch.fcgi",
            {
                "WebEnv": webenv,
                "query_key": query_key,
                "retstart": start,
                "retmax": size,
                "rettype": "abstract",
                "retmode": "xml",
            },
        )
        return resp.text


# ---------------------------------------------------------------- XML parsing


def _text(elem: ET.Element | None) -> str:
    """All text inside an element, including nested tags like <i> or <sup>."""
    if elem is None:
        return ""
    return " ".join("".join(elem.itertext()).split())


def _year(article: ET.Element) -> int | None:
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None
    year = pub_date.findtext("Year")
    if year is None:
        # some records only have e.g. <MedlineDate>2019 Nov-Dec</MedlineDate>
        year = (pub_date.findtext("MedlineDate") or "")[:4]
    return int(year) if year.isdigit() else None


def _abstract(article: ET.Element) -> str:
    """Join abstract sections. Structured abstracts keep their label: 'METHODS: ...'."""
    parts = []
    for section in article.findall(".//Abstract/AbstractText"):
        text = _text(section)
        if not text:
            continue
        label = section.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def parse_articles(xml_text: str) -> list[Article]:
    root = ET.fromstring(xml_text)
    articles = []
    for record in root.findall("PubmedArticle"):
        pmid = record.findtext("MedlineCitation/PMID")
        art = record.find("MedlineCitation/Article")
        if not pmid or art is None:
            continue
        abstract = _abstract(art)
        if not abstract:  # nothing to search over -> skip
            continue
        articles.append(
            Article(
                pmid=pmid,
                title=_text(art.find("ArticleTitle")),
                abstract=abstract,
                year=_year(art),
                journal=_text(art.find("Journal/Title")),
            )
        )
    return articles


# ---------------------------------------------------------------- main flow


def parse_years(spec: str) -> list[int]:
    """'2015-2026' -> [2015, ..., 2026]; '2020' -> [2020]."""
    first, _, last = spec.partition("-")
    return list(range(int(first), int(last or first) + 1))


def download(client: EutilsClient, query: str, years: list[int], max_records: int, out_path: Path) -> int:
    per_year = math.ceil(max_records / len(years))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for year in years:
            total, webenv, query_key = client.search(f"({query}) AND {year}[dp]")
            target = min(total, per_year)
            year_saved = 0
            for start in range(0, target, BATCH_SIZE):
                size = min(BATCH_SIZE, target - start)
                for article in parse_articles(client.fetch_batch(webenv, query_key, start, size)):
                    if article.pmid in seen:
                        continue
                    seen.add(article.pmid)
                    f.write(json.dumps(asdict(article), ensure_ascii=False) + "\n")
                    year_saved += 1
            written += year_saved
            log.info("%d: %d matches, saved %d (total %d)", year, total, year_saved, written)
    return written


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--years", default="2015-2026", help="publication years, e.g. 2015-2026")
    parser.add_argument("--max", type=int, default=5000, dest="max_records", help="total records")
    parser.add_argument("--out", type=Path, default=Path(f"data/raw/pubmed_{date.today():%Y%m%d}.jsonl"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    email = os.getenv("NCBI_EMAIL")
    if not email:
        log.warning("NCBI_EMAIL is not set. It works, but NCBI asks for a contact email (see .env.example).")
    client = EutilsClient(email=email, api_key=os.getenv("NCBI_API_KEY"))

    n = download(client, args.query, parse_years(args.years), args.max_records, args.out)
    log.info("done: %d abstracts -> %s", n, args.out)


if __name__ == "__main__":
    main()
