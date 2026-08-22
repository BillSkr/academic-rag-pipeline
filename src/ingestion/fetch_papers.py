"""
fetch_papers.py

Offline data collection script for the Agentic RAG Academic Assistant.

Queries the PubMed / NCBI E-utilities API to retrieve biomedical
research papers about Amyotrophic Lateral Sclerosis (ALS).

Workflow:
    1. ESearch  – query PubMed and obtain a list of PMIDs.
    2. EFetch   – retrieve full XML records for those PMIDs.
    3. Parse    – extract Title, Abstract, Authors, and Year from the XML.
    4. Persist  – write the structured results to ``data/corpus.json``.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any

import requests
import urllib.parse


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DB = "pubmed"
SEARCH_TERM = "amyotrophic lateral sclerosis"
RETMAX = 55
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "corpus.json"
SLEEP_SECONDS = 1


def search_pubmed(term: str, retmax: int) -> List[str]:
    """Query the PubMed ESearch API and return a list of PMIDs."""
    url = (
        f"{ESEARCH_URL}?db={DB}&term={urllib.parse.quote(term)}"
        f"&retmax={retmax}&retmode=json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    pmids: List[str] = data.get("esearchresult", {}).get("idlist", [])
    return pmids

def fetch_pubmed_records(pmids: List[str]) -> str:
    """Retrieve full XML records from PubMed for the given PMIDs."""
    id_string = ",".join(pmids)
    url = f"{EFETCH_URL}?db={DB}&id={id_string}&rettype=abstract&retmode=xml"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text



def parse_pubmed_xml(xml_string: str) -> List[Dict[str, Any]]:
    """Parse the PubMed XML and extract structured metadata for each article."""
    root = ET.fromstring(xml_string)
    results: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle", default="")
        abstract = " ".join(
            [elem.text for elem in article.findall(".//AbstractText") if elem.text]
        )
        authors = [
            f"{author.findtext('LastName', default='')} {author.findtext('ForeName', default='')}".strip()
            for author in article.findall(".//AuthorList/Author")
        ]
        year = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate", default="")

        results.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year
        })

    return results


def save_corpus(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Persist the extracted article records as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} records to {output_path}")


def run_collection_pipeline() -> None:
    """End-to-end orchestrator that wires the four steps together."""
    pmids = search_pubmed(SEARCH_TERM, RETMAX)
    if not pmids:
        print("No PMIDs found for the given search term.")
        return

    time.sleep(SLEEP_SECONDS)
    xml_string = fetch_pubmed_records(pmids)
    records = parse_pubmed_xml(xml_string)
    save_corpus(records, OUTPUT_PATH)
    print(f"Fetched {len(records)} articles, saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_collection_pipeline()
