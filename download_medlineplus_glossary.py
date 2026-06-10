"""
Download MedlinePlus Health Topics XML into an offline glossary.
================================================================
Run this LOCALLY (with internet) before submitting to the cluster.

The XML is published daily (Tue-Sat) at ~29 MB.  This script parses it
into a compact JSON glossary: { "term": "plain-English definition", ... }

Creates: data/medlineplus_glossary.json

Usage:
    python download_medlineplus_glossary.py
    python download_medlineplus_glossary.py --date 2026-03-10
"""

import json
import os
import re
import argparse
from datetime import datetime, timedelta

import requests
import xml.etree.ElementTree as ET


OUTPUT_PATH = "data/medlineplus_glossary.json"
BASE_URL = "https://medlineplus.gov/xml"


def get_xml_url(date_str: str = None) -> str:
    """Build the URL for the MedlinePlus Health Topics XML for a given date."""
    if date_str is None:
        # Try today, then go back up to 5 days (files only published Tue-Sat)
        for delta in range(6):
            d = datetime.now() - timedelta(days=delta)
            date_str = d.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/mplus_topics_{date_str}.xml"
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    return url
            except Exception:
                continue
        # Fallback: use the compressed zip
        raise RuntimeError("Could not find a recent MedlinePlus XML file.")
    return f"{BASE_URL}/mplus_topics_{date_str}.xml"


def download_and_parse(url: str) -> dict:
    """Download the MedlinePlus Health Topics XML and extract term→definition pairs."""
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    print(f"  Downloaded {len(resp.content) / 1024:.0f} KB")

    root = ET.fromstring(resp.content)
    glossary = {}

    for topic in root.iter("health-topic"):
        title = topic.get("title", "").strip()
        if not title:
            continue

        # Only keep English topics
        lang = topic.get("language", "English")
        if lang != "English":
            continue

        # Get the full-summary element (contains HTML)
        summary_el = topic.find("full-summary")
        summary = ""
        if summary_el is not None:
            # full-summary can contain mixed content / tail text
            raw = ET.tostring(summary_el, encoding="unicode", method="text")
            if not raw or len(raw.strip()) < 10:
                # Try extracting from inner HTML
                raw = ET.tostring(summary_el, encoding="unicode", method="html")
            summary = re.sub(r"<[^>]+>", " ", raw).strip()
            # Collapse whitespace
            summary = re.sub(r"\s+", " ", summary).strip()
            # Take first 2-3 sentences for a concise definition
            sentences = re.split(r"(?<=[.!?])\s+", summary)
            summary = " ".join(sentences[:3]).strip()

        # Collect "also-called" synonyms
        also_called = []
        for ac in topic.iter("also-called"):
            if ac.text:
                also_called.append(ac.text.strip())

        if summary and len(summary) > 15:
            key = title.lower().strip()
            glossary[key] = summary

            # Index by synonyms too
            for syn in also_called:
                syn_key = syn.lower().strip()
                if syn_key and syn_key not in glossary:
                    glossary[syn_key] = summary

    return glossary


def main():
    parser = argparse.ArgumentParser(
        description="Download MedlinePlus Health Topics into an offline glossary"
    )
    parser.add_argument(
        "--date", default=None,
        help="XML file date (YYYY-MM-DD). If omitted, auto-detects most recent."
    )
    parser.add_argument(
        "--output", default=OUTPUT_PATH,
        help="Output JSON path (default: data/medlineplus_glossary.json)"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    url = get_xml_url(args.date)
    glossary = download_and_parse(url)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    print(f"\nGlossary saved to {args.output}")
    print(f"Total terms: {len(glossary)}")

    # Show some examples
    print("\nSample entries:")
    for key in list(glossary.keys())[:5]:
        print(f"  {key}: {glossary[key][:80]}...")


if __name__ == "__main__":
    main()
