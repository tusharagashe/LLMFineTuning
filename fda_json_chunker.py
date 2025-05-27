#!/usr/bin/env python3
"""
chunk_json_file.py
--------------------------

Extract and segment semantic text chunks from a ClinicalTrials.gov-style JSON file.

This script reads structured clinical trial records organized into nested sections and 
subsections (e.g., protocol, results, annotations), flattens the content into text entries, 
and splits large entries into smaller chunks using LangChain’s RecursiveCharacterTextSplitter.

Typical use case:
Run this **before** `combine_chunks.py` to produce manageable, semantically relevant text 
segments for embedding into a vector database or other NLP processing pipelines.

Usage (CLI):
------------
$ python chunk_json_file.py trial_file.json
$ python chunk_json_file.py trial_file.json --chunk_size 800 --chunk_overlap 100 --output trial_file.chunks.json

Arguments:
----------
json_file      (str)  : Path to the ClinicalTrials.gov-style JSON file.
--chunk_size   (int)  : Maximum character length of each chunk (default: 1000).
--chunk_overlap(int)  : Number of overlapping characters between adjacent chunks (default: 100).
--output       (str)  : Optional path to save the chunked JSON file (default: <input>.chunks.json)

Output:
-------
A JSON file where each item has:
- "text": content extracted from the trial sections
- "metadata": {"field": <flattened path of origin in JSON>}

Notes:
------
- Uses a section map to traverse key parts of the clinical trial structure.
- Text fields shorter than the chunk size are left intact.
- Larger text blocks are recursively split while preserving metadata.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

SECTION_MAP = {
    "protocolSection": [
        "identificationModule", "statusModule", "sponsorCollaboratorsModule", "oversightModule",
        "descriptionModule", "conditionsModule", "designModule", "armsInterventionsModule",
        "outcomesModule", "eligibilityModule", "contactsLocationsModule", "referencesModule",
        "ipdSharingStatementModule"
    ],
    "resultsSection": [
        "participantFlowModule", "baselineCharacteristicsModule", "outcomeMeasuresModule",
        "adverseEventsModule", "moreInfoModule"
    ],
    "annotationSection": ["annotationModule"],
    "documentSection": ["largeDocumentModule"],
    "derivedSection": [
        "miscInfoModule", "conditionBrowseModule", "interventionBrowseModule"
    ]
}

def is_large_text(value) -> bool:
    return isinstance(value, str) and len(value.split()) > 100

def flatten_sections(section: Dict, field_prefix: str = "") -> List[Dict[str, str]]:
    chunks = []

    for key, value in section.items():
        current_field = f"{field_prefix}.{key}" if field_prefix else key
        if isinstance(value, dict):
            chunks.extend(flatten_sections(value, current_field))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    chunks.extend(flatten_sections(item, f"{current_field}[{i}]"))
                else:
                    chunks.append({"text": str(item), "metadata": {"field": f"{current_field}[{i}]"}})
        else:
            chunks.append({"text": str(value), "metadata": {"field": current_field}})

    return chunks

def chunk_json_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Dict[str, str]]:
    print(f"Reading file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    print("Successfully loaded JSON data")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for top_section, sub_sections in SECTION_MAP.items():
        print(f"Processing section: {top_section}")
        section_data = data.get(top_section, {})
        if not section_data:
            print(f"No data found for section: {top_section}")
            continue

        for sub_section in sub_sections:
            print(f"Processing subsection: {sub_section}")
            sub_data = section_data.get(sub_section, {})
            if sub_data:
                flat_entries = flatten_sections(sub_data, f"{top_section}.{sub_section}")
                for entry in flat_entries:
                    if is_large_text(entry["text"]):
                        split_texts = text_splitter.split_text(entry["text"])
                        for t in split_texts:
                            all_chunks.append({"text": t, "metadata": entry["metadata"]})
                    else:
                        all_chunks.append(entry)

    if "hasResults" in data:
        all_chunks.append({
            "text": f"hasResults: {data['hasResults']}",
            "metadata": {"field": "hasResults"}
        })

    print(f"Generated {len(all_chunks)} chunks")
    return all_chunks

def save_chunks_to_json(chunks: List[Dict[str, str]], output_path: str):
    print(f"Saving chunks to: {output_path}")
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    print(f"Successfully saved chunks")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk ClinicalTrials.gov JSON into semantic units.")
    parser.add_argument("json_file", type=str, help="Path to ClinicalTrials.gov-style .json file")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Max chunk size (characters)")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap (characters)")
    parser.add_argument("--output", type=str, help="Optional output .json path")

    args = parser.parse_args()
    input_path = args.json_file
    output_path = args.output or Path(input_path).with_suffix(".chunks.json")

    print(f"Starting processing of {input_path}")
    chunks = chunk_json_file(input_path, args.chunk_size, args.chunk_overlap)
    save_chunks_to_json(chunks, output_path)
    print(f" Saved {len(chunks)} chunks to {output_path}")