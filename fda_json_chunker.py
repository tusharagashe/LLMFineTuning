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
    print(f"✅ Saved {len(chunks)} chunks to {output_path}")