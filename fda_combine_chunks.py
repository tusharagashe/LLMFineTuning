#!/usr/bin/env python3
"""
combine_chunks.py
--------------------------

Merge smaller text chunks from a ClinicalTrials.gov-style chunked JSON file into larger,
more semantically meaningful segments. Useful for improving embedding quality in vector
databases by avoiding overly short entries.

This script processes a list of JSON chunks (each with "text" and "metadata"), groups
them together until a minimum word count threshold is met, and aggregates metadata fields.

Typical use case:
Run this **after** `fda_json_chunker.py` to produce final chunks ready for embedding.

Usage (CLI):
------------
$ python combine_chunks.py trial_file.chunks.json --min_words 100
$ python combine_chunks.py trial_file.chunks.json --output trial_file.combined.json

Arguments:
----------
input_path     (str)  : Path to the input chunked JSON file (from fda_json_chunker.py)
--output       (str)  : Optional path to save the combined JSON file (default: <input>.combined.json)
--min_words    (int)  : Minimum number of words per combined chunk (default: 100)
--max_words    (int)  : Maximum number of words (currently unused, placeholder for future logic)

Output:
-------
A JSON file where each item has:
- "text": merged chunk of one or more text fields
- "metadata": list of fields merged into this chunk, with their original JSON paths

Notes:
------
- Chunks with word counts ≥ `min_words` are included directly.
- Smaller chunks are buffered and grouped together until they exceed the minimum.
"""

import json
import argparse
from pathlib import Path

def count_words(text):
    return len(text.split())

def combine_chunks(input_path, output_path, target_min=100, target_max=200):
    with open(input_path, "r") as f:
        data = json.load(f)

    formatted_chunks = []
    buffer_text = ""
    buffer_fields = []
    buffer_word_count = 0

    for item in data:
        text = item["text"]
        field = item["metadata"]["field"]
        word_count = count_words(text)

        if word_count >= target_min:
            if buffer_text:
                formatted_chunks.append({
                    "text": buffer_text.strip(),
                    "metadata": [{"field": ",".join(buffer_fields)}]
                })
                buffer_text = ""
                buffer_fields = []
                buffer_word_count = 0
            formatted_chunks.append({
                "text": text.strip(),
                "metadata": [{"field": field}]
            })
        else:
            buffer_text += " " + text
            buffer_fields.append(field)
            buffer_word_count += word_count
            if buffer_word_count >= target_min:
                formatted_chunks.append({
                    "text": buffer_text.strip(),
                    "metadata": [{"field": ",".join(buffer_fields)}]
                })
                buffer_text = ""
                buffer_fields = []
                buffer_word_count = 0

    if buffer_text:
        formatted_chunks.append({
            "text": buffer_text.strip(),
            "metadata": [{"field": ",".join(buffer_fields)}]
        })

    with open(output_path, "w") as f:
        json.dump(formatted_chunks, f, indent=2)

    print(f"✅ Combined chunks saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine small text chunks in a JSON file into larger chunks with flattened metadata.")
    parser.add_argument("input_path", type=str, help="Path to input JSON file")
    parser.add_argument("--output", type=str, help="Path to save output JSON file (default: <input>.combined.json)")
    parser.add_argument("--min_words", type=int, default=100, help="Minimum number of words per chunk")
    parser.add_argument("--max_words", type=int, default=200, help="Maximum number of words per chunk (currently unused)")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output or Path(input_path).with_suffix(".combined.json")

    combine_chunks(input_path, output_path, args.min_words, args.max_words)
