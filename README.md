# LLM Fine Tuning Data Processing Pipeline

This repository contains a set of tools for processing various document formats (PDF, CSV) into structured data suitable for LLM fine-tuning and embedding in vector databases.

## Workflow Options

### 1. PDF Processing Pipeline
```
PDF → pdf_to_json.py → json_to_db.py → Ready for embedding
```
This workflow requires an Unstructured.io API key and is ideal for processing PDF documents.

### 2. FDA/Clinical Trials Data Pipeline
```
FDA JSON → fda_json_chunker.py → fda_combine_chunks.py → Ready for embedding
```
Specialized pipeline for processing FDA and ClinicalTrials.gov JSON data.

### 3. Direct Unstructured.io Pipeline
```
PDF/CSV → Partition + chunk (Unstructured.io) → json_to_db.py → Ready for embedding
```
Alternative workflow using Unstructured.io's direct partitioning capabilities.

## Components

### pdf_to_json.py
Converts PDFs into structured JSON format using the Unstructured.io partition API.
- Creates one CompositeElement JSON object per page
- Supports different processing strategies (hi_res, vlm, ocr_only)
- Can extract embedded images (as base64 PNG strings)
- Handles header/footer removal

```bash
python pdf_to_json.py myfile.pdf --strategy hi_res --remove-text "footer text"
```

### json_to_db.py
Converts Unstructured-style JSON arrays into SQLite database format.
- Flattens JSON structure into a single table
- Preserves metadata
- Handles nested JSON structures

```bash
python json_to_db.py source.json -o output.db
```

### fda_json_chunker.py
Processes FDA/ClinicalTrials.gov JSON files into semantic chunks for vector embedding.
- Recursively flattens JSON structure
- Creates semantically meaningful text units
- Configurable chunk sizes and overlap
- Preserves document structure metadata

```bash
python fda_json_chunker.py trial_file.json --chunk_size 1000 --chunk_overlap 100
```

### fda_combine_chunks.py
Merges smaller text chunks into larger, more meaningful segments.
- Improves embedding quality by avoiding short entries
- Configurable minimum/maximum word counts
- Aggregates metadata from combined chunks
- Maintains semantic relationships

```bash
python fda_combine_chunks.py trial_file.chunks.json --min_words 100
```

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export UNSTRUCTURED_API_KEY="your_api_key_here"
```

## Notes
- When using interchangeably between pipelines, consider removing headers like "is_continuation" and "data_source" before embedding
- Choose the appropriate workflow based on your input data format and processing needs
- Adjust chunk sizes and overlap parameters based on your specific use case

## Requirements
- Python 3.7+
- Unstructured.io API key (for PDF processing)
- Dependencies listed in requirements.txt

## License
text me
