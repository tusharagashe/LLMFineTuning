"""
json_to_db.py 
Convert an Unstructured‑style JSON file into a compact SQLite database while preserving every field.
Use pdf_to_json.py first to convert a PDF into a JSON file.

The input JSON must be an *array* of objects that look like:

{
  "type": "CompositeElement",
  "element_id": "…",
  "text": "…",
  "metadata": {
      "filename": "…",
      "filetype": "…",
      "languages": ["eng"],
      "page_number": 1,
      "text_as_html": "…"
  }
}

Each record is flattened into eight columns:

    type, element_id, text,
    filename, filetype, languages, page_number, text_as_html

Usage
-----
$ python json_to_db.py path/to/file.json  -o out.db
"""

from __future__ import annotations
import argparse
import json
import sqlite3
import sys
from pathlib import Path


def flatten(entry: dict) -> dict:
    """Pull metadata fields up one level so we can insert a simple row."""
    meta = entry.get("metadata", {})
    return {
        "type": entry.get("type"),
        "element_id": entry.get("element_id"),
        "text": entry.get("text"),
        "filename": meta.get("filename"),
        "filetype": meta.get("filetype"),
        # store the language list as a JSON string to keep the array intact
        "languages": json.dumps(meta.get("languages", [])),
        "page_number": meta.get("page_number"),
        "text_as_html": meta.get("text_as_html", ""),
    }


DDL = """
CREATE TABLE IF NOT EXISTS elements (
    element_id    TEXT PRIMARY KEY,
    type          TEXT,
    text          TEXT,
    filename      TEXT,
    filetype      TEXT,
    languages     TEXT,
    page_number   INTEGER,
    text_as_html  TEXT
);
"""


def json_to_db(json_path: Path, db_path: Path) -> None:
    with json_path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        sys.exit("❌  Top‑level JSON is not an array; aborting.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript(DDL)

    rows = [flatten(rec) for rec in data]
    cur.executemany(
        """
        INSERT OR REPLACE INTO elements
        (element_id, type, text, filename, filetype, languages, page_number, text_as_html)
        VALUES (:element_id, :type, :text, :filename, :filetype, :languages, :page_number, :text_as_html)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    print(f"✅  Wrote {len(rows)} rows to {db_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON → SQLite .db")
    parser.add_argument("json_path", type=Path, help="Path to source .json file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="DB",
        default=Path("output.db"),
        help="Destination .db file (default: ./output.db)",
    )
    args = parser.parse_args()
    json_to_db(args.json_path, args.output)


if __name__ == "__main__":
    main()