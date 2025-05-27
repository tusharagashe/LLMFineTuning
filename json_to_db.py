#!/usr/bin/env python3
"""
new_json_to_db.py
--------------------
Flatten an Unstructured‑style JSON array (one‑level: top + metadata) into
a single SQLite table called `elements`.

Usage
-----
$ python new_json_to_db.py source.json  -o output.db
"""

from __future__ import annotations
import argparse, json, os, sqlite3
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def flatten(record: Dict[str, Any]) -> Dict[str, Any]:
    """Pull `metadata` keys up one level; JSON‑encode lists/dicts."""
    rec = record.copy()
    meta = rec.pop("metadata", {})
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            v = json.dumps(v, ensure_ascii=False)
        rec[k] = v
    return rec


def json_to_db(json_path: Path, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()            # start fresh

    with json_path.open(encoding="utf-8") as fp:
        data: List[Dict[str, Any]] = json.load(fp)

    rows = [flatten(r) for r in data]
    df = pd.DataFrame(rows)

    with sqlite3.connect(db_path) as conn:
        df.to_sql("elements", conn, index=False)

    print(f"  Wrote {len(df)} rows and {len(df.columns)} columns → {db_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Flatten JSON → SQLite")
    p.add_argument("json_path", type=Path, help="Source .json file")
    p.add_argument(
        "-o", "--output", type=str, default="output.db",
        help="Destination .db file name (saved in ./postchunks/)",
    )
    args = p.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path("postchunks")
    output_dir.mkdir(exist_ok=True)

    db_path = output_dir / args.output
    json_to_db(args.json_path, db_path)


if __name__ == "__main__":
    main()
