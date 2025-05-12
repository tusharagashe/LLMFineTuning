"""
pdf_to_json.py (test)
--------------
This script helps you turn a PDF into one JSON object per page using the Unstructured.io API.

You can use it directly in Python scripts or notebooks, or just run it from the command line.

Example usage in Python:
>>> from pdf_to_json import pdf_to_json
>>> pdf_to_json("myfile.pdf", strategy="vlm", remove_text="some footer")
--- Not postive the remove_text is working properly BE WARNED!

Parameters
----------
pdf_path : str | pathlib.Path
    Path to the PDF file you want to process.
strategy : {"hi_res", "vlm", "ocr_only"}, default "hi_res"
    Choose how to process the PDF:
    - "hi_res" for high-res layout parsing
    - "vlm" for vision-language model support
    - "ocr_only" if your PDF is just scanned text
remove_text : str | None, optional
    Any text (like headers or footers) that repeats on every page—pass it here to strip it out.
output_path : str | pathlib.Path | None, optional
    Where to save the JSON output. If you don't set this, it'll save as `<pdf>.json` in the same folder.
extract_images : bool, default True
    If True, grabs images from the PDF and includes them as base64-encoded PNGs.

Returns
-------
pathlib.Path
    The path to the final JSON file so you know where to find it.
"""
from __future__ import annotations
import argparse, json, os, re, uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from unstructured_client import UnstructuredClient
from unstructured_client.models.shared import Files, Strategy

# helpers

def _element_to_dict(el: Any) -> Dict[str, Any]:
    """Normalise Unstructured element -> plain dict."""
    if isinstance(el, dict):
        return el
    if hasattr(el, "model_dump"):  # pydantic v2
        return el.model_dump(by_alias=True, mode="python")
    if hasattr(el, "dict"):  # pydantic v1
        return el.dict(by_alias=True)
    import dataclasses

    if dataclasses.is_dataclass(el):
        return dataclasses.asdict(el)
    return json.loads(str(el))


def _strip_header(text: str, header_re: Optional[re.Pattern[str]]) -> str:
    """Remove the first match of *header_re* from *text* (if provided)."""
    if header_re is None:
        return text
    return header_re.sub("", text, count=1).lstrip()


# core

def pdf_to_json(
    pdf_path: str | Path,
    *,
    strategy: str = "hi_res",
    remove_text: str | None = None,
    output_path: str | Path | None = None,
    extract_images: bool = True,
) -> Path:
    """Convert *pdf_path* → JSON file containing one CompositeElement per page."""

    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

    if output_path is None:
        output_path = pdf_path.with_suffix(".json")
    else:
        output_path = Path(output_path).expanduser().resolve()

    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise EnvironmentError("UNSTRUCTURED_API_KEY environment variable not set")

    # Compile header regex (if any)
    header_re: Optional[re.Pattern[str]] = None
    if remove_text:
        header_re = re.compile(remove_text, re.IGNORECASE | re.DOTALL | re.MULTILINE)

    client = UnstructuredClient(api_key_auth=api_key)
    with pdf_path.open("rb") as fh:
        request = {
            "partition_parameters": {
                "files": Files(content=fh, file_name=pdf_path.name),
                "strategy": Strategy(strategy),
                "split_pdf_page": True,
                "split_pdf_allow_failed": True,
                "unique_element_ids": True,
                "extract_images_in_pdf": extract_images,
                "include_text_as_html": True,
                "image_format": "png",
            }
        }
        resp = client.general.partition(request=request)

    raw = getattr(resp, "elements", None) or getattr(resp, "parsed_elements", None)
    if raw is None:
        raise RuntimeError("Unstructured API returned zero elements")

    #  aggregate by page_number in document order 
    pages: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
    for el in raw:
        d = _element_to_dict(el)

        page_no = d.get("metadata", {}).get("page_number")
        if page_no is None:
            # Elements without page_number (kinda rare) → skip
            continue

        if page_no not in pages:
            pages[page_no] = {
                "texts": [],
                "htmls": [],
                "languages": set(),
                "filetype": d.get("metadata", {}).get("filetype"),
                "filename": d.get("metadata", {}).get("filename"),
            }

        # Add text content (with optional header stripping)
        if d.get("text"):
            pages[page_no]["texts"].append(_strip_header(d["text"], header_re))

        # Add HTML content (with optional header stripping)
        html_frag = d.get("metadata", {}).get("text_as_html")
        if html_frag:
            pages[page_no]["htmls"].append(_strip_header(html_frag, header_re))

        langs: Set[str] = set(d.get("metadata", {}).get("languages") or [])
        pages[page_no]["languages"].update(langs)

    # build one CompositeElement per page
    combined: List[Dict[str, Any]] = []
    for page_no, bundle in pages.items():
        combined.append(
            {
                "type": "CompositeElement",
                "element_id": uuid.uuid4().hex,
                "text": "\n".join(bundle["texts"]).strip(),
                "metadata": {
                    "filename": bundle["filename"],
                    "filetype": bundle["filetype"],
                    "languages": sorted(bundle["languages"]),
                    "page_number": page_no,
                    "text_as_html": "<br/>\n".join(bundle["htmls"]).strip(),
                },
            }
        )

    output_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"  Saved {len(combined)} page objects → {output_path}")
    return output_path


#  CLI wrapper 

def _main() -> None:  
    p = argparse.ArgumentParser(description="PDF → one CompositeElement per page.")
    p.add_argument("pdf", type=Path, help="Source PDF file")
    p.add_argument("-o", "--output", type=Path, help="Output JSON (default: <pdf>.json)")
    p.add_argument(
        "--strategy",
        choices=["hi_res", "vlm", "ocr_only"],
        default="hi_res",
        help="Partition strategy (default: hi_res)",
    )
    p.add_argument(
        "--remove-text",
        help="Regex OR literal string that appears on every page and should be removed",
    )
    p.add_argument("--no-images", action="store_true", help="Skip image extraction")
    args = p.parse_args()

    pdf_to_json(
        pdf_path=args.pdf,
        strategy=args.strategy,
        remove_text=args.remove_text,
        output_path=args.output,
        extract_images=not args.no_images,
    )


if __name__ == "__main__":  
    _main()
