# extract_and_format_prompt.py

import json
from pathlib import Path
from pdf_to_json import pdf_to_json
from openai import OpenAI
import sys

def get_llm_response(prompt: str) -> dict:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Extract the clinical trial structure."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
    )
    content = response.choices[0].message.content
    print("Response content:", content)  # Log the response content
    # Strip triple backticks and 'json' label if present
    content = content.strip('`').replace('json', '').strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        raise

def build_prompt(text: str) -> str:
    return f"""
Extract the following fields from the clinical trial text:
- mechanism
- biomarker
- endpoint (split into Co-primary and Secondary)
- safety

Use this format:
{{
    "mechanism": "...",
    "biomarker": "...",
    "endpoint": {{
        "Co-primary": "...",
        "Secondary": "..."
    }},
    "safety": "..."
}}

Clinical Trial Text:
{text}
"""

def generate_user_prompt(pdf_path: str) -> dict:
    json_path = pdf_to_json(pdf_path, strategy="vlm", extract_images=False)

    with open(json_path, "r") as f:
        pages = json.load(f)

    structured_fields = get_llm_response(build_prompt(pages))

    user_prompt = {
        "user_proposal": pages,
        "mechanism": structured_fields["mechanism"],
        "biomarker": structured_fields["biomarker"],
        "endpoint": (
            f"Co-primary: {structured_fields['endpoint']['Co-primary']} "
            f"Secondary: {structured_fields['endpoint']['Secondary']}"
        ),
        "safety": structured_fields["safety"],
        "iteration_count": 0,
    }

    return user_prompt

if __name__ == "__main__":
    input_pdf = sys.argv[1]
    user_prompt = generate_user_prompt(input_pdf)
    print(json.dumps(user_prompt, indent=2))