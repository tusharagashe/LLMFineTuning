import os
import json
import io
import PyPDF2
from pathlib import Path
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

def parse_pdf_document(file_path: str) -> dict:
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        document_text = ""
        for page in pdf_reader.pages:
            document_text += page.extract_text()
                                
        print(f"Successfully extracted {len(document_text)} characters")
        return {"text": document_text, "success": True}

def format_document_with_llm(document_text: str, user_query: str) -> dict:
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        base_url="http://localhost:11434"  
    )
        
    # PROMPT
    prompt = f"""You are an expert at analyzing clinical trial documents and queries.
    Your task is to extract and format information from the following document and query into a structured format.
        
    Document:
    {document_text}
        
    User Query:
    {user_query}
        
    Please extract and format the information into the following structure. Be detailed and comprehensive:
    {{
        "user_proposal": "A detailed summary of the clinical trial proposal including:
            - Trial phase and design
            - Treatment arms and dosing
            - Sample size and randomization
            - Study duration and follow-up
            - Key inclusion/exclusion criteria
            - Primary objectives",
        "mechanism": "Detailed description of:
            - Drug's mechanism of action
            - Target pathway
            - Molecular interactions
            - Preclinical rationale",
        "biomarker": "Comprehensive list of:
            - Primary biomarkers
            - Secondary biomarkers
            - Exploratory biomarkers
            - Biomarker measurement methods
            - Biomarker validation status",
        "endpoint": "Detailed specification of:
            - Primary endpoints with definitions
            - Secondary endpoints with definitions
            - Exploratory endpoints
            - Endpoint measurement methods
            - Endpoint validation status
            - Statistical analysis plan",
        "indication": "Complete description of:
            - Target disease/condition
            - Disease severity/stage
            - Patient population
            - Prior treatment requirements
            - Disease-specific criteria",
        "safety": "Comprehensive safety considerations:
            - Known safety concerns
            - Risk factors
            - Safety monitoring plan
            - Adverse event collection
            - Safety stopping rules
            - Risk mitigation strategies",
        "iteration_count": 0
    }}
        
    For each field:
    1. Be explicit and detailed
    2. Include all relevant information
    3. Maintain scientific accuracy
    4. Use clear, professional language
    5. Format as a single, well-structured string
        
    Return ONLY the JSON structure, no additional text."""
        
    response = llm.invoke([
        SystemMessage(content="You are an expert at analyzing clinical trial documents and formatting them into structured data. Be concise but comprehensive."),
        HumanMessage(content=prompt)
    ])
        
    formatted_data = json.loads(response.content)
    return formatted_data
            
def main():
    if len(sys.argv) != 3:
        print("Usage: python document_processor.py <pdf_file_path> <query>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    query = sys.argv[2]
    
    result = parse_pdf_document(pdf_path)
    if not result["success"]:
        print("Failed to parse PDF document")
        sys.exit(1)
        
    formatted_data = format_document_with_llm(result["text"], query)
    print(json.dumps(formatted_data, indent=2))

if __name__ == "__main__":
    main() 