import os

# File Paths
MODELS_DIR = "models/"

# LLM Configuration
# LLM_CONFIGS = {
#     "config_list": [
#         {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
#         {"model": "llama", "api_key": os.getenv("LLAMA_API_KEY")},
#         {"model": "nvidia", "api_key": os.getenv("NVIDIA_API_KEY")},
#     ]
# }

LLM_CONFIGS = {
    "open_ai": {"model": "gpt-4o-mini", "api_key": os.getenv("OPENAI_API_KEY")},
    "llama3.2": {"model": "llama3.2", "api_key": os.getenv("LLAMA_API_KEY")},
    "nvidia": {"model": "gpt-4o-mini", "api_key": os.getenv("NVIDIA_API_KEY")},
}

SYSTEM_MESSAGES = {
    "risk_assessment": "You are a FDA regulatory reviewer. Provide a risk \
                        assessment of the user submission and detail specific flaws in the application\
                            that would not pass regulatory approval. Provide an overall summary and a rating \
                                from 1 to 10.",
    "risk_critiquer": "You are an expert regulatory critiquer. \
                                Given the risk assessment and rating, propose \
                                    changes and explain why to make these changes to improve the original proposal.\
                                    Also provide an alternative approach for how they can rework their proposal to a different population or other strategies you can think of for better success \
                                        Give the final result in bullet points summary for each of these categories. \
                                        1) Mechanistic Risk 2) Biomarker Asessment 3) Endpoint Alignment 4) Safety.",
    "proposal_writer": "You are an expert FDA regulatory writer. \
                                Given the feedback for approval, rewrite the original proposal with the appropriate changes. \
                                    Provide a 200-word new proposal well-written incorporating the changes",
}

USER_PROMPT = "This proposed FDA label describes Respilimab, a humanized monoclonal antibody \
    targeting IL-13, for treatment of moderate-to-severe eosinophilic asthma. It outlines dosing \
        (300 mg subcutaneous every 4 weeks), safety data, and trial outcomes showing improved lung \
            function and reduced exacerbations, supporting use in patients uncontrolled on standard inhaled therapies."
