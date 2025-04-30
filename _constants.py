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
