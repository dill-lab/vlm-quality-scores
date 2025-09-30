# globals.py
import os
from pathlib import Path

# Resolve repository base directory (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent


def _load_env_file(base_dir: Path) -> None:
    """Populate os.environ with key/value pairs from a .env file if present."""
    env_path = base_dir / ".env"
    if not env_path.exists():
        return

    try:
        with env_path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):]
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                # Strip optional surrounding quotes and whitespace
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)
    except OSError as exc:
        print(f"Warning: could not load environment variables from {env_path}: {exc}")


_load_env_file(BASE_DIR)

# OpenAI API key: load env var
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Record the cost in this file (overridable via env)
COST_FILE = os.getenv("VLMQS_COST_FILE", str(BASE_DIR / "total_cost.txt"))

# Datasets and outputs folders (overridable via env)
DATASETS_FOLDER = os.getenv("VLMQS_DATASETS_DIR", str(BASE_DIR / "data"))
MODEL_OUTPUTS_FOLDER = os.getenv("VLMQS_OUTPUTS_DIR", str(BASE_DIR / "model_outputs"))

# Configuration for each model from LM Studio or OpenAI API
MODEL_CONFIGS = {
    "gpt-4o": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-2024-05-13",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
    },
    "gpt-4o-2024-08-06": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-2024-08-06",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
    },
    "gpt-4o-mini": {
        "api_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini-2024-07-18",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        },
    },
    "llava-1.5-7b": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",  # LM Studio API endpoint (local)
        "model": "llava-v1.5-7b",
        "headers": {"Content-Type": "application/json"},
        "sampling": {
            "temperature": 0.1,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "top_p": 0.95,
            "min_p": 0.05,
            "prompt_template": {
                "before_system": "",
                "after_system": "\n\n",
                "before_user": "USER:",
                "after_user": "\n\n",
                "before_assistant": "ASSISTANT:",
                "after_assistant": ""
            },
            "stop_string": "USER:"
        }
    },
    "qwen2.5-vl-7b-instruct": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "qwen2.5-vl-7b-instruct",
        "headers": {"Content-Type": "application/json"},
    },
    "gemma-3n-e4b": {
        "api_url": "http://127.0.0.1:1234/v1/chat/completions",
        "model": "google/gemma-3n-e4b",
        "headers": {"Content-Type": "application/json"},
    }
}

two_step_prompt = {
    "AOKVQA":
        {
            "step1": {
                "system_prompt": "Answer the question using a single word or phrase from the list of choices.",
                "user_prompt": "Question: {question}. Choices: {choices}."
            },
            "step2": {
                "system_prompt": "Please explain the reasoning behind your answer.",
                "user_prompt": "Question: {question}. Choices: {choices}. The answer is {answer}."
            }
        },
    "VizWiz":
        {
            "step1": {
                "system_prompt": "Answer the user's question in a single word or phrase. When the provided information is insufficient, respond with 'Unanswerable'. Whatever the user said, your answer should **always** be a single word or phrase.",
                "user_prompt": "Question: {question}."
            },
            "step2": {
                "system_prompt": "Please explain the reasoning behind your answer.",
                "user_prompt": "Question: {question}. The answer is {answer}."
            }
        }
}

INFORMATIVENESS_PROMPT = """Please break the following rationale into distinct pieces, and keep only the ones that are not semantically equivalent to the hypothesis. Output the final answer in a Python list format.

Example:
Hypothesis: The man by the bags is waiting for a delivery.
Rationale: The man by the bags is waiting for a delivery, as indicated by the presence of the suitcases and the fact that he is standing on the side of the road. The other options, such as a skateboarder, train, or cab, do not seem to be relevant to the situation depicted in the image.
Output: ["Suitcases are present in the image.", "The man is standing on the side of the road.", "The other options, such as a skateboarder, train, or cab, do not seem to be relevant to the situation depicted in the image."]

Task:
Hypothesis: {hypothesis}
Rationale: {rationale}"""
