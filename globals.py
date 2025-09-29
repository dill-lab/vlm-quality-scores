# globals.py
# This file contains global variables and configurations used across the project.

with open("../OPENAI_key.txt", "r") as file:
    openai_api_key = file.readlines()[2].strip()
    
# Record the cost in this file.
COST_FILE = "../total_cost.txt"

DATASETS_FOLDER = "../datasets"

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
