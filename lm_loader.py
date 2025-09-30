# lm_loader.py
# This script will define an LMModel class to interact with OpenAI's Language Model API.
# It will also provide a load_model function to load a model configuration and create an LMModel instance.
# The chat_completion method will send a chat request to the model's API endpoint and return the response.

import requests
import json
import time
import os
from globals import MODEL_CONFIGS, COST_FILE
import base64

class LMModel:
    def __init__(self, model_config: dict):
        self.model_config = model_config # Save full config including sampling
        self.api_url = model_config["api_url"]
        self.model = model_config["model"]
        self.headers = model_config.get("headers", {"Content-Type": "application/json"})
        
    # model name
    def __str__(self):
        return self.model
    
    def chat_completion(self, 
                        messages,
                        temperature=0.1,
                        max_tokens=1024,
                        max_retries=3,
                        stream=False):
        """
        Sends a chat request to the model's API endpoint.
        Mimics the openai.ChatCompletion.create interface.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        # Merge in sampling parameters if available
        if "sampling" in self.model_config:
            sampling = self.model_config["sampling"]
            for key, value in sampling.items():
                if key not in payload:
                    payload[key] = value
                    
        retry_count = 0
        while retry_count < max_retries:
            try:
                # If streaming is enabled, pass stream=True to requests.post
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(payload), stream=stream)
                if response.status_code == 200:
                    if stream:
                        # Process streaming response token-by-token
                        return response  # Return the raw streaming response
                    else:
                        # Non-streaming: parse the JSON response immediately
                        result = response.json()
                        
                        # If this is a GPT-4o call and usage info is provided, record the cost.
                        if self.model.startswith("gpt-4o") and "usage" in result:
                            calculate_cost(result["usage"], self.model)
                        return result
                else:
                    print(f"Request failed with status code {response.status_code}: {response.text}")
                    retry_count += 1
                    time.sleep(1)
            except Exception as e:
                print(f"Request failed: {e}")
                retry_count += 1
                time.sleep(1)
        
        print(f"Request failed after {max_retries} attempts.")
        return None

def create_model_instance(model_name: str) -> LMModel:
    """
    Loads a model configuration and returns an LMModel instance.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} is not configured.")
    config = MODEL_CONFIGS[model_name]
    return LMModel(config)

def stream_chat_with_model(user_prompt: str,
                           system_prompt: str = "",
                           model_name: str = "gpt-4o",
                           image_path: str | None = None,
                           temperature: float = 0.0,
                           max_retries: int = 1,
                           max_tokens: int = 512):
    """
    Sends the given user prompt to the specified model with a default system prompt "0".
    Streaming is enabled, and the function prints each token as it's received.
    """
    model = create_model_instance(model_name)
    if image_path is None:
        raise ValueError("image_path must be provided for streaming chat with image support")
    encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
                {
                    "type": "text",
                    "text": user_prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ],
        }
    ]
    
    try:
        response = model.chat_completion(messages, temperature=temperature, max_retries=max_retries, max_tokens=max_tokens, stream=True)
        output_msg = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(delta, end="", flush=True)
                        output_msg += delta
                    except json.JSONDecodeError:
                        continue
        print()  # Newline for clean output
    except Exception as e:
        print("Error during streaming:", e)


def read_total_cost():
    if os.path.exists(COST_FILE):
        with open(COST_FILE, "r") as file:
            content = file.read().strip()
            return float(content) if content != "" else 0.0
    else:
        return 0.0

def write_total_cost(cost):
    prev_cost = read_total_cost()
    new_total_cost = prev_cost + cost
    with open(COST_FILE, "w") as file:
        file.write(f"{new_total_cost}")

def calculate_cost(usage, model, verbose=0):
    """
    Calculate the cost based on the usage dictionary returned by the API and the model name.
    """
    if model == "gpt-4o-2024-05-13":
        input_cost_per_token = 0.005 / 1000
        output_cost_per_token = 0.015 / 1000
    elif model == "gpt-4o-2024-08-06":
        input_cost_per_token = 0.0025 / 1000
        output_cost_per_token = 0.010 / 1000
    elif model == "gpt-4o-mini-2024-07-18":
        input_cost_per_token = 0.00015 / 1000
        output_cost_per_token = 0.00060 / 1000
    else:
        # If model doesn't match any GPT-4o pricing, assume zero cost.
        input_cost_per_token = 0.0
        output_cost_per_token = 0.0
        print(f"Warning: Model {model} not found in pricing table. Assuming zero cost.")

    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    cost = (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)
    if verbose:
        print(f"The cost incurred is ${cost:.3f}")
    write_total_cost(cost)


if __name__ == "__main__":
    # Example usage
    user_prompt = "Question: What color are these pants?"
    system_prompt = "Answer the user's question in a single word or phrase. When the provided information is insufficient, respond with 'Unanswerable'. Whatever the user said, your answer should **always** be a single word or phrase."
    # image_path = "data/VizWiz/images/1385.jpg"
    image_path = os.path.join("data", "VizWiz", "images", "1385.jpg")
    stream_chat_with_model(user_prompt, system_prompt, model_name="gpt-4o", image_path=image_path, max_retries=1)
    stream_chat_with_model(user_prompt, system_prompt, model_name="qwen2.5-vl-7b-instruct", image_path=image_path, max_retries=1)