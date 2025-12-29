# vqa_infer.py
# This script will use the VLMs to answer questions from the Vizwiz and AOKVQA datasets.

import os
import pandas as pd
import ast
import argparse
import base64
import re
from globals import two_step_prompt, single_step_prompt, answer_extraction_prompt, DATASETS_FOLDER, MODEL_OUTPUTS_FOLDER
from lm_loader import LMModel, create_model_instance
from utils import check_answers, load_image, check_answers_LAVE
from tqdm import tqdm

def process_row(model, row, dataset_name, strategy="two-step", parser_model=None, verbose=False):
    image_path = os.path.join(DATASETS_FOLDER, dataset_name, row["image_path"])
    # Data URI expects a base64-encoded string
    encoded_image = load_image(image_path)

    question = row['question']
    if dataset_name in ["AOKVQA", "MMMU-Pro", "MMMU-Pro-4"]:
        correct_answer = row["correct_answer"]
        choices = ast.literal_eval(row['choices'])
        choices_str = ", ".join(choices)  # Plain format for two-step
        if dataset_name in ["MMMU-Pro", "MMMU-Pro-4"]:
            choices_with_letters = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        else:
            choices_with_letters = choices_str  # For AOKVQA, same as plain
    else: # dataset_name == "VizWiz"
        correct_answer = ast.literal_eval(row["correct_answers"])
        choices_str = ""
    
    if strategy == "two-step":
        prompt_step1 = two_step_prompt[dataset_name]["step1"]["user_prompt"].format(question=question, choices=choices_str)
        system_prompt_step1 = two_step_prompt[dataset_name]["step1"]["system_prompt"]
        
        # Check if model supports system role
        if not model.model_config.get("supports_system_role", True):
            prompt_step1 = f"{system_prompt_step1}\n\n{prompt_step1}"
            messages1 = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt_step1}
                ]}
            ]
        else:
            messages1 = [
                {"role": "system", "content": system_prompt_step1},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt_step1}
                ]}
            ]
        # MMMU-Pro has longer answer choices, so need more tokens
        max_tokens_step1 = 100 if dataset_name in ["MMMU-Pro", "MMMU-Pro-4"] else 20
        response1 = model.chat_completion(messages1, max_tokens=max_tokens_step1)
        if response1 is None or 'choices' not in response1 or len(response1['choices']) == 0:
            print(f"Error: No response from model for image {image_path}")
            return "[Error during completion]", "", "", 0
        answer_label = response1['choices'][0]['message']['content'].strip(" .\n").lower()
        
        prompt_step2 = two_step_prompt[dataset_name]["step2"]["user_prompt"].format(question=question, choices=choices_str, answer=answer_label)
        system_prompt_step2 = two_step_prompt[dataset_name]["step2"]["system_prompt"]
        
        if not model.model_config.get("supports_system_role", True):
            prompt_step2 = f"{system_prompt_step2}\n\n{prompt_step2}"
            messages2 = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt_step2}
                ]}
            ]
        else:
            messages2 = [
                {"role": "system", "content": system_prompt_step2},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt_step2}
                ]}
            ]
        response2 = model.chat_completion(messages2, max_tokens=4096)
        if response2 is None or 'choices' not in response2 or len(response2['choices']) == 0:
            print(f"Error: No response from model for image {image_path}")
            return "[Error during completion]", "", "", 0
        rationale = response2['choices'][0]['message']['content'].strip()

    elif strategy == "single-step":
        if dataset_name not in single_step_prompt:
            raise ValueError(f"Single-step strategy not supported for dataset {dataset_name}")
        
        prompt = single_step_prompt[dataset_name]["user_prompt"].format(question=question, choices_with_letters=choices_str)
        system_prompt = single_step_prompt[dataset_name].get("system_prompt", "")
        
        if not model.model_config.get("supports_system_role", True):
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            messages = [
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt}
                ]}
            ]
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                    {"type": "text", "text": prompt}
                ]}
            )
        response = model.chat_completion(messages, max_tokens=4096)
        if response is None or 'choices' not in response or len(response['choices']) == 0:
            print(f"Error: No response from model for image {image_path}")
            return "[Error during completion]", "", "", 0
        rationale = response['choices'][0]['message']['content'].strip()
        
        # Parse the answer letter from the rationale
        answer_letter = ""
        answer_label = ""

        # 1. parse out the "Answer: " from the rationale
        parsed_answer = rationale.split("Answer:")[-1].strip(" .")

        # 2. Check if the remaining matches any of the choices text
        for idx, choice in enumerate(choices):
            if choice.lower() == parsed_answer.lower():
                answer_label = choice
                answer_letter = chr(65 + idx)
                break
        
        # 3. If not, see if the remaining matches any of the choices letter
        if not answer_letter:
            # Remove common punctuation that might surround the letter (e.g., "(A)", "A.")
            cleaned_letter = parsed_answer.strip("().")
            if len(cleaned_letter) == 1:
                letter_idx = ord(cleaned_letter.upper()) - 65
                if 0 <= letter_idx < len(choices):
                    answer_letter = cleaned_letter.upper()
                    answer_label = choices[letter_idx]
        
        # Fallback to parser model if regex fails (optional, but good for robustness)
        if not answer_letter and parser_model:
            # print for debug purposes, rationale, match, answerlabel, etc
            print(f"Regex failed for image {image_path}, using parser model")
        
            parse_prompt = answer_extraction_prompt["user_prompt"].format(question=question, choices_with_letters=choices_with_letters, model_response=rationale)
            parse_system_prompt = answer_extraction_prompt["system_prompt"]
            parse_messages = [
                {"role": "system", "content": parse_system_prompt},
                {"role": "user", "content": parse_prompt}
            ]
            parse_response = parser_model.chat_completion(parse_messages, max_tokens=1)
            print(f"Parser response: {parse_response}")
            if parse_response and 'choices' in parse_response and len(parse_response['choices']) > 0:
                answer_letter = parse_response['choices'][0]['message']['content'].strip(" .\n")
                # Try to see if parser output is just a letter or the full text
                print(f"Answer letter: {answer_letter}")
                # If it's a letter, map it
                if len(answer_letter) == 1 and answer_letter.upper() in [chr(65+i) for i in range(len(choices))]:
                    answer_letter = answer_letter.upper()
                    answer_label = choices[ord(answer_letter) - 65]
                # If it's full text, try to find which letter it corresponds to
                elif answer_letter in choices:
                    answer_letter = chr(65 + choices.index(answer_letter))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Check the answer against the allowed ones.
    allowed_answers = correct_answer
    if isinstance(allowed_answers, str):
        allowed_answers = [ans.strip() for ans in allowed_answers.split(",")]

    if dataset_name in ["AOKVQA", "MMMU-Pro", "MMMU-Pro-4"]:
        is_correct = check_answers(answer_label, allowed_answers)
    else: # dataset_name == "VizWiz"
        is_correct = check_answers_LAVE(answer_label, allowed_answers, question)
    
    print(f"Answer_label: {answer_label}")
    print(f"Answer_letter: {answer_letter}")
    print(f"Is_correct: {is_correct}")
    return rationale, answer_label, answer_letter, is_correct

def answer_question(model, input_dataset, dataset_name, rewrite_file=False, test=False, strategy="two-step", parser_model=None):
    dataset = input_dataset.copy()
    if test:
        dataset = dataset.head(20)
    model_name = str(model)
    output_dir = os.path.join(MODEL_OUTPUTS_FOLDER, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    if test:
        output_filename = os.path.join(output_dir, f"{model_name}_test.csv")
    else:
        output_filename = os.path.join(output_dir, f"{model_name}.csv")
    
    if os.path.exists(output_filename) and not rewrite_file:
        # If the file exists and we're not rewriting, load it and only process rows with error messages.
        dataset_existing = pd.read_csv(output_filename)
        # Initialize new columns if not present.
        if "rationale" not in dataset_existing.columns:
            dataset_existing["rationale"] = ""
        if "predicted_answer" not in dataset_existing.columns:
            dataset_existing["predicted_answer"] = ""
        if "predicted_answer_letter" not in dataset_existing.columns:
            dataset_existing["predicted_answer_letter"] = ""
        if "is_correct" not in dataset_existing.columns:
            dataset_existing["is_correct"] = 0

        error_indices = dataset_existing[dataset_existing["rationale"] == "[Error during completion]"].index
        if not error_indices.empty:
            print(f"Found {len(error_indices)} error rows. Reprocessing them...")
            for i in tqdm(error_indices, desc="Reprocessing error rows"):
                row = dataset.iloc[i]
                rationale, new_pred_ans, new_pred_letter, new_is_correct = process_row(model, row, dataset_name, strategy=strategy, parser_model=parser_model, verbose=True)
                dataset_existing.at[i, "rationale"] = rationale
                dataset_existing.at[i, "predicted_answer"] = new_pred_ans
                dataset_existing.at[i, "predicted_answer_letter"] = new_pred_letter
                dataset_existing.at[i, "is_correct"] = new_is_correct
            dataset_existing.to_csv(output_filename, index=False)
            print(f"Updated error rows saved to {output_filename}")
        else:
            print(f"No error rows to update in {output_filename}")
    else:
        rationales = []
        predict_answers = []
        predict_letters = []
        is_correct = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing rows"):
            rationale, new_pred_ans, new_pred_letter, new_is_correct = process_row(model, row, dataset_name, strategy=strategy, parser_model=parser_model)
            rationales.append(rationale)
            predict_answers.append(new_pred_ans)
            predict_letters.append(new_pred_letter)
            is_correct.append(new_is_correct)
        
        dataset["predicted_answer"] = predict_answers
        dataset["predicted_answer_letter"] = predict_letters
        dataset["is_correct"] = is_correct
        dataset["rationale"] = rationales
        
        dataset.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Infer answers for VQA datasets using various VLMs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["AOKVQA", "VizWiz", "MMMU-Pro", "MMMU-Pro-4", "all"],
        help="Dataset to process: AOKVQA, VizWiz, MMMU-Pro, MMMU-Pro-4, or all."
    )
    # Model 
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["llava", "qwen", "gpt4o", "all"],
        help="Model to use: llava, qwen, gpt4o, or all."
    )
    parser.add_argument(
        "--rewrite_file",
        action="store_true",
        help="If set, rewrite existing output files."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set, run the first 20 rows of the dataset for testing."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="two-step",
        choices=["two-step", "single-step"],
        help="Evaluation strategy: two-step (default) or single-step."
    )
    parser.add_argument(
        "--parser_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for parsing the answer in single-step strategy (default: gpt-4o-mini)."
    )

    args = parser.parse_args()

    # Load datasets based on the dataset argument.
    datasets = {}
    if args.dataset in ["AOKVQA", "all"]:
        datasets["AOKVQA"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "AOKVQA/AOKVQA.csv"))
    if args.dataset in ["VizWiz", "all"]:
        datasets["VizWiz"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "VizWiz/VizWiz.csv"))
    if args.dataset in ["MMMU-Pro", "all"]:
        datasets["MMMU-Pro"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "MMMU-Pro/MMMU-Pro.csv"))
    if args.dataset in ["MMMU-Pro-4", "all"]:
        datasets["MMMU-Pro-4"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "MMMU-Pro-4/MMMU-Pro-4.csv"))
    
    # Define available models.
    available_models = {
        "llava": create_model_instance("llava-1.5-7b"),
        "qwen": create_model_instance("qwen2.5-vl-7b-instruct"),
        "gpt4o": create_model_instance("gpt-4o")
    }
    models_to_use = {}
    if args.model == "all":
        models_to_use = available_models
    else:
        models_to_use[args.model] = available_models[args.model]

    # Load parser model if needed
    parser_model = None
    if args.strategy == "single-step":
        parser_model = create_model_instance(args.parser_model)

    # Process each selected dataset with the chosen model(s).
    for dataset_name, dataset in datasets.items():
        for model_name, model in models_to_use.items():
            print(f"Processing {dataset_name} with model {model_name}...")
            answer_question(model, dataset, dataset_name, rewrite_file=args.rewrite_file, test=args.test, strategy=args.strategy, parser_model=parser_model)

if __name__ == "__main__":
    main()
