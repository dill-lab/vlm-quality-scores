# vqa_infer.py
# This script will use the VLMs to answer questions from the Vizwiz and AOKVQA datasets.

import os
import pandas as pd
import ast
import argparse
import base64
from globals import two_step_prompt, DATASETS_FOLDER
from lm_loader import LMModel, create_model_instance
from utils import check_answers, load_image, check_answers_LAVE
from tqdm import tqdm

# Load models using LMStudio API endpoint
llava_model = create_model_instance("llava-1.5-7b")
qwen_model = create_model_instance("qwen2.5-vl-7b-instruct")
gpt4o_model = create_model_instance("gpt-4o")

def process_row(model, row, dataset_name, verbose=False):
    image_path = os.path.join(DATASETS_FOLDER, dataset_name, row["image_path"])
    # Data URI expects a base64-encoded string
    encoded_image = load_image(image_path)
    
    question = row['question']
    if dataset_name == "AOKVQA":
        correct_answer = row["correct_answer"]
        choices = ast.literal_eval(row['choices'])
        choices_str = ", ".join(choices)
    else: # dataset_name == "VizWiz"
        correct_answer = ast.literal_eval(row["correct_answers"])
        choices_str = ""
    
    prompt_step1 = two_step_prompt[dataset_name]["step1"]["user_prompt"].format(question=question, choices=choices_str)
    system_prompt_step1 = two_step_prompt[dataset_name]["step1"]["system_prompt"]
    messages1 = [
        {"role": "system", "content": system_prompt_step1},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            {"type": "text", "text": prompt_step1}
        ]}
    ]
    response1 = model.chat_completion(messages1, max_tokens=20)
    if response1 is None or 'choices' not in response1 or len(response1['choices']) == 0:
        print(f"Error: No response from model for image {image_path}")
        return "[Error during completion]", "", 0
    answer_label = response1['choices'][0]['message']['content'].strip(" .\n").lower()
    
    prompt_step2 = two_step_prompt[dataset_name]["step2"]["user_prompt"].format(question=question, choices=choices_str, answer=answer_label)
    system_prompt_step2 = two_step_prompt[dataset_name]["step2"]["system_prompt"]
    messages2 = [
        {"role": "system", "content": system_prompt_step2},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
            {"type": "text", "text": prompt_step2}
        ]}
    ]
    response2 = model.chat_completion(messages2, max_tokens=500)
    if response2 is None or 'choices' not in response2 or len(response2['choices']) == 0:
        print(f"Error: No response from model for image {image_path}")
        return "[Error during completion]", "", 0
    rationale = response2['choices'][0]['message']['content'].strip()
    
    # Check the answer against the allowed ones.
    allowed_answers = correct_answer
    if isinstance(allowed_answers, str):
        allowed_answers = [ans.strip() for ans in allowed_answers.split(",")]
        
    if dataset_name == "AOKVQA":
        is_correct = check_answers(answer_label, allowed_answers)
    else: # dataset_name == "VizWiz"
        is_correct = check_answers_LAVE(answer_label, allowed_answers, question)
    
    return rationale, answer_label, is_correct

def answer_question(model, input_dataset, dataset_name, rewrite_file=False, test=False):
    dataset = input_dataset.copy()
    if test:
        dataset = dataset.head(20)
    model_name = str(model)
    output_dir = os.path.join("model_outputs", dataset_name)
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
        if "is_correct" not in dataset_existing.columns:
            dataset_existing["is_correct"] = 0

        error_indices = dataset_existing[dataset_existing["rationale"] == "[Error during completion]"].index
        if not error_indices.empty:
            print(f"Found {len(error_indices)} error rows. Reprocessing them...")
            for i in tqdm(error_indices, desc="Reprocessing error rows"):
                row = dataset.iloc[i]
                rationale, new_pred_ans, new_is_correct = process_row(model, row, dataset_name, verbose=True)
                dataset_existing.at[i, "rationale"] = rationale
                dataset_existing.at[i, "predicted_answer"] = new_pred_ans
                dataset_existing.at[i, "is_correct"] = new_is_correct
            dataset_existing.to_csv(output_filename, index=False)
            print(f"Updated error rows saved to {output_filename}")
        else:
            print(f"No error rows to update in {output_filename}")
    else:
        rationales = []
        predict_answers = []
        is_correct = []
        
        for i, row in tqdm(dataset.iterrows(), total=len(dataset), desc="Processing rows"):
            rationale, new_pred_ans, new_is_correct = process_row(model, row, dataset_name)
            rationales.append(rationale)
            predict_answers.append(new_pred_ans)
            is_correct.append(new_is_correct)
        
        dataset["predicted_answer"] = predict_answers
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
        choices=["AOKVQA", "VizWiz", "both"],
        help="Dataset to process: AOKVQA, VizWiz, or both."
    )
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
    
    args = parser.parse_args()

    # Load datasets based on the dataset argument.
    datasets = {}
    if args.dataset in ["AOKVQA", "both"]:
        datasets["AOKVQA"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "AOKVQA/AOKVQA.csv"))
    if args.dataset in ["VizWiz", "both"]:
        datasets["VizWiz"] = pd.read_csv(os.path.join(DATASETS_FOLDER, "VizWiz/VizWiz.csv"))
    
    # Define available models.
    available_models = {
        "llava": llava_model,
        "qwen": qwen_model,
        "gpt4o": gpt4o_model
    }
    models_to_use = {}
    if args.model == "all":
        models_to_use = available_models
    else:
        models_to_use[args.model] = available_models[args.model]

    # Process each selected dataset with the chosen model(s).
    for dataset_name, dataset in datasets.items():
        for model_name, model in models_to_use.items():
            print(f"Processing {dataset_name} with model {model_name}...")
            answer_question(model, dataset, dataset_name, rewrite_file=args.rewrite_file, test=args.test)

if __name__ == "__main__":
    main()
