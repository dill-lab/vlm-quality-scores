# load_dataset.py
# This script will load the Vizwiz dataset and the AOKVQA dataset (both should be validation first 500),
# download the images to a local folder, and store a .csv file containing the question, choices,
# correct_answer, and local image_path for each question in the dataset.

import io
import os
import pandas as pd
import numpy as np
import argparse
from datasets import (
    load_dataset,
    DatasetDict,
    IterableDatasetDict,
)
from PIL import Image
from collections import Counter
from lm_loader import create_model_instance
from utils import load_image
from tqdm import tqdm
from globals import DATASETS_FOLDER

# Define the base folder path using globals
base_folder = DATASETS_FOLDER


def _get_dataset_split(dataset, split_name: str):
    """Return the requested split for both dict and split objects."""
    if isinstance(dataset, (dict, DatasetDict, IterableDatasetDict)):
        try:
            return dataset[split_name]
        except KeyError:
            return dataset
    return dataset


def _ensure_list(value):
    """Best-effort conversion of array-like objects to Python lists."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    return []


def _load_image_from_field(image_field):
    """Return a PIL Image from a dataset image field when possible."""
    if image_field is None:
        return None
    if isinstance(image_field, Image.Image):
        return image_field.copy()
    if isinstance(image_field, dict):
        image_bytes = image_field.get("bytes")
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
        image_path = image_field.get("path")
        if image_path and os.path.isfile(image_path):
            return Image.open(image_path)
    return None

def process_and_save_aokvqa(dataset, folder_path: str, dataset_name: str):
    """
    Process the first 500 examples of the validation split for AOKVQA,
    download the images, and save a CSV with the question, choices,
    correct_answer, and local image_path.
    """
    split = "validation"
    split_dataset = _get_dataset_split(dataset, split)
    target_count = 500

    os.makedirs(folder_path, exist_ok=True)
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    processed_samples = []
    seen_rows = 0
    with tqdm(total=target_count, desc="Processing Rows") as progress_bar:
        for sample_idx, row in enumerate(split_dataset):
            seen_rows += 1
            # Extract fields using common keys
            question = row.get("question", "")
            choices = _ensure_list(row.get("choices"))

            correct_choice_idx = row.get("correct_choice_idx")
            if (
                not choices
                or correct_choice_idx is None
                or not (0 <= correct_choice_idx < len(choices))
            ):
                continue

            image = None
            try:
                image = _load_image_from_field(row.get("image"))
                if image is None:
                    continue
                image.load()
                local_image_path = os.path.join(images_folder, f"{sample_idx}.jpg")
                image.save(local_image_path)
            except Exception as exc:
                print(f"Image data not valid for entry {sample_idx}: {exc}")
                continue
            finally:
                if image is not None:
                    image.close()

            relative_image_path = os.path.relpath(local_image_path, start=folder_path)
            processed_samples.append({
                "index": sample_idx,
                "question": question,
                "choices": choices,
                "correct_answer": choices[correct_choice_idx],
                "image_path": relative_image_path
            })

            progress_bar.update(1)
            if len(processed_samples) >= target_count:
                break

    if not processed_samples:
        print("No valid entries were collected for AOKVQA.")
        return

    print(f"Processed {seen_rows} rows to collect {len(processed_samples)} AOKVQA samples.")
    processed_df = pd.DataFrame(processed_samples)
    csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
    processed_df.to_csv(csv_path, index=False)
    print(f"Saved {len(processed_df)} entries to {csv_path}")
    

def process_and_save_vizwiz(dataset, folder_path: str, dataset_name: str):
    """
    Process the 600 answerable examples of the val split for VizWiz,
    download the images, and save a CSV with the question_id, question,
    correct_answers, category, and local image_path.
    """
    split = "val"
    split_dataset = _get_dataset_split(dataset, split)
    target_count = 600

    os.makedirs(folder_path, exist_ok=True)
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    processed_samples = []
    considered_rows = 0
    with tqdm(total=target_count, desc="Processing Rows") as progress_bar:
        for sample_idx, row in enumerate(split_dataset):
            considered_rows += 1
            category = row.get("category", "")
            if category == "unanswerable":
                continue

            question = row.get("question", "")
            if not question or not question.endswith("?"):
                continue

            answers = _ensure_list(row.get("answers"))
            if not answers or "unanswerable" in answers:
                continue

            majority_answer = Counter(answers).most_common(1)[0][0]
            if majority_answer in {"yes", "no"} and category != "yes/no":
                continue

            max_size = (800, 800)
            image = None
            try:
                image = _load_image_from_field(row.get("image"))
                if image is None:
                    continue
                image.load()
                if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                    image.thumbnail(max_size, resample=Image.Resampling.LANCZOS)

                local_image_path = os.path.join(images_folder, f"{sample_idx}.jpg")
                image.save(local_image_path)
            except Exception as exc:
                print(f"Image data not valid for entry {sample_idx}: {exc}")
                continue
            finally:
                if image is not None:
                    image.close()

            relative_image_path = os.path.relpath(local_image_path, start=folder_path)
            unique_answers = list(Counter(answers).keys())

            processed_samples.append({
                "index": sample_idx,
                "question_id": row.get("question_id", ""),
                "question": question,
                "correct_answers": unique_answers,
                "majority_answer": majority_answer,
                "category": category,
                "image_path": relative_image_path
            })

            progress_bar.update(1)
            if len(processed_samples) >= target_count:
                break

    if not processed_samples:
        print("No valid entries were collected for VizWiz.")
        return

    print(f"Processed {considered_rows} rows to collect {len(processed_samples)} VizWiz samples.")
    processed_df = pd.DataFrame(processed_samples)
    csv_path = os.path.join(folder_path, f"{dataset_name}.csv")
    # reset index column
    processed_df.reset_index(drop=True, inplace=True)
    processed_df["index"] = processed_df.index
    processed_df.to_csv(csv_path, index=False)
    print(f"Saved {len(processed_df)} entries to {csv_path}")
    filter_out_inappropriate_questions(processed_df, folder_path, remove_indices=False, verbose=True)
    
def filter_out_inappropriate_questions(df, csv_folder_path, remove_indices=False, verbose=False):
    """
    Filter out inappropriate questions from the dataset.
    """
    model = create_model_instance("gpt-4o-2024-08-06")
    
    inappropriate_indices = []
    
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Filtering Inappropriate Questions"):
        question = row.get("question", "")
        answer = row.get("correct_answer", "")
        image_path = os.path.join(csv_folder_path, row.get("image_path", ""))
        
        encoded_image = load_image(image_path)
        
        user_prompt = (
            "Does this question, image, or answer contain inappropriate (NSFW) content? "
            "Respond with 'yes' or 'no', along with a short justification if possible (indicate which part is inappropriate)."
            f"Question: {question}. Answer: {answer}."
        )
        
        messages = [
            {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {"type": "text", "text": user_prompt}
            ]}
        ]
        
        response = model.chat_completion(messages)
        try:
            content = response['choices'][0]['message']['content'].strip(" .\n")
            if "yes" in content.lower():
                inappropriate_indices.append(i)
                if verbose:
                    print(f"Question {i} is inappropriate. The question is: {question}")
                    print(f"Reason: {content}")
        except Exception as e:
            print(f"Error: {e}")
            
    print(f"Found {len(inappropriate_indices)} inappropriate questions.")
    print(f"Indices of inappropriate questions: {inappropriate_indices}")
    if remove_indices:
        print("Removing inappropriate questions...")
        df.drop(inappropriate_indices, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Remaining questions: {df.shape[0]}")
    


def main():
    parser = argparse.ArgumentParser(description="Process datasets for VizWiz and AOKVQA.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["AOKVQA", "VizWiz", "both"],
        help="Name of the dataset to process (AOKVQA, VizWiz, or both)."
    )
    args = parser.parse_args()
    
    if args.dataset in ["AOKVQA", "both"]:
        # Stream the validation split of AOKVQA so we can stop once enough examples are gathered.
        dataset_aok = load_dataset(
            "HuggingFaceM4/A-OKVQA",
            split="validation",
            streaming=True,
        )
        process_and_save_aokvqa(dataset_aok, os.path.join(base_folder, "AOKVQA"), "AOKVQA")
    if args.dataset in ["VizWiz", "both"]:
        # Stream the val split of VizWiz and stop downloading once sufficient examples are collected.
        dataset_viz = load_dataset(
            "lmms-lab/VizWiz-VQA",
            split="val",
            streaming=True,
        )
        process_and_save_vizwiz(dataset_viz, os.path.join(base_folder, "VizWiz"), "VizWiz")

if __name__ == "__main__":
    main()
