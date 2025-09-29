# load_dataset.py
# This script will load the Vizwiz dataset and the AOKVQA dataset (both should be validation first 500),
# download the images to a local folder, and store a .csv file containing the question, choices,
# correct_answer, and local image_path for each question in the dataset.

import io
import os
import pandas as pd
import numpy as np
import argparse
import ast
from datasets import load_dataset
from PIL import Image
from collections import Counter
from lm_loader import LMModel, create_model_instance
from utils import load_image
from tqdm import tqdm

# Define the base folder path relative to the working directory
base_folder = "../datasets"

def process_and_save_aokvqa(dataset, folder_path: str, dataset_name: str):
    """
    Process the first 500 examples of the validation split for AOKVQA,
    download the images, and save a CSV with the question, choices,
    correct_answer, and local image_path.
    """
    # Select the validation split if available; otherwise, use the train split.
    split = "validation"
    df = dataset[split].to_pandas().head(500)

    # Create folder to store the CSV and images.
    os.makedirs(folder_path, exist_ok=True)
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    processed_samples = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        # Extract fields using common keys
        question = row.get("question", "")
        choices = row.get("choices", "").tolist()
            
        correct_answer = choices[row.get("correct_choice_idx")]
        
        # Get the image field
        image_data = row.get("image", {})
        image_bytes = image_data.get("bytes")
        
        if image_bytes:
            # Create an Image object from the byte data
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save the image to the local folder
            local_image_path = os.path.join(images_folder, f"{i}.jpg")
            image.save(local_image_path)
            
            # Store the relative path in the CSV
            relative_image_path = os.path.relpath(local_image_path, start=folder_path)
            processed_samples.append({
                "index": i,
                "question": question,
                "choices": choices,
                "correct_answer": correct_answer,
                "image_path": relative_image_path
            })
        else:
            print(f"Image data not found for entry {i}")

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
    # Select the val split
    split = "val"
    df = dataset[split].to_pandas()
    # print dataset size
    print(f"Original dataset size: {df.shape[0]}")
    # Filter out the unanswerable categories
    df.drop(df[df["category"] == "unanswerable"].index, inplace=True)
    # drop instances where the question is not a question (end with '?')
    df = df[df['question'].apply(lambda x: x[-1] == '?')]
    
    # filter out if there is presence of 'unanswerable' in the row['answers']
    df = df[~df['answers'].apply(lambda x: 'unanswerable' in x)]
    
    # filter out the majority answer is "yes" or "no" but the category of the question is not "yes/no"
    # e.g. Index = 19, "Is this picture clear, and where are we now?" => majority answer is "yes" but the category is "other" 
    df['majority_answer'] = df['answers'].apply(lambda x: Counter(x).most_common(1)[0][0])
    df = df[~((df['majority_answer'] == 'yes') | (df['majority_answer'] == 'no')) | (df['category'] == 'yes/no')]
    
    # print dataset size
    print(f"Filtered dataset size: {df.shape[0]}")
    print(f"Keep the first 600 answerable examples...")
    
    # Select the first 600 answerable examples (leave room for deleting NSFW questions/answers) (finally, we expect to have 500)
    df = df.head(600)

    # sort by index
    df = df.sort_index()

    # Create folder to store the CSV and images.
    os.makedirs(folder_path, exist_ok=True)
    images_folder = os.path.join(folder_path, "images")
    os.makedirs(images_folder, exist_ok=True)

    processed_samples = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        # Extract fields using common keys; adjust if the actual dataset keys differ.
        question_id = row.get("question_id", "")
        question = row.get("question", "")
        answers = row.get("answers", []).tolist()
        majority_answer = row.get("majority_answer", "")
        
        # Drop duplicates in answers
        answers = list(Counter(answers).keys())
        
        # Similar to https://nlp.stanford.edu/helm/vhelm_lite/?group=viz_wiz&subgroup=&runSpecs=%5B%22viz_wiz%3Amodel%3Dopenai_gpt-4o-2024-05-13%22%5D
        # We consider the answers present in the dataset as the allowed answers.
                
        category = row.get("category", "")
        
        # Get the image field
        image_data = row.get("image", {})
        image_bytes = image_data.get("bytes")
        max_size = (800, 800)
        
        if image_bytes:
            # Create an Image object from the byte data
            image = Image.open(io.BytesIO(image_bytes))
            
            # Compress the image if it is too large
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, resample=Image.Resampling.LANCZOS)
            
            # Save the image to the local folder
            local_image_path = os.path.join(images_folder, f"{i}.jpg")
            image.save(local_image_path)
            
            # Store the relative path in the CSV
            relative_image_path = os.path.relpath(local_image_path, start=folder_path)
            processed_samples.append({
                "index": i,
                "question_id": question_id,
                "question": question,
                "correct_answers": answers,
                "majority_answer": majority_answer,
                "category": category,
                "image_path": relative_image_path
            })
        else:
            print(f"Image data not found for entry {i}")

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
        # Load AOKVQA dataset from Hugging Face.
        dataset_aok = load_dataset("HuggingFaceM4/A-OKVQA")
        process_and_save_aokvqa(dataset_aok, os.path.join(base_folder, "AOKVQA"), "AOKVQA")
    if args.dataset in ["VizWiz", "both"]:
        # Load VizWiz dataset from Hugging Face.
        dataset_viz = load_dataset("lmms-lab/VizWiz-VQA")
        process_and_save_vizwiz(dataset_viz, os.path.join(base_folder, "VizWiz"), "VizWiz")

if __name__ == "__main__":
    main()
