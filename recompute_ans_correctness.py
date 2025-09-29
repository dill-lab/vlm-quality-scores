# recompute_ans_correctness.py: recompute the answer's correctness of the model's prediction on the VizWiz dataset.

import os
import pandas as pd
import ast
from tqdm import tqdm
from utils import check_answers_LAVE

dataset_folder_path = "model_outputs/VizWiz"
models = ["llava-v1.5-7b", "qwen2.5-vl-7b-instruct", "gpt-4o-2024-05-13"]
debug = True

def recompute_acc(model_name, dataset_folder_path):
    print(f"Recomputing is_correct for model {model_name}...")
    df = pd.read_csv(os.path.join(dataset_folder_path, f"{model_name}.csv"))
    
    is_correct_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        correct_answers = ast.literal_eval(row["correct_answers"])
        predicted_answer = row["predicted_answer"]
        question = row["question"]
        
        is_correct = check_answers_LAVE(predicted_answer, correct_answers, question)
        is_correct_list.append(is_correct)
        
    df["is_correct"] = is_correct_list
    df.to_csv(os.path.join(dataset_folder_path, f"{model_name}.csv"), index=False)
    
def main():
    for model_name in models:
        recompute_acc(model_name, dataset_folder_path)
    
if __name__ == "__main__":
    main()
        