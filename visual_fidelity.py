import pandas as pd
import base64
import requests
import ast
import os
from tqdm import tqdm
from lm_loader import LMModel, create_model_instance
from globals import DATASETS_FOLDER
from utils import load_image

def gpt_gen_vf_questions(row, rationale_column_name, model, cost_verbose=0):
    system_prompt = f"""You will be shown a question about an image, along with an answer, and a rationale that explains the answer based on details from the image. Your task is to generate a list of yes/no questions that verify the details about the image that are **explicitly** mentioned in the rationale. Your questions should be phrased such that the answer to that question being yes means that the detail in the rationale is correct. Focus on creating questions that can be visually verified or refuted based on the details provided in the rationale. Ensure the questions are specific and directly pertain to aspects that are visually relevant and mentioned in the rationale. Avoid generating questions about elements that are not mentioned in the rationale, or the rationale explicitly states are not relevant or present. Also avoid generating multiple questions that check for the same visual detail.

Here is one example:
Input: 
Question: Why is the person wearing a helmet?
Answer: For safety
Rationale: The person is wearing a helmet because they are riding a bicycle on a busy city street. Helmets are commonly used to protect against head injuries in case of accidents, especially in areas with heavy traffic.

Good Questions:
1. Is the person wearing a helmet while riding a bicycle?
Reason: This question is directly answerable by observing whether the person on the bicycle is wearing a helmet in the image. 
2. Is the street in the image busy with traffic?
Reason: This question can be visually verified by looking at the amount of traffic on the street in the image.

Bad Questions:
1. Is the person wearing the helmet because they are concerned about head injuries?
Reason: This question is not good because it assumes the person’s intentions or concerns, which cannot be visually verified from the image.
2. Does wearing a helmet suggest that the person is highly safety-conscious?
Reason: This question relies on inference and external knowledge about the person’s mindset, rather than on observable details from the image.
3. Is there any indication that the person is wearing a helmet for safety reasons?
Reason: This question verifies the answer to the original question, rather than verifying a detail about the image that's mentioned in the rationale.
4. Is the person wearing a safety vest?
Reason: This question is not good because it tries to verify details about the image that are not explicitly mentioned in the rationale.
5. Is the person not wearing sunglasses?
Reason: This question is not good because it asks for verification by absence and can only be answered with a "no," which is not the preferred type of question.

Respond with a list of (good) questions (without the reasons), starting from '1. '"""
    

    user_input = f"""Question: {row['question']}
Answer: {row['predicted_answer']}
Rationale: {row[rationale_column_name]}"""

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": system_prompt}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": user_input}
        ]}
    ]
    
    response = model.chat_completion(messages)

    try:
        content = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        content = []
        print(f"Error: {e}")
        
    # content is a list of questions, separated by a newline character
    # 1. ... \n 2. ... \n 3. ...

    try:
        # Split the content into individual questions and return a python list
        if '\n' in content:
            parts = content.split('\n')
        else:
            # parts = content.split('. ')  # Split by ". " as a fallback
            parts = [content]
        questions = []

        for part in parts:
            try:
                # Attempt to split and take the second part
                question = part.split('. ')[1]
                questions.append(question)
            except IndexError:
                # If there's an issue with splitting, add the entire part or handle as needed
                questions.append(part)
    except Exception as e:
        questions = [content]
        print(f"Error: {e}")

    return questions

def answer_vf_questions(questions, dataset_name, model, image_path):
    answer_list = []
    image_path = os.path.join(DATASETS_FOLDER, dataset_name, image_path)
    encoded_image = load_image(image_path)
    for question in questions:
        user_input = f"""Question: {question}. Based on the information provided in the image, answer with 'yes' or 'no'. Provide one-word answer only."""
        messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }},
                {"type": "text", "text": user_input}
            ]}
        ]
        response = model.chat_completion(messages)
        try:
            content = response['choices'][0]['message']['content'].strip(" .\n").lower()
        except Exception as e:
            content = ""
            print(f"Error: {e}")
        answer_list.append(content)
    return answer_list


def VF_analysis_all_datasets(rationale_column_name: str, dataset_list: list[pd.DataFrame], dataset_type: str, 
                             model_name: str = "gpt-4o-2024-08-06", overwrite_VF_questions: bool = False):    
    print(f"Running VF analysis using verifier model {model_name}.")
    model = create_model_instance(model_name)
    if model_name == "gpt-4o-2024-08-06":
        vf_q_col     = f"vf_questions"
        vf_a_col     = f"vf_answers"
        vf_score_col = f"visual_fidelity"
    else:
        suffix = model_name
        vf_q_col     = f"vf_questions_{suffix}"
        vf_a_col     = f"vf_answers_{suffix}"
        vf_score_col = f"visual_fidelity_{suffix}"
    
    for i, dataset in enumerate(dataset_list):
        print(f"Working on dataset #{i+1}...")
        
        if vf_q_col not in dataset.columns or overwrite_VF_questions:
            vf_questions = []
            for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Processing Rows"):
                # Generate visual verification questions
                questions = gpt_gen_vf_questions(row, rationale_column_name, model)
                # store the questions as a string (but appear like a list)
                vf_questions.append(str(questions))
            # Assign new column values
            dataset[vf_q_col] = vf_questions
        else:
            print(f"Found {vf_q_col} in dataset's columns (and overwrite flag is set to false), skipping..")
        if vf_a_col not in dataset.columns or overwrite_VF_questions:
            for index, row in tqdm(dataset.iterrows(), total=dataset.shape[0], desc="Answering Questions"):
                # Answer visual verification questions
                questions = ast.literal_eval(row[vf_q_col])
                answers = answer_vf_questions(questions, dataset_type, model, row['image_path'])
                dataset.at[index, vf_a_col] = str(answers)
                vf_score = (sum(1 for ans in answers if ans == 'yes') / len(answers)) if len(answers) != 0 else 0
                dataset.at[index, vf_score_col] = vf_score
        else:
            print(f"Found {vf_a_col} in dataset's columns (and overwrite flag is set to false), skipping..")
            
            
if __name__ == '__main__':
    datasets_folder = "model_outputs"
    model_list = ["llava-v1.5-7b", "qwen2.5-vl-7b-instruct", "gpt-4o-2024-05-13"]
    datasets_types = ["AOKVQA", "VizWiz"]
    # model_list = ['llava-v1.5-7b']
    # datasets_types = ["AOKVQA"]
    vf_eval_model_list = [
                        #   'gpt-4o-2024-08-06', 
                          'qwen2.5-vl-7b-instruct', 
                          'gemma-3n-e4b'
                          ]
    for dataset_type in datasets_types:
        dataset_list = []
        for model in model_list:
            dataset_list.append(pd.read_csv(f"{datasets_folder}/{dataset_type}/{model}.csv"))
        for vf_eval_model in vf_eval_model_list:
            VF_analysis_all_datasets("rationale", dataset_list, dataset_type, model_name=vf_eval_model)
        # Store the updated datasets
        for i, dataset in enumerate(dataset_list):
            dataset.to_csv(f"{datasets_folder}/{dataset_type}/{model_list[i]}.csv", index=False)
        
