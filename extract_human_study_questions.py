# extract_human_study_questions.py
import ast
import pandas as pd
import json
import os
import argparse
import random
from tqdm import tqdm
from lm_loader import create_model_instance
from support_contrastiveness_analysis import integrate_question_answer
import numpy as np

SAMPLE_RUN_LLAVA = None

SAMPLE_RUN_LLAVA = 30       # ← the “best” run picked (AOKVQA)
SAMPLE_SIZE = 100           # ← as in ece_analysis.py

# SAMPLE_RUN_LLAVA = 23       # ← the “best” run picked (VizWiz)
# SAMPLE_SIZE = 100           # ← as in ece_analysis.py

rng = np.random.RandomState(SAMPLE_RUN_LLAVA)

output_folder = "human_study_questions"

model_names = ['llava-v1.5-7b', 'qwen2.5-vl-7b-instruct', 'gpt-4o-2024-05-13']
# datasets_types = ['AOKVQA', 'VizWiz']
datasets_types = ['AOKVQA']
# datasets_types = ['VizWiz']

gpt_4o_mini = create_model_instance("gpt-4o-mini")

def safe_literal_eval(val):
    # Only use ast.literal_eval if the value is a string.
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception as e:
            print(f"Error evaluating {val}: {e}")
            return val
    return val

def format_vf_sentence(question, answer):
    # Clean the inputs
    question = question.strip()
    # answer_clean = answer.strip().lower()
    # We keep the answer to be always "yes" to avoid double negatives
    # THat is because if the instance is from low visual fidelity, the result will be stored in infidel sets.
    answer_clean = "yes"
    descriptive_sentence = integrate_question_answer(question, answer_clean, gpt_4o_mini)
    return "<br> - " + descriptive_sentence

def get_reasons_vf_for_sheet(df):
    reason_vf_correct_list = []
    reason_vf_incorrect_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # reason of high/low visual fidelity
        # show two visual questions and their answers
        # priority to show incorrect answered questions
        visual_questions = safe_literal_eval(row['vf_questions'])
        visual_answers = safe_literal_eval(row['vf_answers'])
        
        # Pair up questions with answers
        qa_pairs = list(zip(visual_questions, visual_answers))
        
        # Prioritize pairs with incorrect answers ("No")
        incorrect_pairs = [pair for pair in qa_pairs if pair[1].strip().lower() == 'no']
        correct_pairs = [pair for pair in qa_pairs if pair[1].strip().lower() != 'no']
        
        # Choose up to two pairs, prioritizing incorrect answers
        selected_incorrect_pairs = incorrect_pairs[:2]
        selected_correct_pairs = correct_pairs[:2]
        
        # Create descriptive sentences for each question-answer pair
        # joined by a line break
        correct_sentences = ""
        for q, a in selected_correct_pairs:
            correct_sentences += format_vf_sentence(q, a)
        
        incorrect_sentences = ""
        for q, a in selected_incorrect_pairs:
            incorrect_sentences += format_vf_sentence(q, a)
        
        reason_vf_correct_list.append(correct_sentences)
        reason_vf_incorrect_list.append(incorrect_sentences)
    
    return reason_vf_correct_list, reason_vf_incorrect_list

def get_reasons_contr_for_sheet(df, choices_column_name):
    reason_contr_list = []
    for idx, row in df.iterrows():
        # Contrastiveness Reasoning ---------------------------------------
        # Parse alternative scores and answer choices
        contr_scores = safe_literal_eval(row['alt_entail_probs'])
        
        # minus the predict answer
        predicted_answer = row['predicted_answer']
        all_choices = ast.literal_eval(row[choices_column_name])
        alt_answers = [choice for choice in all_choices if choice != predicted_answer][:3]
            
        correct_score = row['entail_prob']
        threshold = correct_score / 2
        
        # Select answers with a score at least half of the correct answer's score
        selected_options = []
        if len(alt_answers) != len(contr_scores):
            print(f"Row {idx}: len(alt_answers) = {len(alt_answers)}, len(contr_scores) = {len(contr_scores)}")
            raise ValueError("Length of alternative answers and scores do not match.")
        
        for ans, score in zip(alt_answers, contr_scores):
            if score >= threshold:
                selected_options.append(ans)
        
        if selected_options:
            contr_reason = ", ".join(selected_options)
        else:
            contr_reason = ""
        
        reason_contr_list.append(contr_reason)
    
    return reason_contr_list

def save_selected_rows_to_json(sheet_name, df, choices_column_name):
    df_copy = df.copy()
    combined_questions = []
    combined_random_scores = rng.rand(len(df))
    for idx, row in df_copy.iterrows():
        choices = ast.literal_eval(row[choices_column_name])
        if 'vizwiz' in sheet_name.lower():
            choices = [choice for choice in choices if choice != row['predicted_answer']]
            # select 3 choices, add with row['majority_answer']
            choices = random.sample(choices, 3) + [row['majority_answer']]
            # shuffle the choices
            random.shuffle(choices)
        
        combined_questions.append(f"{row['question']} Choices: {', '.join(choices)}")
    
    df_copy['question'] = combined_questions
    df_copy['uniform_random_score'] = combined_random_scores

    # Define the columns to include and their desired names in JSON
    selected_columns = {
        'index': 'question_id',
        'question': 'question',
        'predicted_answer': 'predicted_answer',
        'is_correct': 'prediction_is_correct',
        'rationale': 'generated_rationale',
        'visual_fidelity': 'visual_fidelity',
        'contrastiveness_score': 'contrastiveness',
        'uniform_random_score': 'uniform_random_score',
        'reason_vf_correct': 'reason_vf_correct',
        'reason_vf_incorrect': 'reason_vf_incorrect',
        'reason_contr': 'reason_contr'
    }

    # Check if all required columns exist
    missing_columns = [col for col in selected_columns.keys() if col not in df_copy.columns]
    if missing_columns:
        raise KeyError(f"The following required columns are missing in the DataFrame of sheet '{sheet_name}': {missing_columns}")

    # Rename columns as per the desired JSON structure
    renamed_df = df_copy[list(selected_columns.keys())].rename(columns=selected_columns)
    
    # convert index to string
    renamed_df['question_id'] = renamed_df['question_id'].astype(str)
    
    # fill NaN values with empty string
    renamed_df = renamed_df.fillna('')

    # Convert to dictionary
    sheet_data = renamed_df.to_dict(orient='records')

    # Generate a valid file name by replacing spaces with underscores
    file_name = f"{sheet_name.replace(' ', '_').lower()}.json"
    save_path = f"{output_folder}/{file_name}"
    
    if SAMPLE_RUN_LLAVA is not None:
        save_path = f"{output_folder}/{file_name[:-5]}_sample{SAMPLE_RUN_LLAVA}.json"

    # Save to a separate JSON file
    with open(save_path, 'w') as file:
        json.dump(sheet_data, file, indent=4)  # Added indent for better readability

    print(f"Data from sheet '{sheet_name}' has been successfully saved to '{save_path}'.")
    
def store_human_study_questions(sheet_name, df, choices_column_name, indice_list = None):
    # Select equal number of correct and incorrect answers (if we did not preselect the indices)
    if indice_list is None:
        # Select equal number of correct and incorrect answers
        max_half_len = 100
        # filter out the rows contained refusal explanations: I'm sorry / I am sorry / I can't / I cannot / I'm unable / I am unable / I apologize
        refused_expls = ['I\'m sorry', 'I am sorry', 'I can\'t', 'I cannot', 'I\'m unable', 'I am unable', 'I apologize']
        df['predicted_answer'] = df['predicted_answer'].str.lower().str.strip('*\n ')
        
        df = df[~df['rationale'].str.contains('|'.join(refused_expls), case=False)]
        # min_row_num = min(df['is_correct'].value_counts().min(), 100)
        # Consider only the rows with is_correct = 0 or 1
        min_row_num = min(df[df['is_correct'].isin([0, 1])]['is_correct'].value_counts().min(), 100)
        
        print(f"Number of correct answers: {df['is_correct'].value_counts()[1]}")
        print(f"Number of incorrect answers: {df['is_correct'].value_counts()[0]}")
        print(f"Selecting {min_row_num} correct and incorrect answers...")
        
        correct_rows = df[df['is_correct'] == 1].sample(n=min_row_num, random_state=42)
        incorrect_rows = df[df['is_correct'] == 0].sample(n=min_row_num, random_state=42)
        selected_rows = pd.concat([correct_rows, incorrect_rows])    
    else:
        selected_rows = df.loc[indice_list]
        
    # Save the selected rows to a new json file
    save_selected_rows_to_json(sheet_name, selected_rows, choices_column_name)

def main():
    parser = argparse.ArgumentParser(
        description="Extract human study questions from model outputs."
    )
    parser.add_argument(
        "--rewrite_files",
        action='store_true',
        help="Whether to rewrite the existing human study question files."
    )
    args = parser.parse_args()
    rewrite_files = args.rewrite_files
    
    for dataset_type in datasets_types:
        for model_name in model_names:
            # Load the dataset
            dataset = pd.read_csv(f"model_outputs/{dataset_type}/{model_name}.csv")

            choices_column_name = 'choices' if dataset_type == 'AOKVQA' else 'generated_choices'
            
            if SAMPLE_RUN_LLAVA is not None and model_name == "llava-v1.5-7b":
                # reproduce exactly the sampling you did in ece_analysis.py
                per_class = SAMPLE_SIZE // 2
                pos_df = dataset[dataset.is_correct == 1]
                neg_df = dataset[dataset.is_correct == 0]

                # sample with the same random_state
                pos_inds = pos_df.sample(n=per_class, random_state=SAMPLE_RUN_LLAVA).index
                neg_inds = neg_df.sample(n=per_class, random_state=SAMPLE_RUN_LLAVA).index

                # combine them to a single list
                indice_list = list(pos_inds) + list(neg_inds)

                print(f"Using SAMPLE_RUN={SAMPLE_RUN_LLAVA}: total indices = {len(indice_list)}")
                store_human_study_questions(
                    f"{model_name}_{dataset_type}", 
                    dataset, 
                    choices_column_name, 
                    indice_list=indice_list
                )

            # if 'reason_vf_correct' not in dataset.columns or rewrite_files:
            #     # Get the reasons for high/low visual fidelity
            #     reasons_vf_correct, reasons_vf_incorrect = get_reasons_vf_for_sheet(dataset)
            #     reason_contr_list = get_reasons_contr_for_sheet(dataset, choices_column_name)
                
            #     # Add the reasons to the dataset
            #     dataset['reason_vf_correct'] = reasons_vf_correct
            #     dataset['reason_vf_incorrect'] = reasons_vf_incorrect
            #     dataset['reason_contr'] = reason_contr_list
                
            #     # Save the dataset
            #     dataset.to_csv(f"model_outputs/{dataset_type}/{model_name}.csv", index=False)
            #     print(f"Saved model_outputs/{dataset_type}/{model_name}.csv")
            
            # llava_aokvqa_indice_file_path = "human_study_questions/llava-1.5_question_ids.json"

            # if model_name == "llava-v1.5-7b" and dataset_type == "AOKVQA":
            #     # We specifically use the indices from the llava-1.5 dataset
            #     # As that dataset is used for the prior human study (where random indice selections then were diffrerent)
            #     # To ensure that the human study questions are the same as the prior study, we load the indices from the prior study
            #     indice_list = pd.read_json(llava_aokvqa_indice_file_path).values.flatten().tolist()
            #     # Generate the human study questions in json format
            #     store_human_study_questions(f"{model_name}_{dataset_type}", dataset, choices_column_name, indice_list=indice_list)
            # else:
            #     # Generate the human study questions in json format
            #     store_human_study_questions(f"{model_name}_{dataset_type}", dataset, choices_column_name)
            
            
if __name__ == '__main__':
    main()
    
        