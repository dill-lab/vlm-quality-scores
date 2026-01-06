import pandas as pd
from support_contrastiveness_analysis import support_contr_analysis_all_datasets
import os

def run_ablation():
    input_dir = "model_outputs/AOKVQA"
    output_dir = "model_outputs/AOKVQA_Reconstructed_Close_Set_Ablation"
    os.makedirs(output_dir, exist_ok=True)

    models = [
        "gpt-4o-2024-05-13.csv",
        "qwen2.5-vl-7b-instruct.csv",
        "llava-v1.5-7b.csv"
    ]

    for model_file in models:
        input_path = os.path.join(input_dir, model_file)
        output_path = os.path.join(output_dir, model_file)
        
        if not os.path.exists(input_path):
            print(f"Skipping {model_file} (not found in {input_dir})")
            continue
            
        print(f"\nProcessing {model_file} for Open-Set Ablation...")
        df = pd.read_csv(input_path)
        
        # Drop existing score columns to force re-calculation
        cols_to_drop = [
            'support', 'entail_prob', 'contrastive', 'contrastiveness_score', 
            'alt_entail_probs', 'hypothesis', 'alternative_hypotheses'
        ]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        # Open-Set Configuration:
        # 1. overwrite_candidate_answers=True: Force generation of new "Hard Negatives" (ignoring existing choices)
        # 2. mask_distractors=False: Do not mask the generated distractors in rationale (since model didn't see them)
        # 3. all_choices_column_name=None: Ensure we don't accidentally use existing choices
        
        support_contr_analysis_all_datasets(
            question_column_name="question",
            answer_column_name="predicted_answer",
            majority_answer_column_name="correct_answer",
            all_choices_column_name=None, # Force generation
            rationale_column_name="rationale",
            dataset_list=[df],
            model_name="gpt-4o", # Use GPT-4o for high-quality negative generation
            same_question_set=False, # Treat files individually just in case
            overwrite_candidate_answers=True,
            overwrite_hypotheses_columns=True, # MUST be True to regenerate alternatives based on new choices
            output_paths=[output_path],
            include_contrastive=True,
            mask_distractors=False 
        )
        
        df.to_csv(output_path, index=False)
        print(f"Completed {model_file}")

if __name__ == "__main__":
    run_ablation()
