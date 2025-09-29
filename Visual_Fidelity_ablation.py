"""
Visual Fidelity Ablation Study

This module implements an ablation study for visual fidelity evaluation by systematically 
varying the question generation and verification models while keeping target VLMs fixed.

Design:
- Fixed target VLMs: Qwen2.5-VL-7B and LLaVA-1.5-7B (whose explanations we score)
- Variable components:
  - m_QGen (question decomposition): {GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B}  
  - m_Verif (yes/no verification): {GPT-4o, Qwen2.5-VL-7B-Instruct, Gemma-3n-E4B}
- Datasets: A-OKVQA and VizWiz (500 items each)
"""

import pandas as pd
import os
from itertools import product
from tqdm import tqdm
import ast
import json
from lm_loader import create_model_instance
from globals import DATASETS_FOLDER
from utils import load_image
from Visual_Fidelity_analysis import gpt_gen_vf_questions, answer_vf_questions


class VFAblationStudy:
    """Visual Fidelity Ablation Study Manager"""
    
    def __init__(self):
        # Fixed target VLMs whose explanations we will evaluate
        self.target_vlms = ["qwen2.5-vl-7b-instruct", "llava-v1.5-7b"]
        
        # Models available for ablation components
        self.ablation_models = ["gpt-4o-2024-08-06", "qwen2.5-vl-7b-instruct", "gemma-3n-e4b"]
        
        # Datasets to evaluate on
        self.datasets = ["AOKVQA", "VizWiz"]
        
        # Core ablation configurations (4 key combinations)
        self.core_configs = [
            ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06"),  # (4o, 4o) baseline
            ("gpt-4o-2024-08-06", "qwen2.5-vl-7b-instruct"),  # (4o, Qwen) swap verifier only  
            ("qwen2.5-vl-7b-instruct", "gpt-4o-2024-08-06"),  # (Qwen, 4o) swap generator only
            ("qwen2.5-vl-7b-instruct", "qwen2.5-vl-7b-instruct"),  # (Qwen, Qwen) both swapped
        ]
        
        # Optional: Full grid for comprehensive analysis
        self.full_configs = list(product(self.ablation_models, self.ablation_models))
        
    def generate_questions_with_model(self, row, rationale_column_name, model_name):
        """Generate VF questions using specified model"""
        model = create_model_instance(model_name)
        return gpt_gen_vf_questions(row, rationale_column_name, model)
    
    def verify_questions_with_model(self, questions, dataset_name, model_name, image_path):
        """Answer VF questions using specified model"""
        model = create_model_instance(model_name)
        return answer_vf_questions(questions, dataset_name, model, image_path)
    
    def get_column_suffix(self, qgen_model, verif_model):
        """Generate column suffix for ablation configuration"""
        qgen_short = qgen_model.replace("gpt-4o-2024-08-06", "4o").replace("qwen2.5-vl-7b-instruct", "qwen").replace("gemma-3n-e4b", "gemma")
        verif_short = verif_model.replace("gpt-4o-2024-08-06", "4o").replace("qwen2.5-vl-7b-instruct", "qwen").replace("gemma-3n-e4b", "gemma")
        return f"{qgen_short}_{verif_short}"
    
    def run_ablation_config(self, dataset_df, dataset_name, target_vlm, 
                          qgen_model, verif_model, rationale_column="rationale"):
        """Run single ablation configuration"""
        
        suffix = self.get_column_suffix(qgen_model, verif_model)
        vf_q_col = f"vf_questions_{suffix}"
        vf_a_col = f"vf_answers_{suffix}"  
        vf_score_col = f"visual_fidelity_{suffix}"
        
        print(f"Running config: QGen={qgen_model.split('-')[-1]}, Verif={verif_model.split('-')[-1]} on {target_vlm}")
        
        # Step 1: Generate questions if not exists
        if vf_q_col not in dataset_df.columns:
            print(f"Generating questions with {qgen_model}")
            vf_questions = []
            for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Generating Questions"):
                questions = self.generate_questions_with_model(row, rationale_column, qgen_model)
                vf_questions.append(str(questions))
            dataset_df[vf_q_col] = vf_questions
        else:
            print(f"Questions already exist for {suffix}, skipping generation")
            
        # Step 2: Answer questions if not exists
        if vf_a_col not in dataset_df.columns:
            print(f"Answering questions with {verif_model}")
            for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc="Answering Questions"):
                questions = ast.literal_eval(row[vf_q_col])
                if questions:  # Only process if questions exist
                    answers = self.verify_questions_with_model(questions, dataset_name, verif_model, row['image_path'])
                    dataset_df.loc[index, vf_a_col] = str(answers)
                    # Calculate VF score as proportion of 'yes' answers
                    yes_count = sum(1 for ans in answers if ans.lower() == 'yes')
                    dataset_df.loc[index, vf_score_col] = yes_count / len(answers) if answers else 0
                else:
                    dataset_df.loc[index, vf_a_col] = "[]"
                    dataset_df.loc[index, vf_score_col] = 0
        else:
            print(f"Answers already exist for {suffix}, skipping verification")
            
        return dataset_df
    
    def run_core_ablation(self, save_results=True):
        """Run the 4 core ablation configurations"""
        results = {}
        
        for dataset_name in self.datasets:
            print(f"\n=== Processing {dataset_name} Dataset ===")
            results[dataset_name] = {}
            
            for target_vlm in self.target_vlms:
                print(f"\n--- Target VLM: {target_vlm} ---")
                results[dataset_name][target_vlm] = {}
                
                # Load dataset
                dataset_path = f"model_outputs/{dataset_name}/{target_vlm}.csv"
                dataset_df = pd.read_csv(dataset_path)
                
                # Run each core configuration
                for qgen_model, verif_model in self.core_configs:
                    config_name = self.get_column_suffix(qgen_model, verif_model)
                    
                    dataset_df = self.run_ablation_config(
                        dataset_df, dataset_name, target_vlm, 
                        qgen_model, verif_model
                    )
                    
                    # Store results
                    vf_score_col = f"visual_fidelity_{config_name}"
                    if vf_score_col in dataset_df.columns:
                        avg_score = dataset_df[vf_score_col].mean()
                        results[dataset_name][target_vlm][config_name] = avg_score
                        print(f"Average VF Score for {config_name}: {avg_score:.3f}")
                
                # Save updated dataset
                if save_results:
                    dataset_df.to_csv(dataset_path, index=False)
                    print(f"Saved updated dataset to {dataset_path}")
        
        return results
    
    def run_full_ablation(self, save_results=True):
        """Run full ablation with all model combinations"""
        results = {}
        
        for dataset_name in self.datasets:
            print(f"\n=== Processing {dataset_name} Dataset (Full Ablation) ===")
            results[dataset_name] = {}
            
            for target_vlm in self.target_vlms:
                print(f"\n--- Target VLM: {target_vlm} ---")
                results[dataset_name][target_vlm] = {}
                
                # Load dataset
                dataset_path = f"model_outputs/{dataset_name}/{target_vlm}.csv"
                dataset_df = pd.read_csv(dataset_path)
                
                # Run each configuration
                for qgen_model, verif_model in self.full_configs:
                    config_name = self.get_column_suffix(qgen_model, verif_model)
                    
                    dataset_df = self.run_ablation_config(
                        dataset_df, dataset_name, target_vlm,
                        qgen_model, verif_model
                    )
                    
                    # Store results
                    vf_score_col = f"visual_fidelity_{config_name}"
                    if vf_score_col in dataset_df.columns:
                        avg_score = dataset_df[vf_score_col].mean()
                        results[dataset_name][target_vlm][config_name] = avg_score
                        print(f"Average VF Score for {config_name}: {avg_score:.3f}")
                
                # Save updated dataset
                if save_results:
                    dataset_df.to_csv(dataset_path, index=False)
                    print(f"Saved updated dataset to {dataset_path}")
        
        return results
    
    def analyze_results(self, results, save_analysis=True):
        """Analyze and visualize ablation results"""
        analysis = {
            "summary": {},
            "component_effects": {},
            "interaction_effects": {}
        }
        
        for dataset_name in results:
            analysis["summary"][dataset_name] = {}
            
            for target_vlm in results[dataset_name]:
                vlm_results = results[dataset_name][target_vlm]
                analysis["summary"][dataset_name][target_vlm] = vlm_results
                
                # Calculate component effects (difference when changing one component)
                if len(vlm_results) >= 4:  # Ensure we have core configs
                    baseline = vlm_results.get("4o_4o", 0)
                    qgen_effect = vlm_results.get("qwen_4o", 0) - baseline  # Change QGen only
                    verif_effect = vlm_results.get("4o_qwen", 0) - baseline  # Change Verif only
                    both_effect = vlm_results.get("qwen_qwen", 0) - baseline  # Change both
                    
                    analysis["component_effects"][f"{dataset_name}_{target_vlm}"] = {
                        "qgen_effect": qgen_effect,
                        "verif_effect": verif_effect, 
                        "both_effect": both_effect,
                        "interaction": both_effect - (qgen_effect + verif_effect)  # Interaction term
                    }
        
        if save_analysis:
            with open("ablation_results.json", "w") as f:
                json.dump(analysis, f, indent=2)
            print("Analysis saved to ablation_results.json")
            
        return analysis
    
    def print_summary_table(self, results):
        """Print a formatted summary table of results"""
        print("\n" + "="*80)
        print("VISUAL FIDELITY ABLATION STUDY RESULTS")
        print("="*80)
        
        for dataset_name in results:
            print(f"\n{dataset_name} Dataset:")
            print("-" * 50)
            
            # Create table header
            configs = ["4o_4o", "4o_qwen", "qwen_4o", "qwen_qwen"]
            header = f"{'VLM':<20} " + " ".join(f"{c:>10}" for c in configs)
            print(header)
            print("-" * len(header))
            
            for target_vlm in results[dataset_name]:
                vlm_results = results[dataset_name][target_vlm]
                row = f"{target_vlm:<20} "
                for config in configs:
                    score = vlm_results.get(config, 0)
                    row += f"{score:>10.3f} "
                print(row)


def main():
    """Main execution function"""
    print("Starting Visual Fidelity Ablation Study")
    
    ablation_study = VFAblationStudy()
    
    # Run core ablation (4 key configurations)
    print("Running core ablation configurations...")
    core_results = ablation_study.run_core_ablation(save_results=True)
    
    # Print summary
    ablation_study.print_summary_table(core_results)
    
    # Analyze results
    analysis = ablation_study.analyze_results(core_results)
    
    # Optional: Run full ablation if requested
    run_full = input("\nRun full ablation with all model combinations? (y/n): ").lower() == 'y'
    if run_full:
        print("Running full ablation...")
        full_results = ablation_study.run_full_ablation(save_results=True)
        ablation_study.print_summary_table(full_results)


if __name__ == "__main__":
    main()