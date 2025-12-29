import os
import pandas as pd
import torch
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq
from PIL import Image
from tqdm import tqdm
import warnings
import argparse
import ast
from ece_analysis import compute_discriminability, compute_ece

warnings.filterwarnings("ignore")

# Configuration
DATASET_PATH = "data/AOKVQA/AOKVQA.csv"
IMAGE_DIR = "data/AOKVQA"
SEED = 42

def load_vlm(model_name):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if model_name == "llava":
        model_id = "llava-hf/llava-1.5-7b-hf"
        print(f"Loading {model_id}...")
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16
        ).to(device)
        return model, processor, device, "llava"
        
    elif model_name == "qwen":
        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Loading {model_id}...")
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
            ).to(device)
            return model, processor, device, "qwen"
        except Exception as e:
            print(f"Error loading Qwen: {e}")
            raise e

    elif model_name == "gpt-4o":
 
    else:
        raise ValueError(f"Unknown model: {model_name}")

def calculate_raw_confidence(scores, generated_token_ids, tokenizer):
    """
    Calculate confidence based on the probability of the first meaningful generated token.
    Skips leading space if present.
    """
    if len(generated_token_ids) == 0 or len(scores) == 0:
        return 0.0
        
    # Determine which token to look at
    target_idx = 0
    
    # Check for leading space (Llama tokenizer specific, but generic check)
    # 29871 is Llama's space token.
    first_token_id = generated_token_ids[0]
    if first_token_id == 29871 or tokenizer.decode([first_token_id]) == " ":
        if len(generated_token_ids) > 1 and len(scores) > 1:
            target_idx = 1
            
    probs = torch.softmax(scores[target_idx], dim=-1)
    chosen_token_id = generated_token_ids[target_idx]
    return probs[0, chosen_token_id].item()

def get_choice_softmax_confidence(model, processor, device, image, question, choices_str, choices, predicted_answer, model_type, image_path=None):
    """
    Get probability of predicted answer by applying softmax over choice probabilities (first token).
    """
    if model_type == "llava":
        prompt = f"USER: <image>\nQuestion: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices.\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    elif model_type == "qwen":
        if image_path is None:
            return 0.0
        # Qwen formatting
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Question: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices."}
                ]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=5, # Generate enough tokens to skip potential leading space
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Analyze generated tokens
    input_len = inputs.input_ids.shape[1]
    generated_ids = outputs.sequences[0, input_len:]
    
    if len(generated_ids) == 0:
        return {
            "confidence": 0.0,
            "choice_probs": [],
            "generated_logit": 0.0,
            "is_generated_in_choices": False
        }

    # Determine which step to look at
    target_step_idx = 0
    
    # Check for leading space
    first_token_id = generated_ids[0].item()
    if (first_token_id == 29871 or processor.tokenizer.decode([first_token_id]) == " ") and len(generated_ids) > 1:
        target_step_idx = 1
    
    # Get logits for the target step
    if target_step_idx < len(outputs.scores):
        target_logits = outputs.scores[target_step_idx][0]
        generated_token_id = generated_ids[target_step_idx].item()
        generated_logit = target_logits[generated_token_id].item()
    else:
        # Fallback if something went wrong
        return {
            "confidence": 0.0,
            "choice_probs": [],
            "generated_logit": 0.0,
            "is_generated_in_choices": False
        }
    
    choice_first_tokens = []
    choice_indices = [] # To map back to original choices
    
    for i, choice in enumerate(choices):
        # Handle variations: original (stripped), capitalized
        variations = set()
        variations.add(choice.strip())
        variations.add(choice.strip().capitalize())
        
        for var in variations:
            tokens = processor.tokenizer.encode(var, add_special_tokens=False)
            if len(tokens) > 0:
                # We use the first token of the word (without leading space)
                choice_first_tokens.append(tokens[0])
                choice_indices.append(i)
    
    # Calculate probs
    probs = torch.softmax(target_logits, dim=0).cpu().numpy()
    
    final_choice_probs = [0.0] * len(choices)
    for token_id, choice_idx in zip(choice_first_tokens, choice_indices):
        final_choice_probs[choice_idx] += probs[token_id]
        
    # Normalize probabilities across choices
    total_prob = sum(final_choice_probs)
    if total_prob > 0:
        normalized_probs = [p / total_prob for p in final_choice_probs]
    else:
        normalized_probs = [0.0] * len(choices)
    
    # Check if generated token is in choices (checking against all valid variations)
    is_generated_in_choices = generated_token_id in choice_first_tokens
    
    try:
        pred_idx = [c.lower().strip() for c in choices].index(predicted_answer.lower().strip())
        confidence = normalized_probs[pred_idx]
    except ValueError:
        confidence = 0.0
    
    return {
        "confidence": float(confidence),
        "choice_probs": final_choice_probs, # Return unnormalized probs to see raw values
        "generated_logit": generated_logit,
        "is_generated_in_choices": is_generated_in_choices
    }





def check_answers(predicted_answer, correct_answers):
    if not isinstance(correct_answers, list):
        correct_answers = [correct_answers]
    pred = str(predicted_answer).lower().strip()
    return any(pred == str(ans).lower().strip() for ans in correct_answers)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["llava", "qwen"])
    parser.add_argument("--input", type=str, default=DATASET_PATH, help="Path to input dataset CSV")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV. If None, auto-generated.")
    args = parser.parse_args()

    # 1. Load Data
    if not os.path.exists(args.input):
        print(f"Error: Dataset not found at {args.input}")
        return
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} items from dataset.")
    
    # Determine Output Path
    if args.output is None:
        # Infer dataset name from input path
        # e.g. data/AOKVQA/AOKVQA.csv -> AOKVQA
        # e.g. data/MMMU-Pro/MMMU-Pro.csv -> MMMU-Pro
        input_dir = os.path.dirname(args.input)
        dataset_name = os.path.basename(input_dir)
        
        # Fallback if path is weird
        if not dataset_name: 
            dataset_name = "unknown_dataset"
            
        output_dir = f"model_outputs/{dataset_name}/baselines"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Determine model filename
        if args.model == "llava":
            model_filename = "llava-v1.5-7b.csv"
        elif args.model == "qwen":
            model_filename = "qwen2.5-vl-7b.csv"
        elif args.model == "gpt-4o":
            model_filename = "gpt-4o.csv"
        else:
            model_filename = f"{args.model}_baseline.csv"
            
        args.output = os.path.join(output_dir, model_filename)
        print(f"Auto-generated output path: {args.output}")
    else:
        # Ensure directory exists for provided output
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load VLM
    vlm_model, vlm_processor, vlm_device, model_type = load_vlm(args.model)
    
    results = []
    
    print("Running VLM Inference and Logits Calculation...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Determine image directory based on input file location
        input_dir = os.path.dirname(args.input)
        # If input is in data/MMMU-Pro, images are in data/MMMU-Pro/images
        # If input is in data/AOKVQA, images are in data/AOKVQA/images (or defined by IMAGE_DIR)
        
        # Try constructing path relative to input file first
        image_path = os.path.join(input_dir, row['image_path'])
        
        if not os.path.exists(image_path):
            # Fallback to global IMAGE_DIR (for AOKVQA legacy)
            image_path = os.path.join(IMAGE_DIR, row['image_path'])
            
        if not os.path.exists(image_path):
            # Try one more: maybe row['image_path'] is absolute or relative to cwd?
            if os.path.exists(row['image_path']):
                image_path = row['image_path']
            else:
                # print(f"Warning: Image not found: {row['image_path']}")
                continue
            
        try:
            image = Image.open(image_path).convert('RGB')
            question = row['question']
            choices = eval(row['choices'])
            choices_str = ", ".join(choices)
            
            # --- VLM Inference ---
            if model_type == "llava":
                prompt = f"USER: <image>\nQuestion: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices.\nASSISTANT:"
                inputs = vlm_processor(text=prompt, images=image, return_tensors="pt").to(vlm_device)
            elif model_type == "qwen":
                messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": f"Question: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices."}]}]
                text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = vlm_processor(text=[text], images=[image], padding=True, return_tensors="pt").to(vlm_device)

            outputs = vlm_model.generate(**inputs, max_new_tokens=20, return_dict_in_generate=True, output_scores=True)
            generated_text = vlm_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            
            if model_type == "llava":
                answer = generated_text.split("ASSISTANT:")[-1].strip()
            elif model_type == "qwen":
                answer = generated_text.split("assistant\n")[-1].strip()

            # Raw Confidence
            prompt_length = inputs['input_ids'].shape[1]
            generated_token_ids = outputs.sequences[0, prompt_length:]
            raw_confidence = calculate_raw_confidence(outputs.scores, generated_token_ids, vlm_processor.tokenizer)
            
            # Check correctness
            correct_ans_raw = row['correct_answer']
            try:
                correct_answers = ast.literal_eval(correct_ans_raw) if isinstance(correct_ans_raw, str) else correct_ans_raw
            except:
                correct_answers = [str(correct_ans_raw)]
            if not isinstance(correct_answers, list):
                correct_answers = [correct_answers]
            is_correct = check_answers(answer, correct_answers)
            
            results.append({
                'question_id': row['index'],
                'image_path': row['image_path'],
                'question': question,
                'choices': row['choices'],
                'correct_answer': row['correct_answer'],
                'predicted_answer': answer,
                'raw_confidence': raw_confidence,
                'is_correct': 1 if is_correct else 0
            })
            
        except Exception as e:
            print(f"Error processing {row['index']}: {e}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    
    # Analysis
    y_true = df_results['is_correct'].values
    
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    
    metrics = {
        "Raw Logits": "raw_confidence",
    }
    
    for name, col in metrics.items():
        if col in df_results.columns:
            scores = df_results[col].values
            # Check if all zeros (e.g. Qwen choice softmax)
            if np.all(scores == 0):
                print(f"{name}: Skipped (All zeros)")
                continue
                
            disc, p_val = compute_discriminability(y_true, scores)
            ece = compute_ece(y_true, scores)
            ece_str = f"{ece:.4f}" if ece is not None else "N/A"
            print(f"{name}: Disc={disc:.4f} (p={p_val:.4f}), ECE={ece_str}")

if __name__ == "__main__":
    main()
