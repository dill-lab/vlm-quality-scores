"""
Unified uncertainty quantification baselines for VLM predictions.

Implements three methods:
  - raw_logits: Softmax probability of first generated token (local models only)
  - ptrue: Prompt VLM "Is this correct?" and extract P(True) from logits

Usage:
  python run_uncertainty_baselines.py --model llava --dataset AOKVQA --method all
  python run_uncertainty_baselines.py --model gpt-4o --dataset all --method ptrue
  python run_uncertainty_baselines.py --model all --dataset all --method all --limit 500
"""

import os
import math
import ast
import argparse
import warnings
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq

from globals import DATASETS_FOLDER, MODEL_OUTPUTS_FOLDER
from lm_loader import create_model_instance
from utils import load_image
from ece_analysis import compute_discriminability, compute_ece

warnings.filterwarnings("ignore")

# Maps CLI model name -> model output CSV filename (without .csv)
MODEL_OUTPUT_NAMES = {
    "llava": "llava-v1.5-7b",
    "qwen": "qwen2.5-vl-7b-instruct",
    "gpt-4o": "gpt-4o-2024-05-13",
}

# Maps CLI model name -> baseline CSV filename
BASELINE_NAMES = {
    "llava": "llava-v1.5-7b.csv",
    "qwen": "qwen2.5-vl-7b.csv",
    "gpt-4o": "gpt-4o.csv",
}

# HuggingFace model IDs
HF_MODEL_IDS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_vlm(model_name):
    """Load a local VLM via HuggingFace for logit-based methods."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = HF_MODEL_IDS[model_name]
    print(f"Loading {model_id}...")

    if model_name == "llava":
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device)
    elif model_name == "qwen":
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(device)
    else:
        raise ValueError(f"No HuggingFace model for {model_name}")

    return model, processor, device


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def format_prompt_local(model_name, text, image_path=None):
    """Build prompt string / messages for local models."""
    if model_name == "llava":
        return f"USER: <image>\n{text}\nASSISTANT:"
    elif model_name == "qwen":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            }
        ]


def prepare_inputs(model_name, processor, prompt_or_messages, image, device, image_path=None):
    """Tokenize / process inputs for local models."""
    if model_name == "llava":
        return processor(text=prompt_or_messages, images=image, return_tensors="pt").to(device)
    elif model_name == "qwen":
        text = processor.apply_chat_template(
            prompt_or_messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)


# ---------------------------------------------------------------------------
# Method 1: Raw logits confidence (local models only)
# ---------------------------------------------------------------------------

def get_raw_logits_confidence(model, processor, device, model_name, image, question, choices_str, image_path=None):
    """Compute softmax probability of the first generated token."""
    if model_name == "llava":
        prompt = f"USER: <image>\nQuestion: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices.\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    elif model_name == "qwen":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": f"Question: {question}\nChoices: {choices_str}\nAnswer the question using a single word or phrase from the list of choices."},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_len:]

    if len(generated_ids) == 0 or len(outputs.scores) == 0:
        return 0.0

    # Skip leading space token (LLaVA / Llama specific)
    target_idx = 0
    first_tok = generated_ids[0].item()
    if first_tok == 29871 or processor.tokenizer.decode([first_tok]) == " ":
        if len(generated_ids) > 1 and len(outputs.scores) > 1:
            target_idx = 1

    probs = torch.softmax(outputs.scores[target_idx], dim=-1)
    chosen_id = generated_ids[target_idx].item()
    return probs[0, chosen_id].item()


# ---------------------------------------------------------------------------
# Method 2: p(True)
# ---------------------------------------------------------------------------

PTRUE_PROMPT = (
    "Question: {question}\n"
    "Choices: {choices_str}\n"
    "Proposed answer: {predicted_answer}\n\n"
    'Is the proposed answer correct? Respond with only "True" or "False".'
)


def _resolve_token_ids(tokenizer, words):
    """Resolve token IDs for a list of words, returning a dict word -> token_id."""
    ids = {}
    for w in words:
        tokens = tokenizer.encode(w, add_special_tokens=False)
        if tokens:
            ids[w] = tokens[0]
    return ids


def get_ptrue_local(model, processor, device, model_name, image, question, choices_str, predicted_answer, image_path=None):
    """p(True) for local HuggingFace models using logits."""
    text = PTRUE_PROMPT.format(
        question=question, choices_str=choices_str, predicted_answer=predicted_answer
    )
    prompt_or_messages = format_prompt_local(model_name, text, image_path)
    inputs = prepare_inputs(model_name, processor, prompt_or_messages, image, device, image_path)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_len:]

    if len(generated_ids) == 0 or len(outputs.scores) == 0:
        return 0.5

    # Skip leading space token
    target_idx = 0
    first_tok = generated_ids[0].item()
    if first_tok == 29871 or processor.tokenizer.decode([first_tok]) == " ":
        if len(generated_ids) > 1 and len(outputs.scores) > 1:
            target_idx = 1

    logits = outputs.scores[target_idx][0]  # shape: (vocab_size,)

    # Resolve token IDs for True/False/Yes/No
    tok_ids = _resolve_token_ids(processor.tokenizer, ["True", "False", "Yes", "No", "true", "false", "yes", "no"])

    # Collect best logit for positive and negative classes
    pos_logit = max(
        logits[tok_ids[w]].item() for w in ["True", "Yes", "true", "yes"] if w in tok_ids
    )
    neg_logit = max(
        logits[tok_ids[w]].item() for w in ["False", "No", "false", "no"] if w in tok_ids
    )

    probs = torch.softmax(torch.tensor([pos_logit, neg_logit]), dim=0)
    return probs[0].item()


def get_ptrue_api(lm_model, image_path_abs, question, choices_str, predicted_answer):
    """p(True) for GPT-4o via API with logprobs."""
    encoded_image = load_image(image_path_abs)
    text = PTRUE_PROMPT.format(
        question=question, choices_str=choices_str, predicted_answer=predicted_answer
    )
    messages = [
        {"role": "system", "content": 'Answer only with "True" or "False".'},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                {"type": "text", "text": text},
            ],
        },
    ]
    response = lm_model.chat_completion(
        messages, max_tokens=5, temperature=0.0, logprobs=True, top_logprobs=5
    )
    if not response or "choices" not in response or len(response["choices"]) == 0:
        return 0.5

    choice = response["choices"][0]

    # Try to extract from logprobs first
    logprobs_data = choice.get("logprobs")
    if logprobs_data and "content" in logprobs_data and logprobs_data["content"]:
        top_lps = logprobs_data["content"][0].get("top_logprobs", [])
        true_lp, false_lp = None, None
        for entry in top_lps:
            token_lower = entry["token"].strip().lower()
            if token_lower in ("true", "yes") and true_lp is None:
                true_lp = entry["logprob"]
            elif token_lower in ("false", "no") and false_lp is None:
                false_lp = entry["logprob"]
        if true_lp is not None or false_lp is not None:
            true_lp = true_lp if true_lp is not None else -100.0
            false_lp = false_lp if false_lp is not None else -100.0
            p_true = math.exp(true_lp)
            p_false = math.exp(false_lp)
            return p_true / (p_true + p_false)

    # Fallback: parse text response
    text_out = choice["message"]["content"].strip().lower()
    if "true" in text_out or "yes" in text_out:
        return 1.0
    elif "false" in text_out or "no" in text_out:
        return 0.0
    return 0.5


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def resolve_image_path(row, dataset_name):
    """Resolve the image path to an absolute path."""
    rel_path = row["image_path"]
    candidate = os.path.join(DATASETS_FOLDER, dataset_name, rel_path)
    if os.path.exists(candidate):
        return candidate
    # Fallback: try from cwd
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    return candidate  # return best guess


def load_model_outputs(model_name, dataset_name, limit=None):
    """Load existing model output CSV."""
    csv_name = MODEL_OUTPUT_NAMES[model_name] + ".csv"
    csv_path = os.path.join(MODEL_OUTPUTS_FOLDER, dataset_name, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Model output not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if limit and limit < len(df):
        df = df.head(limit)
    return df


def load_existing_baseline(model_name, dataset_name):
    """Load existing baseline CSV if it exists, to merge columns."""
    baseline_dir = os.path.join(MODEL_OUTPUTS_FOLDER, dataset_name, "baselines")
    baseline_path = os.path.join(baseline_dir, BASELINE_NAMES[model_name])
    if os.path.exists(baseline_path):
        return pd.read_csv(baseline_path)
    return None


def save_baseline(df, model_name, dataset_name):
    """Save baseline CSV."""
    baseline_dir = os.path.join(MODEL_OUTPUTS_FOLDER, dataset_name, "baselines")
    os.makedirs(baseline_dir, exist_ok=True)
    baseline_path = os.path.join(baseline_dir, BASELINE_NAMES[model_name])
    df.to_csv(baseline_path, index=False)
    print(f"Saved baseline to {baseline_path}")


def check_answers_simple(predicted_answer, correct_answers):
    """Simple case-insensitive answer check."""
    if not isinstance(correct_answers, list):
        correct_answers = [correct_answers]
    pred = str(predicted_answer).lower().strip()
    return int(any(pred == str(ans).lower().strip() for ans in correct_answers))


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def run_baselines(model_name, dataset_name, methods, limit=None, test=False):
    """Run specified baseline methods for a given model and dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name} on {dataset_name}")
    print(f"Methods: {methods}")
    print(f"{'='*60}")

    # Load model outputs (source of truth for questions, answers, correctness)
    df = load_model_outputs(model_name, dataset_name, limit=limit)
    if test:
        df = df.head(20)
    print(f"Loaded {len(df)} samples from model outputs.")

    id_col = "index" if "index" in df.columns else "question_id"

    # Load existing baseline CSV to preserve prior columns (e.g. raw_confidence)
    existing = load_existing_baseline(model_name, dataset_name)
    if existing is not None:
        print(f"Found existing baseline with columns: {list(existing.columns)}")
        # Start from existing baseline and add new columns
        result_df = existing.copy()
    else:
        # Build a fresh result DataFrame from model outputs
        result_df = pd.DataFrame()
        result_df["question_id"] = df[id_col]
        result_df["image_path"] = df["image_path"]
        result_df["question"] = df["question"]
        result_df["choices"] = df["choices"]
        result_df["correct_answer"] = df["correct_answer"]
        result_df["predicted_answer"] = df["predicted_answer"]
        result_df["is_correct"] = df["is_correct"]

    # When running in test/limit mode, trim result_df to match df length
    if len(result_df) > len(df):
        result_df = result_df.head(len(df)).copy()

    is_api_model = model_name == "gpt-4o"

    # Check for columns that would be overwritten and ask for confirmation
    col_for_method = {
        "raw_logits": "raw_confidence",
        "ptrue": "p_true",
    }
    methods_to_run = []
    for m in methods:
        col = col_for_method[m]
        if col in result_df.columns and result_df[col].notna().any():
            answer = input(f"  '{col}' already exists in baseline. Overwrite? [y/N] ")
            if answer.strip().lower() != "y":
                print(f"  Skipping {m}.")
                continue
        methods_to_run.append(m)
    methods = methods_to_run

    if not methods:
        print("No methods to run. Done.")
        return

    # Load HuggingFace model for local methods
    needs_local = not is_api_model and any(m in methods for m in ("raw_logits", "ptrue"))
    hf_model, hf_processor, hf_device = None, None, None
    if needs_local:
        hf_model, hf_processor, hf_device = load_vlm(model_name)

    # Load API model
    api_model = None
    if is_api_model:
        api_model = create_model_instance("gpt-4o")

    # --- Method: Raw Logits (local models only) ---
    if "raw_logits" in methods:
        if is_api_model:
            print("Skipping raw_logits for API model (no logit access).")
        else:
            print("\nComputing raw logits confidence...")
            scores = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Raw Logits"):
                abs_path = resolve_image_path(row, dataset_name)
                try:
                    image = Image.open(abs_path).convert("RGB")
                    choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
                    choices_str = ", ".join(choices)
                    score = get_raw_logits_confidence(
                        hf_model, hf_processor, hf_device, model_name,
                        image, row["question"], choices_str, image_path=abs_path
                    )
                    scores.append(score)
                except Exception as e:
                    print(f"Error at row {row.get(id_col, '?')}: {e}")
                    scores.append(0.0)
            result_df["raw_confidence"] = scores

    # --- Method: p(True) ---
    if "ptrue" in methods:
        print("\nComputing p(True)...")
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="p(True)"):
            abs_path = resolve_image_path(row, dataset_name)
            choices = ast.literal_eval(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
            choices_str = ", ".join(choices)
            try:
                if is_api_model:
                    score = get_ptrue_api(
                        api_model, abs_path, row["question"], choices_str,
                        row["predicted_answer"]
                    )
                else:
                    image = Image.open(abs_path).convert("RGB")
                    score = get_ptrue_local(
                        hf_model, hf_processor, hf_device, model_name,
                        image, row["question"], choices_str,
                        row["predicted_answer"], image_path=abs_path
                    )
                scores.append(score)
            except Exception as e:
                print(f"Error at row {row.get(id_col, '?')}: {e}")
                scores.append(0.5)
        result_df["p_true"] = scores

    # Save
    save_baseline(result_df, model_name, dataset_name)

    # Print evaluation
    print_evaluation(result_df)

    # Cleanup GPU memory
    if hf_model is not None:
        del hf_model, hf_processor
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def print_evaluation(df):
    """Print ECE and Discriminability for all baseline columns."""
    y_true = df["is_correct"].values
    y_true_binary = (y_true >= 0.5).astype(int)

    baseline_cols = {
        "Raw Logits": "raw_confidence",
        "P(True)": "p_true",
    }

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")

    for name, col in baseline_cols.items():
        if col in df.columns:
            scores = df[col].dropna().values
            mask = df[col].notna().values
            if len(scores) == 0 or np.all(scores == 0):
                print(f"{name}: Skipped (no data)")
                continue
            disc, p_val = compute_discriminability(y_true_binary[mask], scores)
            ece = compute_ece(y_true[mask], scores)
            ece_str = f"{ece:.4f}" if ece is not None else "N/A"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"{name}: Disc={disc:.4f}{sig} (p={p_val:.4f}), ECE={ece_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Run uncertainty quantification baselines (raw logits, p(True))."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["llava", "qwen", "gpt-4o", "all"],
        help="Model to evaluate."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["AOKVQA", "MMMU-Pro", "all"],
        help="Dataset to evaluate."
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["raw_logits", "ptrue", "all"],
        help="Which baseline method(s) to run."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of samples to process (default: all rows in model output CSV)."
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick test on first 20 rows."
    )
    args = parser.parse_args()

    # Resolve model list
    if args.model == "all":
        models = ["llava", "qwen", "gpt-4o"]
    else:
        models = [args.model]

    # Resolve dataset list
    if args.dataset == "all":
        datasets = ["AOKVQA", "MMMU-Pro"]
    else:
        datasets = [args.dataset]

    # Resolve method list
    if args.method == "all":
        methods = ["raw_logits", "ptrue"]
    elif args.method == "raw_logits":
        methods = ["raw_logits"]
    else:
        methods = [args.method]

    for model_name in models:
        for dataset_name in datasets:
            try:
                run_baselines(
                    model_name, dataset_name, methods,
                    limit=args.limit, test=args.test
                )
            except FileNotFoundError as e:
                print(f"Skipping {model_name}/{dataset_name}: {e}")
            except Exception as e:
                print(f"Error processing {model_name}/{dataset_name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
