import pandas as pd
import ast
import torch
import re
from tqdm import tqdm
from lm_loader import LMModel, create_model_instance
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

gpt4omini_model = create_model_instance("gpt-4o-mini")

def integrate_question_answer(question: str, predicted_ans: str, model: LMModel) -> str:    
    user_prompt = f"""Integrate the question and the answer into one sentence.
For example, given the question "What is the man waiting for?" and the answer "taxi", you should output "The man is waiting for taxi."

Question: {question}
Answer: {predicted_ans}
"""
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": user_prompt}
        ]}
    ]

    # Generate hypothesis
    response = model.chat_completion(messages)
    try:
        hypothesis = response['choices'][0]['message']['content']
        return hypothesis
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_hypotheses(question: str, predicted_ans: str, other_answers: list[str], model: LMModel) -> dict:
    hypothesis = integrate_question_answer(question, predicted_ans, model)
    alternative_hypotheses = [integrate_question_answer(question, alternative_ans, model) for alternative_ans in other_answers]
    return {
        "hypothesis": hypothesis,
        "alternative_hypotheses": alternative_hypotheses
    }
    
def generate_candidate_answers(questions, correct_answers, model_name: str = "gpt-4o-mini") -> list[list[str]]:
    model = create_model_instance(model_name)
    candidate_answers_list = []
    
    for question, correct_answer in zip(questions, correct_answers):
        # if the question is a yes/no question, then the candidate answers are fixed.
        if correct_answer.lower().strip() in ['yes', 'no']:
            candidate_answers_list.append(["yes", "not sure", "no", "unanswerable"])
            continue
        prompt = f"""
Given the following question and its correct answer, generate exactly four different plausible but incorrect candidate answers that could be reasonable responses.

Example:
Question: "Which one is the blue one?"
Correct Answer: "the left one"
Incorrect Answers: "the right one, neither, both, unanswerable"

Now, generate incorrect answers for the following:

Question: "{question}"
Correct Answer: "{correct_answer}"

Requirements:
- The answers should be reasonable responses to the question but still incorrect.
- Each answer should be distinctly different from each other.
- The answer should not rely too much on a specific correct answer but should follow general reasoning.
- The answer may consist of one or two words.
- Provide exactly four incorrect answers, separated by commas, with no additional text.
"""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
        ]
        
        response = model.chat_completion(messages)
        
        try:
            answer_text = response['choices'][0]['message']['content'].strip(" \n`[]")
            candidate_answers = [word.strip() for word in answer_text.split(",")]
            if len(candidate_answers) == 4:
                candidate_answers_list.append(candidate_answers)
            else:
                print(f"Warning: Expected 4 answers, but got {len(candidate_answers)} for question '{question}'.")
                candidate_answers_list.append(candidate_answers)  # Add anyway for debugging
        except Exception as e:
            print(f"Error: {e}")
            candidate_answers_list.append([])  # Return empty list on failure

    return candidate_answers_list

_model_cache = {}

def get_nli_model():
    if "nli_model" not in _model_cache:
        _model_cache["nli_model"] = AutoModelForSeq2SeqLM.from_pretrained(
            "soumyasanyal/nli-entailment-verifier-xxl",
            device_map="mps",
            torch_dtype=torch.float16
        )
    return _model_cache["nli_model"]

def get_nli_tokenizer():
    if "nli_tokenizer" not in _model_cache:
        _model_cache["nli_tokenizer"] = AutoTokenizer.from_pretrained("google/flan-t5-xxl")
    return _model_cache["nli_tokenizer"]

def calc_support_prob(premise, hypothesis):
    nli_tokenizer = get_nli_tokenizer()
    nli_model = get_nli_model()
    
    def get_score(nli_model, nli_tokenizer, input_ids):
        pos_ids = nli_tokenizer('Yes').input_ids
        neg_ids = nli_tokenizer('No').input_ids
        pos_id = pos_ids[0]
        neg_id = neg_ids[0]

        with torch.no_grad():
            logits = nli_model(input_ids, decoder_input_ids=torch.zeros((input_ids.size(0), 1), dtype=torch.long, device=input_ids.device)).logits
            pos_logits = logits[:, 0, pos_id]
            neg_logits = logits[:, 0, neg_id]
            posneg_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits.unsqueeze(-1)], dim=1)

            # Cast to float before applying softmax
            posneg_logits = posneg_logits.float()
            scores = torch.nn.functional.softmax(posneg_logits, dim=1)
            entail_score = scores[:, 0].item()
            no_entail_score = scores[:, 1].item()
        
        return entail_score, no_entail_score
    
    prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nGiven the premise, is the hypothesis correct?\nAnswer:"
    input_ids = nli_tokenizer(prompt, return_tensors='pt').input_ids.to("mps")
    return get_score(nli_model, nli_tokenizer, input_ids)[0]

def mask_rationale(generated_rationale, answer_options: list[str]):
    if type(answer_options) is not list:
        raise ValueError("answer_options should be a list of strings, but got: " + str(type(answer_options)))
    # print(f"Answer options: {answer_options}")
    masked_rationale = generated_rationale
    for predicted_answer in answer_options:
        # Create a regex pattern to match the predicted answer case-insensitively and as a whole word
        pattern = re.compile(r'\b' + re.escape(predicted_answer) + r'\b', re.IGNORECASE)
        masked_rationale = pattern.sub("<mask>", masked_rationale)
    # print(masked_rationale)
    return masked_rationale

def evaluate_support_and_contrastiveness(data,
                     rationale_column_name, hypothesis_column_name, alt_hypotheses_column_name, choices_column_name,
                     threshold_support=0.5, threshold_contrastiveness=0.5):
    scores = []
    labels = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        # Combine the choices and the predicted answer into a single list and remove duplicates
        all_choices = list(set(ast.literal_eval(row[choices_column_name]) + [str(row['predicted_answer'])]))
        premise = mask_rationale(row[rationale_column_name], all_choices)
        hypothesis = row[hypothesis_column_name]
        alt_hypotheses = ast.literal_eval(row[alt_hypotheses_column_name])
        
        entail_prob = calc_support_prob(premise, hypothesis)
        support = entail_prob > threshold_support
        
        alt_entail_probs = [calc_support_prob(premise, alt_hypothesis) for alt_hypothesis in alt_hypotheses]

        contrastiveness_score = entail_prob / (entail_prob + sum(alt_entail_probs)) if entail_prob + sum(alt_entail_probs) != 0 else 0
        contrastive = contrastiveness_score > threshold_contrastiveness
        
        scores.append({
            'entail_prob': entail_prob,
            'alt_entail_probs': str(alt_entail_probs),
            'contrastiveness_score': contrastiveness_score
        })
        labels.append({
            'support': support,
            'contrastive': contrastive
        })
        
    return scores, labels

def support_contr_analysis_all_datasets(question_column_name: str, answer_column_name: str, majority_answer_column_name: str,
                                  all_choices_column_name: str, rationale_column_name: str,
                                  dataset_list: list[pd.DataFrame], model_name: str = "gpt-4o-mini",
                                  same_question_set=True, overwrite_candidate_answers=False, overwrite_hypotheses_columns=False):
    model = create_model_instance(model_name)
    
    # create a generated answer choices column for each dataset
    if all_choices_column_name is None or all_choices_column_name not in dataset_list[0].columns:
        column_suffix = all_choices_column_name if all_choices_column_name is not None else 'choices'
        all_choices_column_name = 'generated_' + column_suffix
        if overwrite_candidate_answers or all_choices_column_name not in dataset_list[0].columns:
            # Generate candidate answers for each dataset based on the question column
            # In our use case, same_question_set is true because we are using the same test set for all models.
            if same_question_set:
                candidate_answers = generate_candidate_answers(dataset_list[0][question_column_name], dataset_list[0][majority_answer_column_name])
                # For each dataset, assign the candidate answers to a new column.
                for i, dataset in enumerate(dataset_list):
                    for index, row in dataset.iterrows():
                        dataset.at[index, all_choices_column_name] = str(candidate_answers[index])
            else:
                for i, dataset in enumerate(dataset_list):
                    candidate_answers = generate_candidate_answers(dataset[question_column_name], dataset[majority_answer_column_name])
                    for index, row in dataset.iterrows():
                        dataset.at[index, all_choices_column_name] = str(candidate_answers[index])
            
    # For every dataset, generate hypotheses and alternative hypotheses
    hypothesis_column_name = 'hypothesis'
    alt_hypotheses_column_name = 'alternative_hypotheses'
    for i, dataset in enumerate(dataset_list):
        if 'hypothesis' not in dataset.columns or overwrite_hypotheses_columns:
            for index, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Generating hypotheses for dataset {i}"):
                question = row[question_column_name]
                predicted_ans = row[answer_column_name]
                # Get the other *different* answers
                all_choices = ast.literal_eval(row[all_choices_column_name]) if type(row[all_choices_column_name]) == str else row[all_choices_column_name]
                other_answers = [choice for choice in all_choices if choice != predicted_ans][:3]
                hypotheses = generate_hypotheses(question, predicted_ans, other_answers, model)
                
                dataset.at[index, hypothesis_column_name] = hypotheses['hypothesis']
                dataset.at[index, alt_hypotheses_column_name] = str(hypotheses['alternative_hypotheses'])
                
    # Analyze the support (and contrastivity) for each instance
    for i, dataset in enumerate(dataset_list):
        scores, labels = evaluate_support_and_contrastiveness(
            data=dataset,
            rationale_column_name=rationale_column_name,
            hypothesis_column_name=hypothesis_column_name,
            alt_hypotheses_column_name=alt_hypotheses_column_name,
            choices_column_name=all_choices_column_name,
        )
        dataset['support'] = [label['support'] for label in labels]
        dataset['contrastive'] = [label['contrastive'] for label in labels]
        dataset['entail_prob'] = [score['entail_prob'] for score in scores]
        dataset['alt_entail_probs'] = [score['alt_entail_probs'] for score in scores]
        dataset['contrastiveness_score'] = [score['contrastiveness_score'] for score in scores]


if __name__ == "__main__":
    # Example usage
    dataset_list = [pd.read_csv("model_outputs/VizWiz/llava-v1.5-7b_test.csv")]
    support_contr_analysis_all_datasets("question", "predicted_answer", "majority_answer", "choices", "rationale", 
                                  dataset_list, overwrite_candidate_answers=False, overwrite_hypotheses_columns=False)
    # write the updated datasets back to the same files
    dataset_list[0].to_csv("model_outputs/VizWiz/llava-v1.5-7b_test.csv", index=False)
    
