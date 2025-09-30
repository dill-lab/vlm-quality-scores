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
        return ""

def generate_hypotheses(
    question: str,
    predicted_ans: str,
    other_answers: list[str],
    model: LMModel,
    include_alternatives: bool = True,
) -> dict:
    hypothesis = integrate_question_answer(question, predicted_ans, model)
    alternative_hypotheses = []
    if include_alternatives:
        alternative_hypotheses = [
            integrate_question_answer(question, alternative_ans, model)
            for alternative_ans in other_answers
        ]
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

def evaluate_support_and_contrastiveness(
    data,
    rationale_column_name,
    hypothesis_column_name,
    alt_hypotheses_column_name,
    choices_column_name,
    threshold_support=0.5,
    threshold_contrastiveness=0.5,
    include_contrastive=True,
):
    scores = []
    labels = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        raw_choices = None
        if choices_column_name and choices_column_name in row.index:
            raw_choices = row[choices_column_name]

        parsed_choices = []
        if isinstance(raw_choices, str):
            try:
                candidate_list = ast.literal_eval(raw_choices)
                if isinstance(candidate_list, list):
                    parsed_choices = candidate_list
            except (ValueError, SyntaxError):
                parsed_choices = []
        elif isinstance(raw_choices, list):
            parsed_choices = raw_choices
        elif raw_choices is not None and not pd.isna(raw_choices):
            parsed_choices = [raw_choices]

        all_choices = [str(choice).strip() for choice in parsed_choices if str(choice).strip()]
        predicted_answer = str(row['predicted_answer'])
        if predicted_answer:
            all_choices.append(predicted_answer)

        seen = set()
        deduped_choices = []
        for choice in all_choices:
            if choice and choice not in seen:
                seen.add(choice)
                deduped_choices.append(choice)
        if not deduped_choices and predicted_answer:
            deduped_choices.append(predicted_answer)

        premise = mask_rationale(row[rationale_column_name], deduped_choices)
        hypothesis = row[hypothesis_column_name]

        alt_hypotheses = []
        if include_contrastive and alt_hypotheses_column_name and alt_hypotheses_column_name in row.index:
            alt_raw = row[alt_hypotheses_column_name]
            if isinstance(alt_raw, str):
                try:
                    parsed_alt = ast.literal_eval(alt_raw)
                    if isinstance(parsed_alt, list):
                        alt_hypotheses = parsed_alt
                except (ValueError, SyntaxError):
                    alt_hypotheses = []
            elif isinstance(alt_raw, list):
                alt_hypotheses = alt_raw

        entail_prob = calc_support_prob(premise, hypothesis)
        support = entail_prob > threshold_support

        score_entry = {'entail_prob': entail_prob}
        label_entry = {'support': support}

        if include_contrastive:
            alt_entail_probs = [calc_support_prob(premise, alt_hypothesis) for alt_hypothesis in alt_hypotheses]
            contrastiveness_score = entail_prob / (entail_prob + sum(alt_entail_probs)) if entail_prob + sum(alt_entail_probs) != 0 else 0
            contrastive = contrastiveness_score > threshold_contrastiveness

            score_entry['alt_entail_probs'] = str(alt_entail_probs)
            score_entry['contrastiveness_score'] = contrastiveness_score
            label_entry['contrastive'] = contrastive

        scores.append(score_entry)
        labels.append(label_entry)

    return scores, labels


def support_contr_analysis_all_datasets(
    question_column_name: str,
    answer_column_name: str,
    majority_answer_column_name: str,
    all_choices_column_name: str,
    rationale_column_name: str,
    dataset_list: list[pd.DataFrame],
    model_name: str = "gpt-4o-mini",
    same_question_set: bool = True,
    overwrite_candidate_answers: bool = False,
    overwrite_hypotheses_columns: bool = False,
    include_contrastive: bool = True,
):
    model = create_model_instance(model_name)

    choices_column = all_choices_column_name
    if include_contrastive and (choices_column is None or choices_column not in dataset_list[0].columns):
        column_suffix = choices_column if choices_column is not None else 'choices'
        choices_column = 'generated_' + column_suffix
        if overwrite_candidate_answers or choices_column not in dataset_list[0].columns:
            if same_question_set:
                candidate_answers = generate_candidate_answers(
                    dataset_list[0][question_column_name],
                    dataset_list[0][majority_answer_column_name]
                )
                for dataset in dataset_list:
                    for idx, (index, _) in enumerate(dataset.iterrows()):
                        if idx < len(candidate_answers):
                            dataset.at[index, choices_column] = str(candidate_answers[idx])
            else:
                for dataset in dataset_list:
                    candidate_answers = generate_candidate_answers(
                        dataset[question_column_name],
                        dataset[majority_answer_column_name]
                    )
                    for idx, (index, _) in enumerate(dataset.iterrows()):
                        if idx < len(candidate_answers):
                            dataset.at[index, choices_column] = str(candidate_answers[idx])

    hypothesis_column_name = 'hypothesis'
    alt_hypotheses_column_name = 'alternative_hypotheses' if include_contrastive else None
    for i, dataset in enumerate(dataset_list):
        if 'hypothesis' not in dataset.columns or overwrite_hypotheses_columns:
            for idx, (index, row) in enumerate(tqdm(dataset.iterrows(), total=len(dataset), desc=f"Generating hypotheses for dataset {i}")):
                question = str(row[question_column_name])
                predicted_ans = str(row[answer_column_name])

                parsed_choices = []
                if choices_column and choices_column in row.index:
                    raw_choices = row[choices_column]
                    if isinstance(raw_choices, str):
                        try:
                            parsed_choices = ast.literal_eval(raw_choices)
                            if not isinstance(parsed_choices, list):
                                parsed_choices = []
                        except (ValueError, SyntaxError):
                            parsed_choices = []
                    elif isinstance(raw_choices, list):
                        parsed_choices = raw_choices

                other_answers = []
                if include_contrastive and parsed_choices:
                    other_answers = [choice for choice in parsed_choices if choice != predicted_ans][:3]

                hypotheses = generate_hypotheses(
                    question,
                    predicted_ans,
                    other_answers,
                    model,
                    include_alternatives=include_contrastive,
                )

                dataset.at[index, hypothesis_column_name] = hypotheses['hypothesis']
                if include_contrastive:
                    dataset.at[index, alt_hypotheses_column_name] = str(hypotheses['alternative_hypotheses'])

    for dataset in dataset_list:
        scores, labels = evaluate_support_and_contrastiveness(
            data=dataset,
            rationale_column_name=rationale_column_name,
            hypothesis_column_name=hypothesis_column_name,
            alt_hypotheses_column_name=alt_hypotheses_column_name,
            choices_column_name=choices_column,
            include_contrastive=include_contrastive,
        )
        dataset['support'] = [label['support'] for label in labels]
        dataset['entail_prob'] = [score['entail_prob'] for score in scores]
        if include_contrastive:
            dataset['contrastive'] = [label['contrastive'] for label in labels]
            dataset['alt_entail_probs'] = [score['alt_entail_probs'] for score in scores]
            dataset['contrastiveness_score'] = [score['contrastiveness_score'] for score in scores]
        else:
            dataset.drop(
                columns=[
                    'generated_choices',
                    'alternative_hypotheses',
                    'contrastive',
                    'alt_entail_probs',
                    'contrastiveness_score',
                ],
                inplace=True,
                errors='ignore'
            )


if __name__ == "__main__":
    # Example usage
    dataset_list = [pd.read_csv("model_outputs/VizWiz/llava-v1.5-7b_test.csv")]
    support_contr_analysis_all_datasets("question", "predicted_answer", "majority_answer", "choices", "rationale", 
                                  dataset_list, overwrite_candidate_answers=False, overwrite_hypotheses_columns=False)
    # write the updated datasets back to the same files
    dataset_list[0].to_csv("model_outputs/VizWiz/llava-v1.5-7b_test.csv", index=False)
    
