import re
import os
import base64
from lavegpt import LaveChatGPT

_model_cache = {}

def load_image(image_path):
    """
    Load an image from the local filesystem, based on the image_path.
    """
    return base64.b64encode(open(image_path, "rb").read()).decode("utf-8")

def check_answers(predict_answer, correct_answers):
    """
    Check if the predicted answer is among the allowed correct choices.
    Returns a bool which is 1 if the predicted answer is among the allowed correct choices, and 0 otherwise.
    Case insensitive.
    """
    return int(predict_answer.lower() in [ans.lower() for ans in correct_answers])

def check_answers_LAVE(predict_answer, correct_answers, question):
    """
    Check if the predicted answer is among the allowed correct choices, using LAVE_GPT.
    """
    if not isinstance(correct_answers, list):
        raise ValueError("correct_answers must be a list of strings.")
    
    # first run check_answers, if it is correct, return 1
    if check_answers(predict_answer, correct_answers):
        return 1
    
    if 'lave' not in _model_cache:
        _model_cache['lave'] = LaveChatGPT()
    lave_scorer = _model_cache['lave']
    reasoning, score = lave_scorer.compute(predict_answer, correct_answers, question)
    
    debug = True
    if(debug):
        print(f"Predicted: {predict_answer}, Correct: {correct_answers}\n Question: {question}, is_correct: {score}")
        print(f"Reasoning: {reasoning}")
    
    return score


# def extract_and_check_answers_from_rationale(response, correct_answers):
#     """
#     Extracts answer(s) from a response and checks if they are correct.
#     Returns a tuple containing:
#         (list_of_extracted_answers, check_flag)
#     where list_of_extracted_answers is a list of answers (using the original formatting from correct_answers if available),
#     and check_flag is 1 if every extracted answer is among the allowed correct choices, and 0 otherwise.
#     """
#     # Ensure correct_answers is a list.
#     if isinstance(correct_answers, str):
#         correct_answers = [correct_answers]
    
#     # Locate the answer segment using known markers; fallback to the first sentence.
#     markers = ["Thus, the answer is ", "Thus, "]
#     answer_segment = None
#     for marker in markers:
#         if marker in response:
#             answer_segment = response.split(marker, 1)[1].split(".")[0]
#             break
#     if answer_segment is None:
#         answer_segment = response.split(".")[0]

#     # Basic cleanup: remove wrapping quotes and trim spaces.
#     answer_segment = answer_segment.strip().strip('\'"`')
#     # Remove extraneous explanation text.
#     for phrase in [" as ", " because ", " since ", " and not ", " but not "]:
#         answer_segment = answer_segment.split(phrase)[0].strip()

#     # Split the answer segment into candidates on commas, "and", or "or".
#     raw_candidates = re.split(r',|\band\b|\bor\b', answer_segment)
#     candidates = [re.sub(r'[^a-zA-Z0-9\s]', '', cand).strip().lower() 
#                   for cand in raw_candidates if cand.strip()]

#     if not candidates:
#         return ([], 0)

#     # Clean the allowed correct choices similarly.
#     cleaned_choices = [re.sub(r'[^a-zA-Z0-9\s]', '', choice).strip().lower() 
#                        for choice in correct_answers]

#     extracted_answers = []
#     all_valid = True
#     for cand in candidates:
#         if cand in cleaned_choices:
#             # Preserve the original formatting from correct_answers.
#             candidate_choice = correct_answers[cleaned_choices.index(cand)]
#             if candidate_choice not in extracted_answers:
#                 extracted_answers.append(candidate_choice)
#         else:
#             # Instead of an exact match, check if any allowed choice appears as a standalone word in the candidate.
#             found = False
#             for idx, choice in enumerate(cleaned_choices):
#                 if re.search(r'\b' + re.escape(choice) + r'\b', cand):
#                     # Append without repetition
#                     if correct_answers[idx] not in extracted_answers:
#                         extracted_answers.append(correct_answers[idx])
#                     found = True
#                     break
#             if not found:
#                 # Candidate is not among the allowed answers.
#                 if cand not in extracted_answers:
#                     extracted_answers.append(cand)
#                 all_valid = False

#     return (extracted_answers, 1 if all_valid else 0)
