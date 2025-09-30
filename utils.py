import re
import os
import base64
from lavegpt import LaveChatGPT

_model_cache = {}

def load_image(image_path):
    """
    Load an image from the local filesystem, based on the image_path.
    Returns a base64 string. If the file is missing, returns an empty 1x1 PNG.
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
    
    return score

