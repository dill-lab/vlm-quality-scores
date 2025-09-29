# Adapted from https://github.com/tejas1995/ReCoVERR/blob/main/src/utils/eval_utils.py

from pathlib import Path
from typing import List, Union, Tuple
from lm_loader import create_model_instance
from modules.lave.lave import LaveBase

binary_demos_file_path = 'modules/lave/data/lave_demos_binary.json'
demos_file_path = 'modules/lave/data/lave_demos.json'

class LaveChatGPT(LaveBase):
    def __init__(
        self,
        num_shots: int = 8,
        rationalize: bool = True,
        filter_refs: bool = True,
        use_caption: bool = False,
        demos_file: Union[str, Path] = demos_file_path,
        binary_demos_file: Union[str, Path] = binary_demos_file_path,
        debug: bool = False
    ) -> None:
        super().__init__(
            num_shots=num_shots,
            rationalize=rationalize,
            filter_refs=filter_refs,
            use_caption=use_caption,
            demos_file=demos_file,
            binary_demos_file=binary_demos_file,
            debug=debug
        )
        self.input_template = "Question: '{question}' \n Reference answers: {references} \n Candidate answer: '{prediction}'"
        self.output_template = "Output: {output}"

    def build_prompt(self, prediction: str, references: List[str], question: str, caption: str = None) -> str:
        prompt_messages = [{'role': 'system', 'content': self.task_definition}]
        demos = self.select_demos(question, references)
        for demo in demos:
            kwargs = {
                'question': demo['question'],
                'references': self.format_references(demo['references'], filter=False),
                'prediction': demo['prediction'],
                'output': f"{demo['explanation']} So rating={demo['output']}" if self.rationalize else demo['output']
            }
            prompt_messages.append({
                'role': 'user',
                'content': self.input_template.format(**kwargs)
            })
            prompt_messages.append({
                'role': 'assistant',
                'content': self.output_template.format(**kwargs)
            })

        kwargs = {
            'question': question,
            'references': self.format_references(references, filter=self.filter_refs),
            'prediction': prediction,
            'output': ''
        }
        prompt_messages.append({
            'role': 'user',
            'content': self.input_template.format(**kwargs)
        })
        return prompt_messages        


    def compute(self, prediction: str, references: List[str], question: str, caption: str = None) -> Tuple[str, float]:
        prompt_messages = self.build_prompt(prediction, references, question, caption)
        gpt_model = create_model_instance("gpt-4o-mini")
        gpt_response = gpt_model.chat_completion(
                    prompt_messages,        
                    temperature=0.0,
                )
        gpt_response = gpt_response['choices'][0]['message']['content']
        try:
            rating = int(gpt_response.split('=')[-1])
            assert rating in [1, 2, 3]
        except:
            print(f"Error: Unexpected response from GPT-4o-mini: {gpt_response}")
            rating = 1
        score = (rating-1)/2
        reasoning = gpt_response
        return reasoning, score

lave_scorer = LaveChatGPT()

if __name__ == '__main__':
    question = "What is this?"
    references = ['post office box drop', 'mail dropbox', 'mail drop box', 'blue mailbox', 'mailbox', 'post office mailbox']
    prediction = "mailbox"
    reasoning, score = lave_scorer.compute(prediction, references, question)
    print(reasoning)
    print(score)
    