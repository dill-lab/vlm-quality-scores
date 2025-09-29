from globals import INFORMATIVENESS_PROMPT
from lm_loader import create_model_instance
import ast
import pandas as pd
from tqdm import tqdm
from support_contrastiveness_analysis import generate_hypotheses


informativeness_model_name = "gpt-4o"

def extract_distinct_rationale_pieces(hypothesis, rationale, max_retries=5):
    # Choose a model to use for the informativeness analysis
    model = create_model_instance(informativeness_model_name)
    
    retries = 0
    while retries < max_retries:
        try:
            prompt_messages = [{"role": "user", "content": INFORMATIVENESS_PROMPT.format(hypothesis=hypothesis, rationale=rationale)}]
            
            response = model.chat_completion(prompt_messages)
            message = response['choices'][0]['message']['content']
            start_index = message.find('[')
            end_index = message.rfind(']')
            R_list_str = message[start_index:end_index+1]
            R_list = ast.literal_eval(R_list_str)
            return R_list
        
        except Exception as e:
            print(f"Attempt {retries + 1} failed with error: {e}")
            retries += 1
    
    # If all attempts fail, return an empty list or handle it as needed
    print("All attempts failed. Returning an empty list.")
    return []

def informative_analysis(rationale_column_name, df):    
    informative_list = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Extract distinct rationale pieces
        hypothesis = row['hypothesis']
        rationale = row[rationale_column_name]
        distinct_rationale_pieces = extract_distinct_rationale_pieces(hypothesis, rationale)
        
        if len(distinct_rationale_pieces) > 0:
            informative_list.append(1)
        else:
            informative_list.append(0)
        print(f"Row {idx} is informative: {informative_list[-1]}")
    df['informative'] = informative_list

def informativeness_analysis_all_datasets(rationale_column_name, dataset_list):
    for dataset in dataset_list:
        informative_analysis(rationale_column_name, dataset)

# Example usage
def main():
    print("Running demo of informativeness analysis")
    file_path = "model_outputs/VizWiz/llava-v1.5-7b_test.csv"
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    informative_analysis('rationale', df)
    
    # Save the updated dataframe
    df.to_csv(file_path, index=False)  
    
if __name__ == "__main__":
    main()
    
