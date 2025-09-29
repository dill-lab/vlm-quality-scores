import torch
import transformers
from tqdm import tqdm

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check if CUDA is available
device = torch.device('mps')

# Lazy loading of the VERA model
_model_cache = {}

def get_vera_score(statements):
    """
    Get plausibility scores for statements using the VERA model. Loads the model only once.

    Parameters:
    - statements (str or list of str): A single statement or a list of statements to evaluate.

    Returns:
    - scores (list of float): Calibrated plausibility scores for each input statement.
    """
    cache_key = 'vera'

    # Lazy loading of the model
    if cache_key not in _model_cache:
        print("Loading VERA model...")
        model_name = 'liujch1998/vera'
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.T5EncoderModel.from_pretrained(model_name).to(device)
        model.D = model.shared.embedding_dim

        # Define the linear layer
        linear = torch.nn.Linear(model.D, 1, dtype=model.dtype)
        linear.weight = torch.nn.Parameter(model.shared.weight[32099, :].unsqueeze(0))
        linear.bias = torch.nn.Parameter(model.shared.weight[32098, 0].unsqueeze(0))
        model.eval()

        # Get temperature for calibration
        temperature = model.shared.weight[32097, 0].item()
        
        # Save loaded objects in cache
        _model_cache[cache_key] = (tokenizer, model, linear, temperature)
        
    # Retrieve the model and related objects from cache
    tokenizer, model, linear, temperature = _model_cache[cache_key]

    # Ensure input is a list
    if isinstance(statements, str):
        statements = [statements]

    # Tokenize the input and move to device
    inputs = tokenizer.batch_encode_plus(
        statements,
        return_tensors='pt',
        padding='longest',
        truncation='longest_first',
        max_length=128
    )
    input_ids = inputs.input_ids.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_ids)
        last_hidden_state = output.last_hidden_state
        hidden = last_hidden_state[:, -1, :]  # Extract hidden state for the last token
        logits = linear(hidden).squeeze(-1)  # Calculate logits
        logits_calibrated = logits / temperature  # Apply temperature calibration
        scores_calibrated = logits_calibrated.sigmoid()  # Convert to probabilities

    # Return scores as a list
    return scores_calibrated.tolist()


def commonsense_analysis_all_datasets(rationale_column, dataset_list):
    """
    Perform commonsense analysis on the rationales in the given datasets.

    Parameters:
    - rationale_column (str): The name of the column containing the rationales.
    - dataset_list (list of pd.DataFrame): A list of datasets to analyze.
    - overwrite_candidate_answers (bool): Whether to overwrite the candidate answers in the dataset.
    """
    for dataset in tqdm(dataset_list, desc="Commonsense analysis"):
        dataset['commonsense_plausibility'] = get_vera_score(list(dataset[rationale_column]))
    print("Commonsense analysis completed.")

def main():
    # Test
    statements = [
        "Water freezes at 0 degrees Celsius under normal atmospheric pressure.",
        "The sun rises in the west.",
        "The sun rises in the east.",
        "Since the density of a marble is much less than the density of mercury, the marble would sink to the bottom of the bowl if placed in it.",
        "Since the density of a marble is much more than the density of water, the marble would sink to the bottom of the bowl if placed in it.",
        "Since the density of water is much less than the density of a marble, the marble would sink to the bottom of the bowl if placed in it."
    ]

    # Get plausibility scores
    scores = get_vera_score(statements)

    # Print the results
    for statement, score in zip(statements, scores):
        print(f"Statement: {statement}")
        print(f"Plausibility score: {score}")

if __name__ == '__main__':
    main()
