## Code for the paper _**Believing without Seeing: Quality Scores for Contextualizing Vision-Language Model Explanations**_ (https://arxiv.org/abs/2509.25844)

### Setup

1) Init submodules:
```bash
git submodule update --init --recursive
```

2) Create and activate a virtual environment, then install deps:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

3) Configure environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `VLMQS_DATASETS_DIR`: Path to datasets dir (default: `data/`).
- `VLMQS_OUTPUTS_DIR`: Path to outputs dir (default: `model_outputs/`).
- `VLMQS_COST_FILE`: Path to track API costs (default: `total_cost.txt`).

### Data preparation
Download subsets and materialize CSVs and images:
```bash
python3 load_dataset.py --dataset all
```
This creates `data/AOKVQA/AOKVQA.csv` and `data/VizWiz/VizWiz.csv` with local image paths.

### Inference
Generate predictions and rationales. Example (all models, both datasets):
```bash
python3 vqa_infer.py --dataset all --model all --rewrite_file
```
Outputs will be saved under `model_outputs/<DATASET>/<MODEL>.csv`.

For a quick test run (20 samples):
```bash
python3 vqa_infer.py --dataset VizWiz --model qwen --test --rewrite_file
```

### Rationale quality analyses
Run support/contrastiveness, visual fidelity, informativeness, and commonsense:
```bash
python3 rationale_quality_analysis.py --dataset all --quality all
```
This updates each `model_outputs/<DATASET>/<MODEL>.csv` with new columns.
