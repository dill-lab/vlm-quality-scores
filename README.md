1. Run the following command:
```
python3 load_dataset.py
```
to load the dataset. The dataset is stored in the `../datasets` folder. 
The dataset contains the first 500 instances in A-OKVQA validation set and VizWiz validation set.

2. Run
```
python3 vqa_infer.py --dataset both --model all --rewrite_file
```
to generate models' predictions on the dataset. The predictions are stored in the `model_outputs` folder.

3. Then run
```
python3 rationale_quality_analysis.py --dataset both --model all --rewrite_file
```
to generate the rationale quality analysis results. The results are stored in the `results` folder.