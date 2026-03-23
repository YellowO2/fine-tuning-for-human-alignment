# Datasets

Contains the datasets used in this project for evaluating LLM humor judgment.

## File structure

- `jester/`: Contains the processed Jester joke dataset files. The base data features jokes rated by users. We transform these into pairwise comparisons where the model must choose the funnier joke (expected output: "A" or "B").
- `newyorker/`: Contains the New Yorker Caption Contest dataset splits. The task is to evaluate which caption is funnier for a given cartoon description. The datasets are restructured for a pairwise ranking task (expected output: "A" or "B").

## Preparation
We provide scripts to rebuild or prepare the datasets in `scripts/prepare_datasets/`:
- `prepare_jester_dataset.py`: Cleans and ranks the Jester jokes.
- `prepare_newyorker_dataset.py`: Downloads and processes the New Yorker dataset into folds with the expected target set to "A" or "B".

Note: When evaluating, the target `expected` field will strictly map to `A` or `B` to match the LLM prompt's generation structure.

