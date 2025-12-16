from utils.evaluation.evaluate import MCQEvaluator
from utils.evaluation.data_loader import load_dataset_from_file
import os

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    evaluator = MCQEvaluator()
    data = load_dataset_from_file()
    evaluator.evaluate_dataset(data)
    evaluator.save_results()