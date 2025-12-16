from utils.evaluation.data_loader import load_arc_dataset, save_dataset
import os

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    data = load_arc_dataset(num_samples=100)
    save_dataset(data)