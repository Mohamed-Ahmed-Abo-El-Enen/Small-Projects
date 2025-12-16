import os
import numpy as np
import requests
import json
import time
from typing import List, Dict
import pandas as pd


def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class MCQEvaluator:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.results = []
    
    def evaluate_dataset(self, mcq_data: List[Dict]):
        """Evaluate entire MCQ dataset"""
        print(f"Evaluating {len(mcq_data)} questions...")
        
        for i, item in enumerate(mcq_data):
            print(f"Processing question {i+1}/{len(mcq_data)}...", end='\r')
            
            try:
                response = requests.post(
                    f"{self.api_url}/evaluate_mcq",
                    json={
                        "question": item['question'],
                        "choices": item['choices'],
                        "correct_answer": item['correct_answer']
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.results.append({
                        "question": item['question'],
                        "category": item.get('category', 'Unknown'),
                        "correct_answer": item['correct_answer'],
                        "model_answer": result['model_answer'],
                        "is_correct": result['is_correct'],
                        "response_time": result['response_time'],
                        "full_response": result['full_response']
                    })
                else:
                    print(f"\nError on question {i+1}: {response.status_code}")
                    
            except Exception as e:
                print(f"\nError processing question {i+1}: {e}")
                
            time.sleep(0.1)
        
        print(f"\nCompleted evaluation of {len(self.results)} questions")
        return self.results
    
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        metrics = {
            "overall_accuracy": df['is_correct'].mean(),
            "total_questions": len(df),
            "correct_answers": df['is_correct'].sum(),
            "average_response_time": df['response_time'].mean(),
            "median_response_time": df['response_time'].median()
        }
        
        if 'category' in df.columns:
            category_acc = df.groupby('category')['is_correct'].agg(['mean', 'count'])
            metrics['category_accuracy'] = category_acc.to_dict()
        
        return metrics

    def save_results(self, filename="results/evaluation_results.json"):
        """Save results to file"""
        output = {
            "results": self.results,
            "metrics": self.calculate_metrics()
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=convert_to_serializable)

        print(f"Results saved to {filename}")
