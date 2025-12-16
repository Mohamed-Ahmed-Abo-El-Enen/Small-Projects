import json
from datasets import load_dataset
import random


def load_arc_dataset(split="test", num_samples=100):
    """Load ARC-Easy dataset"""
    dataset = load_dataset("ai2_arc", "ARC-Easy", split=split)
    
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    
    mcq_data = []
    for item in dataset:
        choices_dict = {}
        for i, choice in enumerate(item['choices']['text']):
            label = item['choices']['label'][i]
            choices_dict[label] = choice
        
        mcq_data.append({
            "question": item['question'],
            "choices": choices_dict,
            "correct_answer": item['answerKey'],
            "category": "Science"
        })
    
    return mcq_data


def load_mmlu_dataset(subject="abstract_algebra", split="test", num_samples=100):
    """Load MMLU dataset"""
    try:
        dataset = load_dataset("cais/mmlu", subject, split=split)
        
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)
        
        mcq_data = []
        for item in dataset:
            choices_dict = {
                'A': item['choices'][0],
                'B': item['choices'][1],
                'C': item['choices'][2],
                'D': item['choices'][3]
            }
            
            answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
            
            mcq_data.append({
                "question": item['question'],
                "choices": choices_dict,
                "correct_answer": answer_map[item['answer']],
                "category": subject
            })
        
        return mcq_data
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return []


def save_dataset(data, filename="data/mcq_dataset.json"):
    """Save dataset to JSON"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_dataset_from_file(filename="data/mcq_dataset.json"):
    """Load dataset from JSON"""
    with open(filename, 'r') as f:
        return json.load(f)