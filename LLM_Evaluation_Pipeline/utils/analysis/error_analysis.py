import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class ErrorAnalyzer:
    def __init__(self, results_file="results/evaluation_results.json"):
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        self.results = data['results']
        self.metrics = data['metrics']
        self.df = pd.DataFrame(self.results)
    
    def generate_report(self):
        """Generate comprehensive error analysis report"""
        report = []
        
        report.append("="*60)
        report.append("LLM EVALUATION - ERROR ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        report.append("QUANTITATIVE ANALYSIS")
        report.append("-"*60)
        report.append(f"Overall Accuracy: {self.metrics['overall_accuracy']*100:.2f}%")
        report.append(f"Total Questions: {self.metrics['total_questions']}")
        report.append(f"Correct Answers: {self.metrics['correct_answers']}")
        report.append(f"Incorrect Answers: {self.metrics['total_questions'] - self.metrics['correct_answers']}")
        report.append(f"Average Response Time: {self.metrics['average_response_time']:.3f}s")
        report.append(f"Median Response Time: {self.metrics['median_response_time']:.3f}s")
        report.append("")
        
        if 'category' in self.df.columns:
            report.append("Per-Category Performance:")
            for category in self.df['category'].unique():
                cat_df = self.df[self.df['category'] == category]
                accuracy = cat_df['is_correct'].mean()
                count = len(cat_df)
                report.append(f"{category}: {accuracy*100:.2f}% ({count} questions)")
            report.append("")
        
        report.append("QUALITATIVE ANALYSIS")
        report.append("-"*60)
        
        incorrect = self.df[self.df['is_correct']==False]
        if len(incorrect) > 0:
            report.append(f"\nCommon Failure Patterns:")
            report.append(f"Total failures: {len(incorrect)}")
            
            answer_dist = Counter(incorrect['model_answer'])
            report.append(f"Most common wrong answers: {dict(answer_dist.most_common(3))}")
            
            report.append("\nExample Incorrect Responses:")
            for idx, row in incorrect.head(3).iterrows():
                report.append(f"\nQuestion: {row['question'][:100]}...")
                report.append(f"Correct Answer: {row['correct_answer']}")
                report.append(f"Model Answer: {row['model_answer']}")
                report.append(f"Model Response: {row['full_response'][:150]}...")
        
        correct = self.df[self.df['is_correct']==True]
        if len(correct) > 0:
            report.append("\n\nExample Correct Responses:")
            for idx, row in correct.head(3).iterrows():
                report.append(f"\nQuestion: {row['question'][:100]}...")
                report.append(f"Correct Answer: {row['correct_answer']}")
                report.append(f"Model Response: {row['full_response'][:150]}...")
        
        report.append("")
        report.append("\n" + "="*60)
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir="results"):
        """Create visualization plots"""
        if 'category' in self.df.columns:
            plt.figure(figsize=(10, 6))
            cat_acc = self.df.groupby('category')['is_correct'].mean()
            cat_acc.plot(kind='bar')
            plt.title('Accuracy by Category')
            plt.xlabel('Category')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/accuracy_by_category.png')
            plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['response_time'], bins=30, edgecolor='black')
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/response_time_distribution.png')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")