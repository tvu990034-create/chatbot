"""
Simple score tracker for benchmark progress
Shows only correct/wrong counts and accuracy
"""

import json
import sys

def track_score():
    try:
        with open('comprehensive_benchmark_checkpoint.json', 'r') as f:
            data = json.load(f)
        
        results = data.get('results', [])
        
        correct = sum(1 for r in results if r.get('is_correct'))
        wrong = sum(1 for r in results if not r.get('is_correct'))
        total = len(results)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        print(f"Correct: {correct}")
        print(f"Wrong: {wrong}")
        print(f"Accuracy: {accuracy:.1f}%")
        
    except FileNotFoundError:
        print("No checkpoint file found yet")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    track_score()