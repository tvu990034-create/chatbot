"""
Continuous score tracker for benchmark progress
Updates automatically every few seconds until benchmark finishes
"""

import json
import time
import sys
import os

def continuous_score_tracker():
    print("=" * 40)
    print("CONTINUOUS SCORE TRACKER")
    print("=" * 40)
    print("Press Ctrl+C to stop tracking")
    print("=" * 40)
    
    last_total = 0
    no_change_count = 0
    
    try:
        while True:
            try:
                with open('comprehensive_benchmark_checkpoint.json', 'r') as f:
                    data = json.load(f)
                
                results = data.get('results', [])
                total_processed = data.get('total_processed', len(results))
                
                correct = sum(1 for r in results if r.get('is_correct'))
                wrong = sum(1 for r in results if not r.get('is_correct'))
                total = len(results)
                accuracy = (correct / total * 100) if total > 0 else 0
                
                # Check if benchmark is complete
                if total_processed >= 12063:  # Total questions expected
                    print("\n" + "=" * 40)
                    print("BENCHMARK COMPLETE!")
                    print("=" * 40)
                    print(f"Final Score:")
                    print(f"Correct: {correct}")
                    print(f"Wrong: {wrong}")
                    print(f"Accuracy: {accuracy:.1f}%")
                    print("=" * 40)
                    break
                
                # Clear previous output and show current status
                os.system('cls' if os.name == 'nt' else 'clear')
                
                print("=" * 40)
                print("LIVE SCORE TRACKING")
                print("=" * 40)
                print(f"Questions: {total_processed}/12063")
                print(f"Progress: {(total_processed/12063*100):.1f}%")
                print("-" * 40)
                print(f"Correct: {correct}")
                print(f"Wrong: {wrong}")
                print(f"Accuracy: {accuracy:.1f}%")
                print("=" * 40)
                
                # Check if benchmark is still running
                if total == last_total:
                    no_change_count += 1
                    if no_change_count > 10:
                        print("Waiting for benchmark to continue...")
                else:
                    no_change_count = 0
                    last_total = total
                
                time.sleep(3)  # Update every 3 seconds
                
            except FileNotFoundError:
                print("Waiting for checkpoint file...")
                time.sleep(5)
            except json.JSONDecodeError:
                print("Checkpoint file being updated...")
                time.sleep(2)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
                
    except KeyboardInterrupt:
        print("\n\nTracking stopped by user")
        print(f"Final score at stop:")
        print(f"Correct: {correct}")
        print(f"Wrong: {wrong}")
        print(f"Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    continuous_score_tracker()