"""
Speed Challenge
Tests ability to provide accurate answers under extreme time pressure
"""

import time
from typing import Dict
from benchmark_integrations.simplified_fixed_integration import SimplifiedFixedIntegrationLayer
from gateway.litellm_gateway import get_optimization_stats

class SpeedChallenge:
    """Tests performance under time constraints."""
    
    def __init__(self):
        self.integration_layer = SimplifiedFixedIntegrationLayer()
        
        # Each question has a target time limit
        self.speed_questions = [
            {
                "question": "What is 23 times 47?",
                "target_time": 2.0,
                "expected": "1081",
                "category": "Arithmetic"
            },
            {
                "question": "What is the capital of France?",
                "target_time": 1.0,
                "expected": "Paris",
                "category": "Geography"
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "target_time": 1.0,
                "expected": "Shakespeare",
                "category": "Literature"
            },
            {
                "question": "What is the chemical symbol for gold?",
                "target_time": 1.0,
                "expected": "Au",
                "category": "Chemistry"
            },
            {
                "question": "What year did World War II end?",
                "target_time": 1.0,
                "expected": "1945",
                "category": "History"
            },
            {
                "question": "What is the square root of 144?",
                "target_time": 1.0,
                "expected": "12",
                "category": "Mathematics"
            },
            {
                "question": "What is the largest planet in our solar system?",
                "target_time": 1.0,
                "expected": "Jupiter",
                "category": "Astronomy"
            },
            {
                "question": "Who painted the Mona Lisa?",
                "target_time": 1.0,
                "expected": "Da Vinci",
                "category": "Art"
            },
            {
                "question": "What is the boiling point of water in Celsius?",
                "target_time": 1.0,
                "expected": "100",
                "category": "Physics"
            },
            {
                "question": "How many continents are there?",
                "target_time": 1.0,
                "expected": "7",
                "category": "Geography"
            },
            {
                "question": "What is the speed of light in km/s (approximately)?",
                "target_time": 2.0,
                "expected": "300000",
                "category": "Physics"
            },
            {
                "question": "Who discovered penicillin?",
                "target_time": 1.0,
                "expected": "Fleming",
                "category": "Biology"
            },
            {
                "question": "What is the currency of Japan?",
                "target_time": 1.0,
                "expected": "Yen",
                "category": "Economics"
            },
            {
                "question": "What is the formula for the area of a circle?",
                "target_time": 1.0,
                "expected": "πr²",
                "category": "Mathematics"
            },
            {
                "question": "Who was the first person to walk on the moon?",
                "target_time": 1.0,
                "expected": "Armstrong",
                "category": "Space"
            },
            {
                "question": "What is the hardest natural substance on Earth?",
                "target_time": 1.0,
                "expected": "Diamond",
                "category": "Geology"
            },
            {
                "question": "What year did the Titanic sink?",
                "target_time": 1.0,
                "expected": "1912",
                "category": "History"
            },
            {
                "question": "What is the largest ocean on Earth?",
                "target_time": 1.0,
                "expected": "Pacific",
                "category": "Geography"
            },
            {
                "question": "Who composed the 'Moonlight Sonata'?",
                "target_time": 1.0,
                "expected": "Beethoven",
                "category": "Music"
            },
            {
                "question": "What is the atomic number of carbon?",
                "target_time": 1.0,
                "expected": "6",
                "category": "Chemistry"
            }
        ]
    
    def evaluate_speed_response(self, response: str, expected: str, actual_time: float, target_time: float) -> Dict:
        """Evaluate speed and accuracy."""
        # Check if answer is correct (contains expected keyword)
        is_correct = expected.lower() in response.lower()
        
        # Check if within time limit
        within_limit = actual_time <= target_time
        
        # Calculate time performance (ratio of target to actual)
        time_performance = min(100, (target_time / actual_time) * 100) if actual_time > 0 else 0
        
        return {
            'correct': is_correct,
            'within_limit': within_limit,
            'time_performance': time_performance,
            'overall': (100 if is_correct else 0) * (100 if within_limit else time_performance / 100) / 100
        }
    
    def run_speed_challenge(self) -> Dict:
        """Run the speed challenge."""
        print(f"\n{'='*70}")
        print("SPEED CHALLENGE")
        print("Tests accuracy under extreme time pressure (1-2 second limits)")
        print("="*70)
        
        results = []
        total_time = 0
        correct_count = 0
        within_limit_count = 0
        
        for i, item in enumerate(self.speed_questions, 1):
            print(f"\n[{i}/{len(self.speed_questions)}] {item['category']}: {item['question']}")
            print(f"Target: {item['target_time']}s")
            
            start_time = time.time()
            try:
                response, enhancement_info = self.integration_layer.query_with_gateway(item['question'])
                actual_time = time.time() - start_time
                
                evaluation = self.evaluate_speed_response(
                    response, item['expected'], actual_time, item['target_time']
                )
                
                if evaluation['correct']:
                    correct_count += 1
                if evaluation['within_limit']:
                    within_limit_count += 1
                
                total_time += actual_time
                
                results.append({
                    'question': item['question'],
                    'category': item['category'],
                    'target_time': item['target_time'],
                    'actual_time': actual_time,
                    'response': response,
                    'evaluation': evaluation
                })
                
                status = "✓" if evaluation['correct'] else "✗"
                time_status = "⚡" if evaluation['within_limit'] else "⏱"
                print(f"{status}{time_status} Time: {actual_time:.2f}s | Response: {response[:60]}...")
                
            except Exception as e:
                print(f"[ERROR] {item['category']}: {str(e)}")
                results.append({
                    'question': item['question'],
                    'category': item['category'],
                    'target_time': item['target_time'],
                    'actual_time': 0,
                    'response': f"Error: {str(e)}",
                    'evaluation': {'correct': False, 'within_limit': False, 'time_performance': 0, 'overall': 0},
                    'error': str(e)
                })
        
        accuracy = (correct_count / len(self.speed_questions)) * 100
        within_limit_rate = (within_limit_count / len(self.speed_questions)) * 100
        avg_time = total_time / len(self.speed_questions)
        
        # Get optimization stats
        try:
            stats = get_optimization_stats()
            print(f"\n{'='*70}")
            print("MINIMAL GATEWAY STATUS")
            print("="*70)
            print(f"Retry logic: Enabled")
            print(f"Timeout handling: Enabled")
        except Exception as e:
            print(f"Could not get optimization stats: {e}")
        
        print(f"\n{'='*70}")
        print("SPEED CHALLENGE RESULTS")
        print("="*70)
        print(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(self.speed_questions)})")
        print(f"Within Time Limit: {within_limit_rate:.1f}% ({within_limit_count}/{len(self.speed_questions)})")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Average Target: {sum(q['target_time'] for q in self.speed_questions) / len(self.speed_questions):.2f}s")
        
        # Category breakdown
        category_results = {}
        for result in results:
            category = result['category']
            if category not in category_results:
                category_results[category] = {'correct': 0, 'total': 0, 'within_limit': 0}
            category_results[category]['total'] += 1
            if result['evaluation']['correct']:
                category_results[category]['correct'] += 1
            if result['evaluation']['within_limit']:
                category_results[category]['within_limit'] += 1
        
        print(f"\n{'='*70}")
        print("CATEGORY-BY-CATEGORY RESULTS")
        print("="*70)
        for category, counts in sorted(category_results.items()):
            acc = (counts['correct'] / counts['total']) * 100
            limit_rate = (counts['within_limit'] / counts['total']) * 100
            print(f"{category}: {acc:.1f}% accuracy, {limit_rate:.1f}% within limit")
        
        # Assessment
        if accuracy >= 90 and within_limit_rate >= 50:
            assessment = "EXCEPTIONAL - Fast and accurate"
        elif accuracy >= 75 and within_limit_rate >= 30:
            assessment = "EXCELLENT - Good speed-accuracy tradeoff"
        elif accuracy >= 60:
            assessment = "GOOD - Accurate but slow"
        elif within_limit_rate >= 60:
            assessment = "MODERATE - Fast but inaccurate"
        else:
            assessment = "WEAK - Slow and inaccurate"
        
        print(f"\n{'='*70}")
        print("DIFFICULTY ASSESSMENT")
        print("="*70)
        print(assessment)
        
        return {
            'accuracy': accuracy,
            'within_limit_rate': within_limit_rate,
            'avg_time': avg_time,
            'category_results': category_results,
            'results': results
        }


if __name__ == "__main__":
    challenge = SpeedChallenge()
    results = challenge.run_speed_challenge()
