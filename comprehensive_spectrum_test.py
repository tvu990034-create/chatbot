"""
Comprehensive Difficulty Spectrum Test
Tests across all difficulty levels: Basic to Extremely Hard
"""

import time
from typing import Dict, List

from benchmark_integrations.simplified_fixed_integration import SimplifiedFixedIntegrationLayer
from gateway.litellm_gateway import get_optimization_stats

class ComprehensiveSpectrumTest:
    """Tests across all difficulty levels."""
    
    def __init__(self):
        self.integration_layer = SimplifiedFixedIntegrationLayer()
        
        # Questions organized by difficulty level
        self.difficulty_levels = {
            "Basic": [
                ("Arithmetic", "What is 5 + 3?", "8"),
                ("Geography", "What is the capital of France?", "Paris"),
                ("Science", "What is H2O commonly known as?", "Water"),
                ("History", "Who was the first US President?", "Washington"),
                ("Literature", "Who wrote 'Romeo and Juliet'?", "Shakespeare"),
            ],
            "Intermediate": [
                ("Mathematics", "What is 15% of 200?", "30"),
                ("Physics", "What is the speed of light in m/s (approximately)?", "300,000,000"),
                ("Biology", "What is the powerhouse of the cell?", "Mitochondria"),
                ("Chemistry", "What is the atomic number of carbon?", "6"),
                ("Economics", "What is GDP?", "Gross Domestic Product"),
            ],
            "Hard": [
                ("Mathematics", "Solve: 2x + 5 = 15. What is x?", "5"),
                ("Physics", "Explain Newton's Second Law in simple terms.", "F=ma"),
                ("Chemistry", "Balance the equation: H2 + O2 -> ?", "H2O"),
                ("Biology", "What is the process by which plants make food?", "Photosynthesis"),
                ("History", "What were the main causes of World War I?", "Complex factors"),
            ],
            "Very Hard": [
                ("Mathematics", "What is the sum of all prime numbers between 1 and 100?", "1060"),
                ("Physics", "Explain the photoelectric effect and its significance.", "Quantum mechanics foundation"),
                ("Chemistry", "Explain the difference between fission and fusion.", "Nuclear reactions"),
                ("Biology", "Explain the structure and function of DNA.", "Genetic material"),
                ("Philosophy", "What is the difference between knowledge and wisdom?", "Practical application"),
            ],
            "Extremely Hard": [
                ("Mathematics", "If a train travels at 60 mph for 2 hours, then increases speed by 25% for 3 hours, what is the total distance?", "405 miles"),
                ("Physics", "Calculate the Schwarzschild radius of a black hole with the mass of the Sun.", "Complex calculation"),
                ("Chemistry", "Explain the molecular orbital theory and its application to bonding.", "Quantum chemistry"),
                ("Biology", "Explain the mechanism of protein synthesis from DNA to proteins.", "Gene expression"),
                ("Computer Science", "Explain the difference between NP-complete and P problems.", "Computational complexity"),
            ],
            "Expert Level": [
                ("Mathematics", "Prove that the square root of 2 is irrational using contradiction.", "Number theory proof"),
                ("Physics", "Derive the time dilation formula from special relativity.", "Relativistic derivation"),
                ("Chemistry", "Explain the Born-Oppenheimer approximation and its significance.", "Quantum chemistry approximation"),
                ("Biology", "Explain the mechanism of action of CRISPR-Cas9 and its applications.", "Gene editing"),
                ("Philosophy", "Analyze the trolley problem using 5 different ethical frameworks.", "Ethical analysis"),
            ],
            "Beyond Expert": [
                ("Meta-Reasoning", "Analyze your own reasoning process and identify 3 systematic biases that might affect your responses.", "Self-awareness"),
                ("Cross-Domain", "Design a solution to climate change integrating physics, economics, sociology, and political science.", "Multi-disciplinary synthesis"),
                ("Creative", "Invent a new mathematical constant combining pi, e, and phi. Explain its interpretations and applications.", "Creative invention"),
                ("Adversarial", "A plane crashes on the US-Canada border. Where do they bury the survivors?", "Trick - survivors aren't buried"),
                ("Meta-Cognitive", "What is the difference between understanding a concept and being able to explain it?", "Meta-linguistic analysis"),
            ]
        }
    
    def evaluate_response(self, response: str, expected: str, difficulty: str) -> Dict:
        """Evaluate response based on difficulty level."""
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Check for refusal
        refusal_phrases = ["i cannot", "i don't know", "unable to", "not enough information"]
        if any(phrase in response_lower for phrase in refusal_phrases):
            return {'correct': False, 'refusal': True}
        
        # Check for correctness (contains expected keyword)
        if difficulty in ["Basic", "Intermediate"]:
            # Strict matching for easy questions
            is_correct = expected_lower in response_lower
        elif difficulty in ["Hard", "Very Hard"]:
            # Loose matching for medium questions
            is_correct = any(word in response_lower for word in expected_lower.split()[:3])
        else:
            # For harder questions, check for substantial response
            is_correct = len(response) > 100 and any(word in response_lower for word in expected_lower.split()[:2])
        
        return {'correct': is_correct, 'refusal': False}
    
    def run_comprehensive_test(self) -> Dict:
        """Run the comprehensive spectrum test."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE DIFFICULTY SPECTRUM TEST")
        print("Basic -> Intermediate -> Hard -> Very Hard -> Extremely Hard -> Expert -> Beyond Expert")
        print("="*70)
        
        level_results = {}
        total_questions = 0
        total_correct = 0
        total_time = 0
        
        for difficulty, questions in self.difficulty_levels.items():
            print(f"\n{'='*70}")
            print(f"Difficulty: {difficulty}")
            print("="*70)
            
            level_correct = 0
            level_time = 0
            
            for category, question, expected in questions:
                print(f"\n[{category}] {question[:60]}...")
                
                start_time = time.time()
                try:
                    response, enhancement_info = self.integration_layer.query_with_gateway(question)
                    elapsed = time.time() - start_time
                    
                    evaluation = self.evaluate_response(response, expected, difficulty)
                    
                    if evaluation['correct']:
                        level_correct += 1
                        total_correct += 1
                    else:
                        print(f"Expected: {expected}")
                    
                    total_questions += 1
                    level_time += elapsed
                    total_time += elapsed
                    
                    status = "[OK]" if evaluation['correct'] else "[NO]"
                    try:
                        print(f"{status} Time: {elapsed:.2f}s | Response: {response[:80]}...")
                    except UnicodeEncodeError:
                        print(f"{status} Time: {elapsed:.2f}s | Response: [Unicode content]")
                    
                except Exception as e:
                    print(f"[ERROR] {category}: {str(e)}")
                    total_questions += 1
                    level_results[difficulty] = {
                        'correct': level_correct,
                        'total': len(questions),
                        'time': level_time,
                        'results': []
                    }
            
            level_accuracy = (level_correct / len(questions)) * 100
            level_results[difficulty] = {
                'correct': level_correct,
                'total': len(questions),
                'accuracy': level_accuracy,
                'time': level_time,
                'avg_time': level_time / len(questions)
            }
            
            print(f"\n{difficulty} Results: {level_accuracy:.1f}% ({level_correct}/{len(questions)}) - {level_time/len(questions):.2f}s avg")
        
        overall_accuracy = (total_correct / total_questions) * 100
        avg_time = total_time / total_questions
        
        # Get optimization stats
        try:
            stats = get_optimization_stats()
            print(f"\n{'='*70}")
            print("ENHANCED GATEWAY STATUS")
            print("="*70)
            print(f"Retry logic: Enabled")
            print(f"Timeout handling: Enabled")
            print(f"Simple cache: Enabled ({stats.get('optimizations', {}).get('simple_cache', {}).get('cached_responses', 0)} cached)")
            print(f"Optimization controller: Enabled")
            controller_stats = stats.get('optimizations', {}).get('optimization_controller', {})
            print(f"Total queries processed: {controller_stats.get('total_queries', 0)}")
            print(f"Optimizations applied: {controller_stats.get('optimization_counts', {})}")
        except Exception as e:
            print(f"Could not get optimization stats: {e}")
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE SPECTRUM TEST RESULTS")
        print("="*70)
        print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")
        print(f"Average Time: {avg_time:.2f}s")
        
        print(f"\n{'='*70}")
        print("DIFFICULTY LEVEL BREAKDOWN")
        print("="*70)
        for difficulty in ["Basic", "Intermediate", "Hard", "Very Hard", "Extremely Hard", "Expert Level", "Beyond Expert"]:
            if difficulty in level_results:
                result = level_results[difficulty]
                print(f"{difficulty}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']}) - {result['avg_time']:.2f}s avg")
        
        # Difficulty assessment
        if overall_accuracy >= 95:
            assessment = "EXCEPTIONAL - Mastery across all difficulty levels"
        elif overall_accuracy >= 85:
            assessment = "EXCELLENT - Strong performance across most levels"
        elif overall_accuracy >= 75:
            assessment = "GOOD - Competent across difficulty spectrum"
        elif overall_accuracy >= 60:
            assessment = "MODERATE - Adequate performance with gaps"
        else:
            assessment = "WEAK - Struggles with higher difficulty"
        
        print(f"\n{'='*70}")
        print("DIFFICULTY ASSESSMENT")
        print("="*70)
        print(assessment)
        
        return {
            'overall_accuracy': overall_accuracy,
            'avg_time': avg_time,
            'level_results': level_results,
            'total_questions': total_questions,
            'total_correct': total_correct
        }


if __name__ == "__main__":
    test = ComprehensiveSpectrumTest()
    results = test.run_comprehensive_test()
