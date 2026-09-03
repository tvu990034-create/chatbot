"""
Adversarial Challenge
Tests ability to handle tricky, confusing, and adversarial questions
"""

import time
from typing import Dict
from benchmark_integrations.simplified_fixed_integration import SimplifiedFixedIntegrationLayer
from gateway.litellm_gateway import get_optimization_stats

class AdversarialChallenge:
    """Tests ability to handle adversarial inputs."""
    
    def __init__(self):
        self.integration_layer = SimplifiedFixedIntegrationLayer()
        
        # Adversarial questions designed to trick or confuse
        self.adversarial_questions = [
            {
                "question": "If a plane crashes on the border between the US and Canada, where do they bury the survivors?",
                "trap": "Trick question - survivors are not buried",
                "category": "Trick Question"
            },
            {
                "question": "Which month has 28 days?",
                "trap": "All months have 28 days",
                "category": "Trick Question"
            },
            {
                "question": "A farmer has 10 sheep. All but 9 die. How many sheep are left?",
                "trap": "9 sheep are left (not 1)",
                "category": "Trick Question"
            },
            {
                "question": "If you take 2 apples from 3 apples, how many do you have?",
                "trap": "You have 2 apples (not 1)",
                "category": "Trick Question"
            },
            {
                "question": "Is it possible for a man to marry his widow's sister?",
                "trap": "Impossible - if he has a widow, he's dead",
                "category": "Logical Trap"
            },
            {
                "question": "What was the President's name in 1950?",
                "trap": "Same as current president (Truman)",
                "category": "Historical Trap"
            },
            {
                "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "trap": "5 minutes (parallel processing)",
                "category": "Parallel Processing Trap"
            },
            {
                "question": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                "trap": "Not 10 cents - the ball costs 5 cents",
                "category": "Cognitive Bias Trap"
            },
            {
                "question": "If you're running a race and you pass the person in second place, what place are you in?",
                "trap": "Second place (not first)",
                "category": "Reasoning Trap"
            },
            {
                "question": "What comes down but never goes up?",
                "trap": "Rain (or age, temperature, etc.)",
                "category": "Riddle"
            },
            {
                "question": "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?",
                "trap": "A map",
                "category": "Riddle"
            },
            {
                "question": "The more you take, the more you leave behind. What am I?",
                "trap": "Footsteps",
                "category": "Riddle"
            },
            {
                "question": "What can you catch but not throw?",
                "trap": "A cold (or a ball)",
                "category": "Riddle"
            },
            {
                "question": "If you have a bowl with six apples and you take away four, how many do you have?",
                "trap": "You have 4 apples (not 2)",
                "category": "Trick Question"
            },
            {
                "question": "How many birthdays does the average person have?",
                "trap": "One per year (many in lifetime)",
                "category": "Language Ambiguity"
            },
            {
                "question": "Some months have 30 days, some have 31. How many have 28?",
                "trap": "All 12 months",
                "category": "Trick Question"
            },
            {
                "question": "If a doctor gives you three pills and tells you to take one every half hour, how long will they last?",
                "trap": "One hour (not 1.5 hours)",
                "category": "Mathematical Trap"
            },
            {
                "question": "What is full of holes but still holds water?",
                "trap": "A sponge",
                "category": "Riddle"
            },
            {
                "question": "What can travel around the world while staying in a corner?",
                "trap": "A stamp",
                "category": "Riddle"
            },
            {
                "question": "If you were in a race and passed the person in last place, what place would you be in?",
                "trap": "Last place is impossible to pass (you'd be last)",
                "category": "Logical Impossibility"
            }
        ]
    
    def evaluate_adversarial_response(self, response: str, trap: str) -> Dict:
        """Evaluate if the model avoided the trap."""
        response_lower = response.lower()
        trap_lower = trap.lower()
        
        # Check if response acknowledges the trap
        trap_indicators = ['trick', 'not', 'actually', 'however', 'but', 'although', 'correct', 'right']
        avoided = any(indicator in response_lower for indicator in trap_indicators)
        
        # Check if response gives the trapped answer
        trapped_keywords = {
            'Trick Question': ['10 cents', '1', '2', '1.5 hours'],
            'Logical Trap': ['first place', 'yes'],
            'Historical Trap': ['eisenhower', 'different'],
            'Parallel Processing Trap': ['100 minutes', '20 minutes'],
            'Cognitive Bias Trap': ['10 cents', '$0.10'],
            'Reasoning Trap': ['first place'],
            'Logical Impossibility': ['second to last', 'second last']
        }
        
        is_trapped = False
        for category, keywords in trapped_keywords.items():
            if any(keyword in response_lower for keyword in keywords):
                is_trapped = True
                break
        
        # Check for substantial response
        is_substantial = len(response) > 50
        
        return {
            'avoided': avoided,
            'trapped': is_trapped,
            'substantial': is_substantial,
            'overall': (100 if avoided and not is_trapped and is_substantial else 50 if avoided and is_substantial else 0)
        }
    
    def run_adversarial_challenge(self) -> Dict:
        """Run the adversarial challenge."""
        print(f"\n{'='*70}")
        print("ADVERSARIAL CHALLENGE")
        print("Tests ability to handle tricky, confusing, and adversarial questions")
        print("="*70)
        
        results = []
        total_time = 0
        avoided_count = 0
        trapped_count = 0
        
        for i, item in enumerate(self.adversarial_questions, 1):
            print(f"\n[{i}/{len(self.adversarial_questions)}] {item['category']}: {item['question']}")
            print(f"Trap: {item['trap']}")
            
            start_time = time.time()
            try:
                response, enhancement_info = self.integration_layer.query_with_gateway(item['question'])
                actual_time = time.time() - start_time
                
                evaluation = self.evaluate_adversarial_response(response, item['trap'])
                
                if evaluation['avoided']:
                    avoided_count += 1
                if evaluation['trapped']:
                    trapped_count += 1
                
                total_time += actual_time
                
                results.append({
                    'question': item['question'],
                    'category': item['category'],
                    'trap': item['trap'],
                    'time': actual_time,
                    'response': response,
                    'evaluation': evaluation
                })
                
                status = "[OK]" if evaluation['avoided'] else "[NO]"
                trap_status = "[TRAPPED]" if evaluation['trapped'] else "[SAFE]"
                print(f"{status} {trap_status} Time: {actual_time:.2f}s | Response: {response[:80]}...")
                
            except Exception as e:
                print(f"[ERROR] {item['category']}: {str(e)}")
                results.append({
                    'question': item['question'],
                    'category': item['category'],
                    'trap': item['trap'],
                    'time': 0,
                    'response': f"Error: {str(e)}",
                    'evaluation': {'avoided': False, 'trapped': True, 'substantial': False, 'overall': 0},
                    'error': str(e)
                })
        
        avoidance_rate = (avoided_count / len(self.adversarial_questions)) * 100
        trap_rate = (trapped_count / len(self.adversarial_questions)) * 100
        avg_time = total_time / len(self.adversarial_questions)
        
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
        print("ADVERSARIAL CHALLENGE RESULTS")
        print("="*70)
        print(f"Trap Avoidance: {avoidance_rate:.1f}% ({avoided_count}/{len(self.adversarial_questions)})")
        print(f"Trapped Rate: {trap_rate:.1f}% ({trapped_count}/{len(self.adversarial_questions)})")
        print(f"Average Time: {avg_time:.2f}s")
        
        # Category breakdown
        category_results = {}
        for result in results:
            category = result['category']
            if category not in category_results:
                category_results[category] = {'avoided': 0, 'total': 0, 'trapped': 0}
            category_results[category]['total'] += 1
            if result['evaluation']['avoided']:
                category_results[category]['avoided'] += 1
            if result['evaluation']['trapped']:
                category_results[category]['trapped'] += 1
        
        print(f"\n{'='*70}")
        print("CATEGORY-BY-CATEGORY RESULTS")
        print("="*70)
        for category, counts in sorted(category_results.items()):
            avoidance = (counts['avoided'] / counts['total']) * 100
            trapped = (counts['trapped'] / counts['total']) * 100
            print(f"{category}: {avoidance:.1f}% avoided, {trapped:.1f}% trapped")
        
        # Assessment
        if avoidance_rate >= 80 and trap_rate <= 20:
            assessment = "EXCEPTIONAL - Highly resistant to adversarial attacks"
        elif avoidance_rate >= 60 and trap_rate <= 40:
            assessment = "EXCELLENT - Good adversarial resistance"
        elif avoidance_rate >= 40:
            assessment = "GOOD - Moderate adversarial resistance"
        elif avoidance_rate >= 20:
            assessment = "MODERATE - Limited adversarial resistance"
        else:
            assessment = "WEAK - Poor adversarial resistance"
        
        print(f"\n{'='*70}")
        print("DIFFICULTY ASSESSMENT")
        print("="*70)
        print(assessment)
        
        return {
            'avoidance_rate': avoidance_rate,
            'trap_rate': trap_rate,
            'avg_time': avg_time,
            'category_results': category_results,
            'results': results
        }


if __name__ == "__main__":
    challenge = AdversarialChallenge()
    results = challenge.run_adversarial_challenge()
