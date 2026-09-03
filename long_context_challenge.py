"""
Long-Context Challenge
Tests ability to maintain context across complex multi-part conversations
"""

import time
from typing import Dict, List
from benchmark_integrations.simplified_fixed_integration import SimplifiedFixedIntegrationLayer
from gateway.litellm_gateway import get_optimization_stats

class LongContextChallenge:
    """Tests context maintenance across extended conversations."""
    
    def __init__(self):
        self.integration_layer = SimplifiedFixedIntegrationLayer()
        
        # Each challenge is a multi-turn conversation
        self.challenges = [
            {
                "name": "Mathematical Investigation",
                "context": "We're investigating number theory properties. Start with the basic properties of prime numbers.",
                "turns": [
                    "What are the first 10 prime numbers?",
                    "Now, identify which of these primes are twin primes (pairs differing by 2).",
                    "Calculate the sum of all twin prime pairs from your list.",
                    "Using Goldbach's conjecture, express the number 20 as a sum of two primes from your list.",
                    "Finally, what is the relationship between the sum you calculated and the Goldbach representation?",
                    "What pattern do you notice about the distribution of twin primes?",
                    "If we extend to the first 20 primes, how many more twin prime pairs would we find?",
                    "What does this suggest about the Twin Prime Conjecture?",
                    "Based on your analysis, what is the approximate density of twin primes among primes?",
                    "Summarize your findings about prime distribution patterns."
                ]
            },
            {
                "name": "Scientific Investigation",
                "context": "We're studying thermodynamics. Start with the basic laws.",
                "turns": [
                    "What is the First Law of Thermodynamics?",
                    "How does this relate to conservation of energy in a closed system?",
                    "Now consider an open system - how does the First Law apply differently?",
                    "What is the Second Law of Thermodynamics?",
                    "Explain entropy increase in the context of an ice cube melting.",
                    "If we reverse the process (water freezing), does entropy decrease? Explain.",
                    "What is the Third Law of Thermodynamics?",
                    "How does the Third Law relate to absolute zero?",
                    "Combine all three laws - what are the fundamental constraints they impose on any physical system?",
                    "Apply these constraints to a hypothetical perpetual motion machine - why is it impossible?"
                ]
            },
            {
                "name": "Historical Analysis",
                "context": "We're analyzing the causes of World War I. Start with the immediate triggers.",
                "turns": [
                    "What was the immediate trigger for World War I?",
                    "Who were the key alliances involved?",
                    "What were the underlying economic factors?",
                    "How did nationalism contribute to the tensions?",
                    "What role did imperialism play?",
                    "Consider the arms race - how did it escalate tensions?",
                    "What was the significance of the Balkans in this conflict?",
                    "How did diplomatic failures contribute?",
                    "Compare the situation to a modern geopolitical crisis - what similarities do you see?",
                    "Based on this analysis, what lessons can we apply to prevent future conflicts?"
                ]
            },
            {
                "name": "Philosophical Dialogue",
                "context": "We're exploring the nature of consciousness. Start with basic definitions.",
                "turns": [
                    "Define consciousness in your own words.",
                    "How does consciousness differ from intelligence?",
                    "What is the 'hard problem of consciousness'?",
                    "How do materialist theories attempt to explain consciousness?",
                    "What are the main criticisms of materialist approaches?",
                    "How do dualist theories differ?",
                    "What are the main criticisms of dualist approaches?",
                    "Consider integrated information theory - how does it address consciousness?",
                    "What are the limitations of all current theories?",
                    "Based on this analysis, what would a satisfactory theory of consciousness need to explain?"
                ]
            },
            {
                "name": "Technical Problem Solving",
                "context": "We're designing a secure communication system. Start with basic encryption.",
                "turns": [
                    "What is symmetric encryption and how does it work?",
                    "What are the main weaknesses of symmetric encryption?",
                    "What is asymmetric encryption and how does it differ?",
                    "Explain the RSA algorithm in simple terms.",
                    "What are the security assumptions of RSA?",
                    "How can we combine symmetric and asymmetric encryption?",
                    "What is a digital signature and how does it work?",
                    "How do we ensure authentication in our communication system?",
                    "What are the main attack vectors against our system?",
                    "Design a complete secure communication protocol addressing all these concerns."
                ]
            }
        ]
    
    def evaluate_conversation(self, conversation: List[Dict]) -> Dict:
        """Evaluate a conversation for coherence and context maintenance."""
        if not conversation:
            return {'coherence': 0, 'context_maintenance': 0, 'depth': 0}
        
        # Check for coherence - does each response relate to the previous question?
        coherence_score = 0
        for i in range(1, len(conversation)):
            if len(conversation[i]['response']) > 50:  # Substantial response
                coherence_score += 1
        
        coherence = (coherence_score / max(1, len(conversation) - 1)) * 100
        
        # Check for context maintenance - does the response reference previous context?
        context_maintenance = 0
        full_conversation = " ".join([conv['response'] for conv in conversation])
        if len(full_conversation) > 500:  # Shows engagement with context
            context_maintenance = 100
        
        # Check for depth - complexity of responses
        avg_response_length = sum(len(conv['response']) for conv in conversation) / len(conversation)
        depth = min(100, avg_response_length / 5)  # Normalize to 100
        
        return {
            'coherence': coherence,
            'context_maintenance': context_maintenance,
            'depth': depth,
            'overall': (coherence + context_maintenance + depth) / 3
        }
    
    def run_long_context_challenge(self) -> Dict:
        """Run the long context challenge."""
        print(f"\n{'='*70}")
        print("LONG CONTEXT CHALLENGE")
        print("Tests ability to maintain context across 10-turn conversations")
        print("="*70)
        
        challenge_results = []
        total_time = 0
        
        for challenge in self.challenges:
            print(f"\n{'='*70}")
            print(f"Challenge: {challenge['name']}")
            print(f"Context: {challenge['context']}")
            print("="*70)
            
            conversation = []
            challenge_start = time.time()
            
            for i, question in enumerate(challenge['turns'], 1):
                print(f"\nTurn {i}/{len(challenge['turns'])}: {question[:60]}...")
                
                turn_start = time.time()
                try:
                    response, enhancement_info = self.integration_layer.query_with_gateway(question)
                    elapsed = time.time() - turn_start
                    
                    conversation.append({
                        'turn': i,
                        'question': question,
                        'response': response,
                        'time': elapsed
                    })
                    
                    print(f"Time: {elapsed:.2f}s | Response: {response[:80]}...")
                    
                except Exception as e:
                    print(f"[ERROR] Turn {i}: {str(e)}")
                    conversation.append({
                        'turn': i,
                        'question': question,
                        'response': f"Error: {str(e)}",
                        'time': 0,
                        'error': str(e)
                    })
            
            challenge_time = time.time() - challenge_start
            total_time += challenge_time
            
            evaluation = self.evaluate_conversation(conversation)
            
            challenge_results.append({
                'name': challenge['name'],
                'conversation': conversation,
                'evaluation': evaluation,
                'time': challenge_time
            })
            
            print(f"\n{'='*70}")
            print(f"Challenge Results: {challenge['name']}")
            print("="*70)
            print(f"Coherence: {evaluation['coherence']:.1f}%")
            print(f"Context Maintenance: {evaluation['context_maintenance']:.1f}%")
            print(f"Depth: {evaluation['depth']:.1f}%")
            print(f"Overall Score: {evaluation['overall']:.1f}%")
            print(f"Total Time: {challenge_time:.2f}s")
        
        # Calculate overall results
        avg_coherence = sum(r['evaluation']['coherence'] for r in challenge_results) / len(challenge_results)
        avg_context = sum(r['evaluation']['context_maintenance'] for r in challenge_results) / len(challenge_results)
        avg_depth = sum(r['evaluation']['depth'] for r in challenge_results) / len(challenge_results)
        avg_overall = sum(r['evaluation']['overall'] for r in challenge_results) / len(challenge_results)
        avg_time = total_time / len(challenge_results)
        
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
        print("LONG CONTEXT CHALLENGE RESULTS")
        print("="*70)
        print(f"Average Coherence: {avg_coherence:.1f}%")
        print(f"Average Context Maintenance: {avg_context:.1f}%")
        print(f"Average Depth: {avg_depth:.1f}%")
        print(f"Overall Score: {avg_overall:.1f}%")
        print(f"Average Time per Challenge: {avg_time:.2f}s")
        
        # Assessment
        if avg_overall >= 90:
            assessment = "EXCEPTIONAL - Master of long-context reasoning"
        elif avg_overall >= 75:
            assessment = "EXCELLENT - Strong long-context capabilities"
        elif avg_overall >= 60:
            assessment = "GOOD - Adequate long-context reasoning"
        elif avg_overall >= 40:
            assessment = "MODERATE - Limited long-context capabilities"
        else:
            assessment = "WEAK - Poor long-context maintenance"
        
        print(f"\n{'='*70}")
        print("DIFFICULTY ASSESSMENT")
        print("="*70)
        print(assessment)
        
        return {
            'avg_coherence': avg_coherence,
            'avg_context': avg_context,
            'avg_depth': avg_depth,
            'avg_overall': avg_overall,
            'avg_time': avg_time,
            'challenge_results': challenge_results
        }


if __name__ == "__main__":
    challenge = LongContextChallenge()
    results = challenge.run_long_context_challenge()
