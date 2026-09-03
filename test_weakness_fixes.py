"""
Test weakness fixes for enhanced gateway
Test formal math, agent tasks, and academic knowledge enhancements
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway.fixed_enhanced_gateway import get_fixed_enhanced_gateway

def test_formal_math_enhancement():
    """Test formal math proof enhancement (MiniF2F weakness)."""
    gateway = get_fixed_enhanced_gateway()
    
    formal_query = "Prove that for all natural numbers n, 1 + 2 + ... + n = n(n+1)/2"
    
    enhanced = gateway._enhance_formal_math_prompt(formal_query)
    
    print("Formal Math Enhancement Test:")
    print(f"Original: {formal_query}")
    print(f"Enhanced: {enhanced}")
    print(f"Enhancement active: {gateway.formal_math_enhancement}")
    print(f"Detection works: {gateway._is_formal_math_task(formal_query)}")
    print()

def test_agent_task_enhancement():
    """Test agent coordination enhancement (OSWorld/GAIA/AgentBench weakness)."""
    gateway = get_fixed_enhanced_gateway()
    
    agent_query = "Coordinate multiple agents to complete a file management task in the operating system"
    
    enhanced = gateway._enhance_agent_coordination_prompt(agent_query)
    
    print("Agent Task Enhancement Test:")
    print(f"Original: {agent_query}")
    print(f"Enhanced: {enhanced}")
    print(f"Enhancement active: {gateway.agent_task_enhancement}")
    print(f"Detection works: {gateway._is_agent_task(agent_query)}")
    print()

def test_academic_knowledge_enhancement():
    """Test academic knowledge enhancement (MMLU/GPQA weakness)."""
    gateway = get_fixed_enhanced_gateway()
    
    academic_query = "Which of the following is the correct answer: A) 42 B) 24 C) 18 D) 36"
    
    enhanced = gateway._enhance_academic_knowledge_prompt(academic_query)
    
    print("Academic Knowledge Enhancement Test:")
    print(f"Original: {academic_query}")
    print(f"Enhanced: {enhanced}")
    print(f"Enhancement active: {gateway.academic_knowledge_enhancement}")
    print(f"Detection works: {gateway._is_academic_task(academic_query)}")
    print()

def test_integration():
    """Test that enhancements work in actual chat."""
    gateway = get_fixed_enhanced_gateway()
    
    print("Integration Test - Actual Chat with Enhancements:")
    
    # Test formal math
    print("\n1. Testing formal math query...")
    messages = [{"role": "user", "content": "Prove that sqrt(2) is irrational"}]
    try:
        response = gateway.chat(messages=messages, use_cache=False)
        print(f"Response length: {len(response)}")
        print(f"Contains proof terms: {'proof' in response.lower() or 'theorem' in response.lower()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test agent task
    print("\n2. Testing agent task query...")
    messages = [{"role": "user", "content": "Coordinate two agents to search and replace text in files"}]
    try:
        response = gateway.chat(messages=messages, use_cache=False)
        print(f"Response length: {len(response)}")
        print(f"Contains coordination terms: {'step' in response.lower() or 'coordinate' in response.lower()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test academic knowledge
    print("\n3. Testing academic knowledge query...")
    messages = [{"role": "user", "content": "Which of the following is the correct answer: A) 42 B) 24 C) 18 D) 36"}]
    try:
        response = gateway.chat(messages=messages, use_cache=False)
        print(f"Response length: {len(response)}")
        print(f"Contains reasoning: {'because' in response.lower() or 'answer' in response.lower()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("WEAKNESS FIXES TEST")
    print("="*60)
    print()
    
    test_formal_math_enhancement()
    test_agent_task_enhancement()
    test_academic_knowledge_enhancement()
    test_integration()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)