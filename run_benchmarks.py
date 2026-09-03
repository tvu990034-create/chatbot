import requests
import json
import time

# API endpoint
url = 'http://localhost:8000/api/v1/chat'

# Comprehensive benchmarks to run
comprehensive_benchmarks = [
    'BENCHMARK: MMLU',
    'BENCHMARK: GSM8K', 
    'BENCHMARK: HellaSwag',
    'BENCHMARK: HumanEval',
    'BENCHMARK: ARC',
    'BENCHMARK: reasoning',
    'BENCHMARK: math',
    'BENCHMARK: coding',
    'BENCHMARK: general',
    'BENCHMARK: logical',
    'BENCHMARK: analytical',
    'BENCHMARK: creative',
    'BENCHMARK: factual',
    'BENCHMARK: comprehension',
    'BENCHMARK: synthesis',
    'BENCHMARK: evaluation',
    'BENCHMARK: explanation',
    'BENCHMARK: prediction',
    'BENCHMARK: comparison',
    'BENCHMARK: analysis',
    'BENCHMARK: summary',
    'BENCHMARK: interpretation',
    'BENCHMARK: inference',
    'BENCHMARK: application',
    'BENCHMARK: problem_solving',
    'BENCHMARK: decision_making'
]

print('Running Comprehensive Benchmark Suite with 26 Benchmarks')
print('=' * 60)

results = []

for benchmark in comprehensive_benchmarks:
    print(f'\nRunning: {benchmark}')
    
    payload = {
        'messages': [{'role': 'user', 'content': benchmark}],
        'model': 'gemma2:2b',
        'provider': 'ollama',
        'performance_mode': 'balanced'
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=120)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f'OK {benchmark} completed in {duration:.2f}s')
            print(f'   Model: {data["model"]}')
            print(f'   Response length: {len(data["response"])} chars')
            
            results.append({
                'benchmark': benchmark,
                'duration': duration,
                'model': data['model'],
                'response': data['response'][:200] + '...' if len(data['response']) > 200 else data['response']
            })
        else:
            print(f'FAIL {benchmark} failed: {response.status_code}')
    except Exception as e:
        print(f'ERROR {benchmark} error: {str(e)}')
    
    time.sleep(1)  # Brief pause between benchmarks

print('\n' + '=' * 60)
print('COMPREHENSIVE BENCHMARK SUMMARY')
print('=' * 60)

for result in results:
    print(f'{result["benchmark"]}: {result["duration"]:.2f}s')
    
print(f'\nTotal benchmarks completed: {len(results)}')
print(f'Average duration: {sum(r["duration"] for r in results)/len(results):.2f}s')

# Save results
with open('comprehensive_benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)
    
print('\nResults saved to comprehensive_benchmark_results.json')