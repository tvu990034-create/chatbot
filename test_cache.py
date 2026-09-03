import requests
import time

url = 'http://localhost:8000/api/v1/chat'

# Test 1: First query (should be cache miss)
print('Test 1: First query (cache miss expected)')
payload = {
    'messages': [{'role': 'user', 'content': 'hello bro'}],
    'model': 'gemma2:2b',
    'provider': 'ollama',
    'performance_mode': 'balanced'
}
start = time.time()
response = requests.post(url, json=payload)
duration1 = time.time() - start
print(f'Duration: {duration1:.2f}s')
print(f'Response length: {len(response.json()["response"])} chars')

time.sleep(1)

# Test 2: Same query (should be cache hit)
print('\nTest 2: Same query (cache hit expected)')
start = time.time()
response = requests.post(url, json=payload)
duration2 = time.time() - start
print(f'Duration: {duration2:.2f}s')
print(f'Response length: {len(response.json()["response"])} chars')

# Test 3: Intelligence test
print('\nTest 3: Intelligence test')
payload = {
    'messages': [{'role': 'user', 'content': 'explain why the sky is blue in simple terms'}],
    'model': 'gemma2:2b',
    'provider': 'ollama',
    'performance_mode': 'balanced'
}
start = time.time()
response = requests.post(url, json=payload)
duration3 = time.time() - start
print(f'Duration: {duration3:.2f}s')
print(f'Response length: {len(response.json()["response"])} chars')

print(f'\nCache improvement: {duration1/duration2:.1f}x faster on cache hit')