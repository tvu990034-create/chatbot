"""
Direct gateway test
"""

from gateway.litellm_gateway import chat

print("Testing direct gateway call...")
response = chat(messages=[{"role": "user", "content": "What is 5 + 3?"}], use_cache=False)
print(f"Response: {response}")
print("Gateway is working!")
