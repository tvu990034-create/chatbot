"""
Simple gateway test
"""

from gateway.litellm_gateway import chat

print("Testing gateway connection...")
try:
    response = chat(messages=[{"role": "user", "content": "What is 2+2?"}], use_cache=False)
    print(f"Response: {response}")
    print("Gateway is working!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
