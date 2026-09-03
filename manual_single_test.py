"""
Manual single query test to check if the system is working at all
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gateway.litellm_gateway import chat

MODEL = "ollama/phi3:mini"

print("Testing single query to phi3:mini...")
print("Query: What is 2+2?")

try:
    import time
    start = time.time()
    response = chat(
        messages=[{"role": "user", "content": "What is 2+2?"}],
        model=MODEL,
        use_cache=False
    )
    elapsed = time.time() - start
    
    print(f"\nResponse received in {elapsed:.2f}s:")
    print(response)
    
    # Check if answer is correct
    if "4" in response:
        print("\n✅ Correct answer detected!")
    else:
        print("\n❌ Answer may be incorrect or unclear")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()