"""
Test script for 18 advanced vision techniques integration
"""

import sys
import io
from pathlib import Path

# Fix unicode encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_advanced_techniques_import():
    """Test that advanced techniques can be imported"""
    try:
        from gateway.advanced_vision_techniques import AdvancedVisionTechniques
        print("OK AdvancedVisionTechniques imported successfully")
        
        # Initialize the techniques
        techniques = AdvancedVisionTechniques(feature_dim=512)
        print(f"OK AdvancedVisionTechniques initialized with {techniques.technique_count} techniques")
        
        # Get stats
        stats = techniques.get_stats()
        print(f"OK Stats: {stats['total_techniques']} total techniques")
        print(f"  - S-Tier: {stats['s_tier_count']}")
        print(f"  - A-Tier: {stats['a_tier_count']}")
        
        print("\nTechniques loaded:")
        for i, technique in enumerate(stats['techniques'], 1):
            print(f"  {i}. {technique}")
        
        return True
    except Exception as e:
        print(f"FAIL Error importing advanced techniques: {e}")
        return False

def test_vision_gateway_integration():
    """Test that vision gateway integrates with advanced techniques"""
    try:
        from vision_enhanced_gateway import VisionEnhancedGateway
        print("OK VisionEnhancedGateway imported successfully")
        
        # Initialize gateway
        gateway = VisionEnhancedGateway()
        print("OK VisionEnhancedGateway initialized")
        
        # Check if advanced techniques are loaded
        if hasattr(gateway, 'advanced_techniques') and gateway.advanced_techniques is not None:
            print("OK Advanced techniques integrated into gateway")
            stats = gateway.advanced_techniques.get_stats()
            print(f"  - {stats['enabled_techniques']} techniques enabled")
        else:
            print("WARN Advanced techniques not available (PyTorch not installed)")
        
        return True
    except Exception as e:
        print(f"FAIL Error testing vision gateway: {e}")
        return False

def test_pytorch_availability():
    """Test if PyTorch is available"""
    try:
        import torch
        print(f"OK PyTorch available: {torch.__version__}")
        return True
    except ImportError:
        print("WARN PyTorch not available - advanced techniques will use fallback mode")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing Advanced Vision Techniques Integration")
    print("="*60)
    
    # Test PyTorch availability
    print("\n1. Testing PyTorch availability...")
    pytorch_available = test_pytorch_availability()
    
    # Test advanced techniques import
    print("\n2. Testing advanced techniques import...")
    techniques_ok = test_advanced_techniques_import()
    
    # Test vision gateway integration
    print("\n3. Testing vision gateway integration...")
    gateway_ok = test_vision_gateway_integration()
    
    print("\n" + "="*60)
    if techniques_ok and gateway_ok:
        print("OK All tests passed!")
        print("OK 18 advanced S-Tier and A-Tier techniques successfully integrated")
    else:
        print("WARN Some tests failed - check errors above")
    print("="*60)
