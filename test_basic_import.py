#!/usr/bin/env python3
"""Test basic functionality without API key."""

from psy_bench import PsyBench

def test_basic_functionality():
    """Test basic functionality."""
    # This should work without API key
    try:
        bench = PsyBench()
        print("✅ PsyBench initialized")
    except Exception as e:
        print(f"❌ PsyBench initialization failed: {e}")
        return
    
    # This should work - listing cases
    try:
        cases = bench.list_cases()
        print(f"✅ Found {len(cases)} test cases")
        print(f"   First case: {cases[0]}")
    except Exception as e:
        print(f"❌ Failed to list cases: {e}")
    
    # This should work - getting themes
    try:
        themes = bench.get_themes()
        print(f"✅ Found {len(themes)} themes: {', '.join(themes[:3])}...")
    except Exception as e:
        print(f"❌ Failed to get themes: {e}")
    
    # This should fail gracefully - running experiment without API key
    try:
        result = bench.run("Case 1.1: The Conduit [EXPLICIT]")
        print(f"❌ Unexpectedly succeeded: {result}")
    except Exception as e:
        print(f"✅ Expected failure when running without API key: {type(e).__name__}")

if __name__ == "__main__":
    test_basic_functionality()