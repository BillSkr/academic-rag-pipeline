#!/usr/bin/env python3
"""
Test suite for the RAG Pipeline
Run: python tests/test_api.py
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8001"

# Test queries: Academic (should work) vs Non-academic (should reject)
ACADEMIC_QUERIES = [
    "What is SOD1 protein and its role in ALS?",
    "Explain amyotrophic lateral sclerosis",
    "What are the mechanisms of neurodegeneration?",
]

NON_ACADEMIC_QUERIES = [
    "What's your favorite food?",
    "Tell me a joke",
    "How do I bake cookies?",
]

def test_health():
    """Test 1: Health Check"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        assert response.status_code == 200
        print("✓ PASSED")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_academic_query():
    """Test 2: Academic Query (Should Return Results)"""
    print("\n" + "="*60)
    print("TEST 2: Academic Query")
    print("="*60)
    
    query = ACADEMIC_QUERIES[0]
    print(f"Query: {query}")
    print("Waiting for response (this takes 90-120 seconds)...\n")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": query},
            timeout=180,
            stream=True
        )
        
        final_response = None
        for line in response.iter_lines():
            if line and b'completed' in line:
                final_response = json.loads(line.decode().replace('data: ', ''))
                break
        
        elapsed = time.time() - start
        
        if final_response:
            resp_text = final_response.get('response', '')[:100]
            citations = len(final_response.get('citations', []))
            
            print(f"Time: {elapsed:.1f}s")
            print(f"Response: {resp_text}...")
            print(f"Citations: {citations}")
            
            # Check if answer is meaningful (not the fallback message)
            if "wasn't able to find enough" not in resp_text:
                print("✓ PASSED - Retrieved relevant information")
                return True
            else:
                print("✗ FAILED - Only fallback response (no matches)")
                return False
        else:
            print("✗ FAILED - No response")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_non_academic_query():
    """Test 3: Non-Academic Query (Should Be Rejected)"""
    print("\n" + "="*60)
    print("TEST 3: Non-Academic Query (Should Be Rejected)")
    print("="*60)
    
    query = NON_ACADEMIC_QUERIES[0]
    print(f"Query: {query}")
    print("Sending request...\n")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": query},
            timeout=180,
            stream=True
        )
        
        final_response = None
        for line in response.iter_lines():
            if line and b'completed' in line:
                final_response = json.loads(line.decode().replace('data: ', ''))
                break
        
        elapsed = time.time() - start
        
        if final_response:
            resp_text = final_response.get('response', '')
            
            print(f"Time: {elapsed:.1f}s")
            print(f"Response: {resp_text[:100]}...")
            
            # Should get the "not related to academic research" message
            if "does not appear to be related to academic research" in resp_text:
                print("✓ PASSED - Correctly rejected non-academic query")
                return True
            else:
                print("✗ FAILED - Unexpectedly accepted non-academic query")
                return False
        else:
            print("✗ FAILED - No response")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def test_response_format():
    """Test 4: Response Format Validation"""
    print("\n" + "="*60)
    print("TEST 4: Response Format Validation")
    print("="*60)
    
    query = "What is CRISPR gene editing?"
    print(f"Query: {query}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": query},
            timeout=180,
            stream=True
        )
        
        # Collect all SSE events
        events = []
        for line in response.iter_lines():
            if line:
                try:
                    event_data = json.loads(line.decode().replace('data: ', ''))
                    events.append(event_data)
                except:
                    pass
        
        print(f"Total events received: {len(events)}")
        
        # Check event sequence
        checks = [
            ("Initial status", any("Analyzing" in str(e.get('status', '')) for e in events)),
            ("Final status", any(e.get('status') == 'completed' for e in events)),
            ("Has response", any(e.get('response') for e in events)),
            ("SSE format", all('status' in e for e in events)),
        ]
        
        passed = 0
        for check_name, check_result in checks:
            status = "✓" if check_result else "✗"
            print(f"  {status} {check_name}")
            if check_result:
                passed += 1
        
        if passed == len(checks):
            print("✓ PASSED - Response format valid")
            return True
        else:
            print(f"✗ FAILED - {len(checks) - passed} checks failed")
            return False
            
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("RAG PIPELINE TEST SUITE")
    print(f"Target: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {
        "Health Check": test_health(),
        "Response Format": test_response_format(),
        # Note: Academic and non-academic tests take >3 minutes each
        # Uncomment to run them
        # "Academic Query": test_academic_query(),
        # "Non-Academic Query": test_non_academic_query(),
    }
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    total_passed = sum(1 for p in results.values() if p)
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")

if __name__ == "__main__":
    main()
