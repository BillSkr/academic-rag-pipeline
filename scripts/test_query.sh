#!/bin/bash
# Quick test script for RAG Pipeline
# Usage: bash scripts/test_query.sh "Your question here"

QUESTION="${1:-What is SOD1 protein?}"
API_URL="http://localhost:8001"

echo "================================"
echo "RAG Pipeline Query Test"
echo "================================"
echo "Question: $QUESTION"
echo "API: $API_URL"
echo ""
echo "Sending request (may take 90-120 seconds)..."
echo ""

python3 << 'EOF'
import requests
import json
import sys
import time

question = sys.argv[1] if len(sys.argv) > 1 else "What is SOD1 protein?"
api_url = "http://localhost:8001"

try:
    start = time.time()
    response = requests.post(
        f"{api_url}/query",
        json={"question": question},
        timeout=180,
        stream=True
    )
    
    events_received = 0
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line.decode().replace('data: ', ''))
                events_received += 1
                
                if data.get('status'):
                    if data['status'] == 'completed':
                        elapsed = time.time() - start
                        response_text = data.get('response', '')
                        citations = len(data.get('citations', []))
                        
                        print(f"\n✓ Response received in {elapsed:.1f}s\n")
                        print("=" * 50)
                        print("ANSWER:")
                        print("=" * 50)
                        print(response_text)
                        print("\n" + "=" * 50)
                        print(f"SOURCES: {citations} citations")
                        print("=" * 50)
                        
                        if citations > 0:
                            print("\nFirst citation:")
                            first = data['citations'][0]
                            print(f"  - {first['metadata'].get('title', 'N/A')}")
                            print(f"  - Year: {first['metadata'].get('year', 'N/A')}")
                        break
            except json.JSONDecodeError:
                pass
    
    print(f"\nEvents processed: {events_received}")
    
except requests.exceptions.Timeout:
    print("✗ Request timeout. The service may be overloaded.")
except requests.exceptions.ConnectionError:
    print("✗ Cannot connect to API at http://localhost:8001")
    print("Make sure docker compose is running: docker compose up -d")
except Exception as e:
    print(f"✗ Error: {e}")

EOF "$QUESTION"
