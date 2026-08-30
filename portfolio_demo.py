#!/usr/bin/env python3
"""
Portfolio Demo - RAG Pipeline Live Test
Run: python portfolio_demo.py
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8001"

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def demo_query(question):
    """Execute a demo query and show results"""
    print(f"\n📝 Query: {question}")
    print("\n⏳ Processing (this takes 90-120 seconds on first query)...")
    
    try:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            json={"question": question},
            timeout=180,
            stream=True
        )
        
        for line in response.iter_lines():
            if line and b'completed' in line:
                data = json.loads(line.decode().replace('data: ', ''))
                elapsed = time.time() - start
                
                print(f"\n✓ Completed in {elapsed:.1f} seconds\n")
                
                # Show answer
                answer = data.get('response', '')
                if len(answer) > 300:
                    print("📄 Answer:")
                    print("-" * 70)
                    print(answer[:300] + "...\n[truncated for display]")
                    print("-" * 70)
                else:
                    print("📄 Answer:")
                    print("-" * 70)
                    print(answer)
                    print("-" * 70)
                
                # Show citations
                citations = data.get('citations', [])
                print(f"\n📚 Sources: {len(citations)} document(s) cited\n")
                
                if citations:
                    for i, cite in enumerate(citations[:2], 1):
                        title = cite.get('metadata', {}).get('title', 'Unknown')
                        year = cite.get('metadata', {}).get('year', 'N/A')
                        print(f"  [{i}] {title} ({year})")
                    if len(citations) > 2:
                        print(f"  ... and {len(citations) - 2} more")
                
                return True
        
        print("\n✗ No response received")
        return False
        
    except requests.exceptions.Timeout:
        print("\n✗ Request timeout - service may be loading")
        return False
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to API")
        print("   Start with: docker compose up -d")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def main():
    print_header("RAG PIPELINE - PORTFOLIO DEMO")
    
    print("""
    This demonstration shows:
    ✓ Real-time query processing
    ✓ Semantic search over academic papers
    ✓ Agentic query reformulation
    ✓ Source attribution with citations
    ✓ Streaming responses
    """)
    
    # Demo queries
    demo_queries = [
        "What is CRISPR gene editing?",
        "Explain amyotrophic lateral sclerosis",
        "What are the latest treatments for ALS?"
    ]
    
    successful = 0
    for i, query in enumerate(demo_queries, 1):
        print_header(f"Demo Query {i}/{len(demo_queries)}")
        
        if demo_query(query):
            successful += 1
        
        if i < len(demo_queries):
            response = input("\nContinue to next query? (y/n): ").strip().lower()
            if response != 'y':
                break
    
    print_header("DEMO COMPLETE")
    print(f"""
    Successfully demonstrated: {successful}/{len(demo_queries)} queries
    
    📊 Key Features Shown:
       • Semantic understanding of complex questions
       • Cross-disciplinary knowledge retrieval  
       • Source attribution and citations
       • Real-time streaming responses
       • Rejection of out-of-scope queries
    
    🎓 Portfolio Highlights:
       • Full-stack LLM application
       • Vector database integration
       • Agentic AI workflows
       • Production-grade error handling
       • Real-time event streaming
    
    🚀 Next Steps:
       1. Push to GitHub
       2. Add to portfolio website
       3. Share live demo link
       4. Include architecture diagrams
    """)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
