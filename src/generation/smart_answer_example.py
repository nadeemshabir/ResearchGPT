"""
Example: Using Smart Answer Generation
Demonstrates automatic query routing
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from generation.answer_generator import AnswerGenerator


def main():
    """Demonstrate smart answer generation"""
    
    print("\n" + "="*80)
    print("SMART ANSWER GENERATION DEMO")
    print("="*80)
    
    # Initialize with smart routing enabled
    generator = AnswerGenerator(
        use_multi_agent=True,
        use_citations=True,
        use_smart_routing=True  # ← Enable smart routing
    )
    
    # Test queries of different types
    test_queries = [
        # Q&A queries
        "What is BERT?",
        "How does attention mechanism work?",
        
        # Comparison queries
        "Compare BERT and GPT",
        "What is the difference between RNN and LSTM?",
        
        # Review queries
        "Give me an overview of transformer models",
        "Review recent advances in NLP"
    ]
    
    print("\n" + "="*80)
    print("TESTING SMART ROUTING")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"USER QUERY: {query}")
        print(f"{'─'*80}")
        
        # Smart answer automatically routes to the right method!
        response = generator.smart_answer(query)
        
        # Display result
        print(f"\n📝 ANSWER:")
        print(response['answer'])
        
        print(f"\n📊 METADATA:")
        print(f"   Query Type: {response['metadata'].get('query_type', 'N/A')}")
        print(f"   Routing Confidence: {response['metadata'].get('routing_confidence', 'N/A')}")
        print(f"   Sources: {response['metadata'].get('num_sources', len(response.get('sources', [])))}")
        
        print(f"\n{'─'*80}\n")
        input("Press Enter for next query...")
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
