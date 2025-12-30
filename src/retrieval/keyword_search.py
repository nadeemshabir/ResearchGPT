"""
Keyword Search Module
Traditional search using BM25 algorithm
"""

import os
import sys
from typing import List, Dict
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ingestion.database import VectorDatabase

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("⚠️ Installing rank-bm25...")
    os.system("pip install rank-bm25")
    from rank_bm25 import BM25Okapi


class KeywordSearcher:
    """Keyword-based search using BM25 algorithm"""
    
    def __init__(self, database: VectorDatabase = None):
        """
        Initialize keyword searcher
        
        Args:
            database: Vector database instance
        """
        print("🔤 Initializing Keyword Search (BM25)...")
        
        # Initialize database
        if database is None:
            self.database = VectorDatabase()
        else:
            self.database = database
        
        # Load all documents
        self.corpus = []
        self.corpus_metadata = []
        self._load_corpus()
        
        # Initialize BM25
        if self.corpus:
            self._initialize_bm25()
        else:
            self.bm25 = None
            print("⚠️ No documents in database for BM25 index")
    
    def _load_corpus(self):
        """Load all documents from database"""
        print("   Loading corpus from database...")
        
        # Get all documents
        all_docs = self.database.collection.get()
        
        if not all_docs['documents']:
            print("   ⚠️ No documents found in database")
            return
        
        self.corpus = all_docs['documents']
        self.corpus_metadata = [
            {
                'id': all_docs['ids'][i],
                'metadata': all_docs['metadatas'][i]
            }
            for i in range(len(all_docs['documents']))
        ]
        
        print(f"   ✅ Loaded {len(self.corpus)} documents")
    
    def _initialize_bm25(self):
        """Initialize BM25 index"""
        print("   Building BM25 index...")
        
        # Tokenize corpus (simple whitespace + lowercase)
        tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"   ✅ BM25 index ready")
    
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search using BM25 keyword matching
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not self.bm25:
            print("⚠️ BM25 index not initialized")
            return []
        
        if not query or not query.strip():
            print("⚠️ Empty query")
            return []
        
        print(f"\n🔤 BM25 Search: '{query[:100]}...'")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Format results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = scores[idx]
            
            # Skip if score is too low
            if score < 0.01:
                continue
            
            result = {
                'rank': rank,
                'text': self.corpus[idx],
                'bm25_score': round(float(score), 4),
                'id': self.corpus_metadata[idx]['id'],
                'metadata': self.corpus_metadata[idx]['metadata']
            }
            
            results.append(result)
        
        print(f"✅ Found {len(results)} results (filtered low scores)")
        
        return results
    
    def search_with_phrases(
        self,
        query: str,
        top_k: int = 10,
        boost_phrases: bool = True
    ) -> List[Dict]:
        """
        Search with phrase matching boost
        
        Args:
            query: Search query
            top_k: Number of results
            boost_phrases: Whether to boost exact phrase matches
            
        Returns:
            Search results with boosted scores
        """
        # Get base BM25 results
        results = self.search(query, top_k=top_k * 2)  # Get more initially
        
        if boost_phrases and '"' in query:
            # Extract phrases in quotes
            import re
            phrases = re.findall(r'"([^"]*)"', query)
            
            print(f"   Boosting phrases: {phrases}")
            
            # Boost scores for exact phrase matches
            for result in results:
                text_lower = result['text'].lower()
                boost = 1.0
                
                for phrase in phrases:
                    if phrase.lower() in text_lower:
                        boost += 0.5  # 50% boost per phrase match
                
                result['bm25_score'] = result['bm25_score'] * boost
                result['phrase_boost'] = boost
            
            # Re-sort by boosted scores
            results.sort(key=lambda x: x['bm25_score'], reverse=True)
        
        # Re-rank and trim to top_k
        for i, result in enumerate(results[:top_k], 1):
            result['rank'] = i
        
        return results[:top_k]
    
    def get_term_frequencies(self, query: str) -> Dict:
        """
        Get term frequencies for query terms across corpus
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with term frequencies
        """
        if not self.bm25:
            return {}
        
        tokenized_query = query.lower().split()
        
        # Get document frequencies
        term_freqs = {}
        for term in tokenized_query:
            # Count documents containing term
            count = sum(1 for doc in self.corpus if term in doc.lower())
            term_freqs[term] = {
                'doc_frequency': count,
                'doc_percentage': round(count / len(self.corpus) * 100, 2)
            }
        
        return term_freqs
    
    def visualize_results(self, results: List[Dict], max_text_length: int = 200):
        """
        Print formatted search results
        
        Args:
            results: Search results
            max_text_length: Max characters to display
        """
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"BM25 SEARCH RESULTS ({len(results)} found)")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"Rank {result['rank']}")
            print(f"─" * 80)
            print(f"BM25 Score: {result['bm25_score']:.4f}", end='')
            
            if 'phrase_boost' in result:
                print(f" (Phrase boost: {result['phrase_boost']:.2f}x)", end='')
            print()
            
            # Metadata
            metadata = result['metadata']
            print(f"Paper: {metadata.get('title', 'Unknown')[:60]}")
            print(f"Section: {metadata.get('section_title', 'Unknown')}")
            
            # Text
            text = result['text']
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"\nText: {text}")
            
            print(f"\n{'='*80}\n")


def test_keyword_search():
    """Test keyword search"""
    
    # Initialize
    searcher = KeywordSearcher()
    
    if not searcher.corpus:
        print("\n⚠️ No documents in database!")
        print("   Run ingestion pipeline first")
        return
    
    # Test queries
    test_queries = [
        "attention mechanism",
        "transformer architecture",
        "BERT model",
        '"self-attention"', 
         "what is Os" # Phrase query
    ]
    
    print("\n" + "="*80)
    print("TESTING KEYWORD SEARCH (BM25)")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        # Get term frequencies
        term_freqs = searcher.get_term_frequencies(query.replace('"', ''))
        print(f"\nTerm Frequencies:")
        for term, freq in term_freqs.items():
            print(f"  '{term}': {freq['doc_frequency']} docs ({freq['doc_percentage']}%)")
        
        # Search
        if '"' in query:
            results = searcher.search_with_phrases(query, top_k=3)
        else:
            results = searcher.search(query, top_k=3)
        
        # Visualize
        searcher.visualize_results(results, max_text_length=150)


if __name__ == "__main__":
    test_keyword_search()