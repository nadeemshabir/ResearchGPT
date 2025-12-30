"""
Hybrid Search Module
Combines semantic and keyword search for best results
"""

from typing import List, Dict, Optional
from pathlib import Path

from .semantic_search import SemanticSearcher
from .keyword_search import KeywordSearcher


class HybridSearcher:
    """Combine semantic and keyword search"""
    
    def __init__(
        self,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid searcher
        
        Args:
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        print("🔀 Initializing Hybrid Search...")
        print(f"   Semantic weight: {semantic_weight}")
        print(f"   Keyword weight: {keyword_weight}")
        
        # Validate weights
        if not (0 <= semantic_weight <= 1) or not (0 <= keyword_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        # Normalize weights to sum to 1
        total = semantic_weight + keyword_weight
        self.semantic_weight = semantic_weight / total
        self.keyword_weight = keyword_weight / total
        
        print(f"   Normalized - Semantic: {self.semantic_weight:.2f}, Keyword: {self.keyword_weight:.2f}")
        
        # Initialize searchers
        self.semantic_searcher = SemanticSearcher()
        self.keyword_searcher = KeywordSearcher(
            database=self.semantic_searcher.database
        )
        
        print("✅ Hybrid search ready")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        retrieval_k: int = None,
        return_component_scores: bool = False
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword approaches
        
        Args:
            query: Search query
            top_k: Number of final results to return
            retrieval_k: Number of results to get from each method (default: top_k * 2)
            return_component_scores: Include individual semantic/keyword scores
            
        Returns:
            Combined and re-ranked results
        """
        if not query or not query.strip():
            print("⚠️ Empty query")
            return []
        
        # Default: retrieve more candidates than needed
        if retrieval_k is None:
            retrieval_k = top_k * 2
        
        print(f"\n🔀 Hybrid Search: '{query[:100]}...'")
        print(f"   Retrieving {retrieval_k} candidates from each method")
        
        # Step 1: Get semantic results
        print("   1️⃣ Running semantic search...")
        semantic_results = self.semantic_searcher.search(query, top_k=retrieval_k)
        
        # Step 2: Get keyword results
        print("   2️⃣ Running keyword search...")
        keyword_results = self.keyword_searcher.search(query, top_k=retrieval_k)
        
        # Step 3: Normalize and combine scores
        print("   3️⃣ Combining scores...")
        combined_results = self._combine_results(
            semantic_results,
            keyword_results,
            return_component_scores
        )
        
        # Step 4: Sort by hybrid score
        combined_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        # Step 5: Re-rank and trim
        final_results = combined_results[:top_k]
        for i, result in enumerate(final_results, 1):
            result['rank'] = i
        
        print(f"✅ Returning top {len(final_results)} results")
        
        return final_results
    
    def _combine_results(
        self,
        semantic_results: List[Dict],
        keyword_results: List[Dict],
        return_components: bool
    ) -> List[Dict]:
        """
        Combine and normalize scores from both methods
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            return_components: Include individual scores
            
        Returns:
            Combined results with hybrid scores
        """
        # Normalize scores to 0-1 range
        semantic_scores = self._normalize_scores(
            semantic_results,
            score_key='similarity_score'
        )
        keyword_scores = self._normalize_scores(
            keyword_results,
            score_key='bm25_score'
        )
        
        # Combine results by chunk ID
        combined = {}
        
        # Add semantic results
        for result, norm_score in zip(semantic_results, semantic_scores):
            chunk_id = result['id']
            combined[chunk_id] = {
                'result': result,
                'semantic_score': norm_score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result, norm_score in zip(keyword_results, keyword_scores):
            chunk_id = result['id']
            if chunk_id in combined:
                combined[chunk_id]['keyword_score'] = norm_score
            else:
                combined[chunk_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': norm_score
                }
        
        # Calculate hybrid scores
        final_results = []
        for chunk_id, data in combined.items():
            result = data['result']
            
            # Calculate weighted hybrid score
            hybrid_score = (
                data['semantic_score'] * self.semantic_weight +
                data['keyword_score'] * self.keyword_weight
            )
            
            result['hybrid_score'] = round(hybrid_score, 4)
            
            if return_components:
                result['semantic_score_normalized'] = round(data['semantic_score'], 4)
                result['keyword_score_normalized'] = round(data['keyword_score'], 4)
            
            final_results.append(result)
        
        return final_results
    
    def _normalize_scores(
        self,
        results: List[Dict],
        score_key: str
    ) -> List[float]:
        """
        Normalize scores to 0-1 range using min-max scaling
        
        Args:
            results: List of results with scores
            score_key: Key containing the score
            
        Returns:
            List of normalized scores
        """
        if not results:
            return []
        
        # Extract scores
        scores = [r.get(score_key, 0) for r in results]
        
        # Handle edge cases
        if not scores or all(s == 0 for s in scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            return [1.0] * len(scores)
        
        # Min-max normalization
        normalized = [
            (score - min_score) / (max_score - min_score)
            for score in scores
        ]
        
        return normalized
    
    def adaptive_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Adaptive search that adjusts weights based on query characteristics
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results with adaptive weighting
        """
        # Analyze query
        query_features = self._analyze_query(query)
        
        # Adjust weights based on query type
        if query_features['has_technical_terms']:
            # Technical queries → favor keyword search
            semantic_w = 0.4
            keyword_w = 0.6
        elif query_features['is_conceptual']:
            # Conceptual queries → favor semantic search
            semantic_w = 0.8
            keyword_w = 0.2
        else:
            # Balanced
            semantic_w = 0.6
            keyword_w = 0.4
        
        print(f"   🎯 Adaptive weights: Semantic={semantic_w:.2f}, Keyword={keyword_w:.2f}")
        
        # Temporarily adjust weights
        original_sem = self.semantic_weight
        original_key = self.keyword_weight
        
        self.semantic_weight = semantic_w / (semantic_w + keyword_w)
        self.keyword_weight = keyword_w / (semantic_w + keyword_w)
        
        # Search
        results = self.search(query, top_k=top_k)
        
        # Restore original weights
        self.semantic_weight = original_sem
        self.keyword_weight = original_key
        
        return results
    
    def _analyze_query(self, query: str) -> Dict:
        """
        Analyze query characteristics
        
        Args:
            query: Search query
            
        Returns:
            Dictionary of query features
        """
        query_lower = query.lower()
        
        # Technical terms (common in papers)
        technical_keywords = [
            'algorithm', 'model', 'equation', 'function', 'method',
            'architecture', 'training', 'optimization', 'loss',
            'accuracy', 'precision', 'recall', 'parameter'
        ]
        
        # Conceptual question words
        conceptual_words = [
            'what', 'why', 'how', 'explain', 'describe', 'compare',
            'difference', 'relationship', 'concept', 'idea'
        ]
        
        has_technical = any(term in query_lower for term in technical_keywords)
        is_conceptual = any(word in query_lower for word in conceptual_words)
        
        return {
            'has_technical_terms': has_technical,
            'is_conceptual': is_conceptual,
            'length': len(query.split()),
            'has_quotes': '"' in query
        }
    
    def visualize_results(
        self,
        results: List[Dict],
        max_text_length: int = 200,
        show_component_scores: bool = True
    ):
        """
        Print formatted hybrid search results
        
        Args:
            results: Search results
            max_text_length: Max text to display
            show_component_scores: Show individual semantic/keyword scores
        """
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"HYBRID SEARCH RESULTS ({len(results)} found)")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"Rank {result['rank']}")
            print(f"─" * 80)
            print(f"Hybrid Score: {result['hybrid_score']:.4f}")
            
            if show_component_scores:
                if 'semantic_score_normalized' in result:
                    print(f"  ├─ Semantic: {result['semantic_score_normalized']:.4f} (weight: {self.semantic_weight:.2f})")
                    print(f"  └─ Keyword:  {result['keyword_score_normalized']:.4f} (weight: {self.keyword_weight:.2f})")
            
            # Metadata
            metadata = result['metadata']
            print(f"\nPaper: {metadata.get('title', 'Unknown')[:60]}")
            print(f"Section: {metadata.get('section_title', 'Unknown')}")
            
            # Text
            text = result['text']
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"\nText: {text}")
            
            print(f"\n{'='*80}\n")


def test_hybrid_search():
    """Test hybrid search"""
    
    # Initialize with custom weights
    searcher = HybridSearcher(
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "BERT architecture",
        "How do transformers work?",
        "training loss optimization"
    ]
    
    print("\n" + "="*80)
    print("TESTING HYBRID SEARCH")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        # Regular hybrid search
        results = searcher.search(
            query,
            top_k=3,
            return_component_scores=True
        )
        
        searcher.visualize_results(results, max_text_length=150)
    
    # Test adaptive search
    print("\n" + "="*80)
    print("TESTING ADAPTIVE SEARCH")
    print("="*80)
    
    adaptive_queries = [
        "transformer architecture model",  # Technical
        "What is the main idea?",          # Conceptual
    ]
    
    for query in adaptive_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        results = searcher.adaptive_search(query, top_k=3)
        searcher.visualize_results(results, max_text_length=150, show_component_scores=False)


if __name__ == "__main__":
    test_hybrid_search()