"""
Complete Retrieval System
Orchestrates all retrieval components
"""

from typing import List, Dict, Optional

from .semantic_search import SemanticSearcher
from .keyword_search import KeywordSearcher
from .hybrid_search import HybridSearcher
from .reranker import Reranker
from .query_processor import QueryProcessor


class RetrievalSystem:
    """
    Complete retrieval system combining all components
    """
    
    def __init__(
        self,
        use_reranking: bool = True,
        use_query_processing: bool = True,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize complete retrieval system
        
        Args:
            use_reranking: Whether to use cross-encoder reranking
            use_query_processing: Whether to process queries
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
        """
        print("🚀 Initializing Complete Retrieval System...")
        print("="*60)
        
        self.use_reranking = use_reranking
        self.use_query_processing = use_query_processing
        
        # Initialize components
        print("\n1️⃣ Initializing Hybrid Search...")
        self.hybrid_searcher = HybridSearcher(
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        if use_query_processing:
            print("\n2️⃣ Initializing Query Processor...")
            self.query_processor = QueryProcessor()
        else:
            self.query_processor = None
        
        if use_reranking:
            print("\n3️⃣ Initializing Reranker...")
            self.reranker = Reranker()
        else:
            self.reranker = None
        
        print("\n" + "="*60)
        print("✅ Retrieval System Ready!")
        print("="*60)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 50,
        process_query: bool = None,
        rerank: bool = None,
        return_metadata: bool = True
    ) -> Dict:
        """
        Complete search pipeline
        
        Args:
            query: Search query
            top_k: Number of final results
            rerank_top_k: Number of candidates to rerank
            process_query: Override query processing setting
            rerank: Override reranking setting
            return_metadata: Include search metadata
            
        Returns:
            Dictionary with results and metadata
        """
        if not query or not query.strip():
            return {'results': [], 'metadata': {'error': 'Empty query'}}
        
        print(f"\n{'='*80}")
        print(f"SEARCH QUERY: {query}")
        print(f"{'='*80}")
        
        # Determine settings
        process_query = process_query if process_query is not None else self.use_query_processing
        rerank = rerank if rerank is not None else self.use_reranking
        
        metadata = {
            'original_query': query,
            'processed_query': query,
            'query_variations': [query],
            'search_method': 'hybrid',
            'reranked': rerank,
            'top_k': top_k
        }
        
        # Step 1: Process query (optional)
        if process_query and self.query_processor:
            print("\n📝 Step 1: Processing query...")
            query_result = self.query_processor.process_query(query)
            processed_query = query_result['cleaned']
            query_variations = query_result['variations']
            
            metadata['processed_query'] = processed_query
            metadata['query_variations'] = query_variations
            metadata['query_intent'] = query_result['intent']
            metadata['key_terms'] = query_result['key_terms']
            
            # Use best variation for search
            search_query = processed_query
        else:
            search_query = query
        
        # Step 2: Hybrid search
        print("\n🔍 Step 2: Hybrid search...")
        results = self.hybrid_searcher.search(
            query=search_query,
            top_k=rerank_top_k if rerank else top_k,
            return_component_scores=True
        )
        
        metadata['initial_results_count'] = len(results)
        
        if not results:
            print("⚠️ No results found")
            return {'results': [], 'metadata': metadata}
        
        # Step 3: Re-ranking (optional)
        if rerank and self.reranker and len(results) > 1:
            print(f"\n🔄 Step 3: Re-ranking top {len(results)} results...")
            results = self.reranker.rerank(
                query=query,  # Use original query for reranking
                results=results,
                top_k=top_k
            )
            metadata['reranked_count'] = len(results)
        else:
            results = results[:top_k]
        
        # Step 4: Format final results
        print(f"\n✅ Returning {len(results)} results")
        
        if not return_metadata:
            return {'results': results}
        
        return {
            'results': results,
            'metadata': metadata
        }
    
    def multi_query_search(
        self,
        query: str,
        top_k: int = 10,
        num_variations: int = 3
    ) -> Dict:
        """
        Search using multiple query variations
        
        Args:
            query: Original query
            top_k: Number of final results
            num_variations: Number of query variations
            
        Returns:
            Search results
        """
        print(f"\n🔍 Multi-Query Search: {num_variations} variations")
        
        # Generate query variations
        if self.query_processor:
            variations = self.query_processor.generate_query_variations(
                query,
                num_variations=num_variations
            )
        else:
            variations = [query]
        
        print(f"   Generated {len(variations)} variations:")
        for i, var in enumerate(variations, 1):
            print(f"   {i}. {var}")
        
        # Search with each variation
        all_results = {}
        for i, var_query in enumerate(variations, 1):
            print(f"\n   Searching with variation {i}...")
            var_results = self.hybrid_searcher.search(var_query, top_k=top_k*2)
            
            # Aggregate results
            for result in var_results:
                chunk_id = result['id']
                score = result.get('hybrid_score', 0)
                
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        'result': result,
                        'scores': [score],
                        'found_in_variations': 1
                    }
                else:
                    all_results[chunk_id]['scores'].append(score)
                    all_results[chunk_id]['found_in_variations'] += 1
        
        # Aggregate scores (mean)
        final_results = []
        for chunk_id, data in all_results.items():
            result = data['result']
            result['multi_query_score'] = round(sum(data['scores']) / len(data['scores']), 4)
            result['found_in_variations'] = data['found_in_variations']
            final_results.append(result)
        
        # Sort by multi-query score
        final_results.sort(key=lambda x: x['multi_query_score'], reverse=True)
        
        # Re-rank
        for i, result in enumerate(final_results[:top_k], 1):
            result['rank'] = i
        
        return {
            'results': final_results[:top_k],
            'metadata': {
                'query': query,
                'variations': variations,
                'num_variations': len(variations),
                'unique_results': len(all_results)
            }
        }
    
    def search_with_filters(
        self,
        query: str,
        filters: Dict,
        top_k: int = 10
    ) -> Dict:
        """
        Search with metadata filters
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {'author': 'John Doe'})
            top_k: Number of results
            
        Returns:
            Filtered search results
        """
        print(f"\n🔍 Filtered Search")
        print(f"   Query: {query}")
        print(f"   Filters: {filters}")
        
        # Search with filters
        semantic_searcher = self.hybrid_searcher.semantic_searcher
        results = semantic_searcher.search(
            query=query,
            top_k=top_k,
            filter_dict=filters
        )
        
        return {
            'results': results,
            'metadata': {
                'query': query,
                'filters': filters,
                'num_results': len(results)
            }
        }
    
    def get_relevant_chunks(
        self,
        query: str,
        max_tokens: int = 4000,
        min_score: float = 0.5
    ) -> Dict:
        """
        Get relevant chunks for LLM context
        
        Args:
            query: Search query
            max_tokens: Maximum tokens for context window
            min_score: Minimum relevance score
            
        Returns:
            Chunks formatted for LLM context
        """
        print(f"\n📝 Getting relevant chunks for LLM")
        print(f"   Max tokens: {max_tokens}")
        print(f"   Min score: {min_score}")
        
        # Search with high top_k
        search_result = self.search(query, top_k=50)
        results = search_result['results']
        
        # Filter by score
        if self.use_reranking:
            filtered = [r for r in results if r.get('rerank_score', 0) >= min_score]
        else:
            filtered = [r for r in results if r.get('hybrid_score', 0) >= min_score]
        
        print(f"   Found {len(filtered)} chunks above threshold")
        
        # Build context within token limit
        selected_chunks = []
        total_tokens = 0
        
        for result in filtered:
            chunk_tokens = result['metadata'].get('num_tokens', len(result['text']) // 4)
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append(result)
                total_tokens += chunk_tokens
            else:
                break
        
        print(f"   Selected {len(selected_chunks)} chunks ({total_tokens} tokens)")
        
        # Format context
        context = "\n\n---\n\n".join([
            f"[Source {i+1}: {chunk['metadata'].get('title', 'Unknown')}]\n{chunk['text']}"
            for i, chunk in enumerate(selected_chunks)
        ])
        
        return {
            'context': context,
            'chunks': selected_chunks,
            'metadata': {
                'num_chunks': len(selected_chunks),
                'total_tokens': total_tokens,
                'query': query
            }
        }
    
    def visualize_results(self, search_result: Dict, max_results: int = 5):
        """
        Visualize search results
        
        Args:
            search_result: Result from search()
            max_results: Max results to display
        """
        results = search_result['results'][:max_results]
        metadata = search_result.get('metadata', {})
        
        print(f"\n{'='*80}")
        print("SEARCH RESULTS")
        print(f"{'='*80}")
        
        # Show metadata
        print(f"\nQuery: {metadata.get('original_query', 'N/A')}")
        if 'processed_query' in metadata and metadata['processed_query'] != metadata.get('original_query'):
            print(f"Processed: {metadata['processed_query']}")
        
        if 'query_intent' in metadata:
            print(f"Intent: {metadata['query_intent'].get('type', 'N/A')}")
        
        print(f"Method: {metadata.get('search_method', 'N/A')}")
        print(f"Reranked: {metadata.get('reranked', False)}")
        print(f"\nShowing {len(results)} of {len(search_result['results'])} results")
        print(f"{'='*80}\n")
        
        # Show results
        for result in results:
            print(f"Rank {result['rank']}")
            print("─" * 80)
            
            # Scores
            if 'rerank_score' in result:
                print(f"Rerank Score: {result['rerank_score']:.4f}")
                if 'original_rank' in result:
                    print(f"  (Original rank: {result['original_rank']})")
            elif 'hybrid_score' in result:
                print(f"Hybrid Score: {result['hybrid_score']:.4f}")
            
            # Metadata
            meta = result['metadata']
            print(f"\nPaper: {meta.get('title', 'Unknown')[:70]}")
            print(f"Section: {meta.get('section_title', 'Unknown')}")
            print(f"Chunk: {meta.get('chunk_id', 'N/A')}")
            
            # Text preview
            text = result['text']
            if len(text) > 250:
                text = text[:250] + "..."
            print(f"\n{text}")
            
            print(f"\n{'='*80}\n")


def test_retrieval_system():
    """Test complete retrieval system"""
    
    # Initialize system
    system = RetrievalSystem(
        use_reranking=True,
        use_query_processing=True,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How does transformer architecture work?",
        "Compare BERT and GPT models",
    ]
    
    print("\n" + "="*80)
    print("TESTING COMPLETE RETRIEVAL SYSTEM")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        # Search
        result = system.search(query, top_k=3)
        
        # Visualize
        system.visualize_results(result, max_results=3)
    
    # Test multi-query search
    print("\n" + "="*80)
    print("TESTING MULTI-QUERY SEARCH")
    print("="*80)
    
    result = system.multi_query_search(
        "transformer attention mechanism",
        top_k=5,
        num_variations=3
    )
    
    system.visualize_results(result, max_results=5)
    
    # Test context retrieval for LLM
    print("\n" + "="*80)
    print("TESTING CONTEXT RETRIEVAL FOR LLM")
    print("="*80)
    
    context_result = system.get_relevant_chunks(
        "What is attention mechanism?",
        max_tokens=2000,
        min_score=3.0
    )
    
    print(f"\nRetrieved {context_result['metadata']['num_chunks']} chunks")
    print(f"Total tokens: {context_result['metadata']['total_tokens']}")
    print(f"\nContext preview (first 500 chars):")
    print(context_result['context'][:500] + "...")


if __name__ == "__main__":
    test_retrieval_system()