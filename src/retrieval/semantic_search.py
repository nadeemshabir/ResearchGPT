"""
Semantic Search Module
Find similar chunks using vector embeddings
"""

import os
from typing import List, Dict, Optional
from pathlib import Path

from ..ingestion.database import VectorDatabase
from ..ingestion.embedder import EmbeddingGenerator


class SemanticSearcher:
    """Semantic search using vector embeddings"""
    
    def __init__(
        self,
        embedder: EmbeddingGenerator = None,
        database: VectorDatabase = None
    ):
        """
        Initialize semantic searcher
        
        Args:
            embedder: Embedding generator (creates new if None)
            database: Vector database (creates new if None)
        """
        print("🔍 Initializing Semantic Search...")
        
        # Initialize embedder
        if embedder is None:
            self.embedder = EmbeddingGenerator()
        else:
            self.embedder = embedder
        
        # Initialize database
        if database is None:
            self.database = VectorDatabase()
        else:
            self.database = database
        
        print(f"✅ Semantic search ready")
        print(f"   Database has {self.database.collection.count()} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict] = None,
        return_distances: bool = True
    ) -> List[Dict]:
        """
        Semantic search for similar chunks
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            return_distances: Whether to include similarity scores
            
        Returns:
            List of search results with text, metadata, and scores
        """
        if not query or not query.strip():
            print("⚠️ Empty query provided")
            return []
        
        print(f"\n🔍 Searching for: '{query[:100]}...'")
        print(f"   Top K: {top_k}")
        
        # Step 1: Generate query embedding
        query_embedding = self.embedder.generate_single_embedding(query)
        
        # Step 2: Search database
        raw_results = self.database.query(
            query_embedding=query_embedding,
            n_results=top_k,
            filter_dict=filter_dict
        )
        
        # Step 3: Format results
        results = self._format_results(raw_results, return_distances)
        
        print(f"✅ Found {len(results)} results")
        
        return results
    
    def _format_results(self, raw_results: Dict, include_distance: bool = True) -> List[Dict]:
        """
        Format raw ChromaDB results into clean structure
        
        Args:
            raw_results: Raw results from ChromaDB
            include_distance: Whether to include distance scores
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        # ChromaDB returns nested lists
        documents = raw_results['documents'][0] if raw_results['documents'] else []
        metadatas = raw_results['metadatas'][0] if raw_results['metadatas'] else []
        distances = raw_results['distances'][0] if raw_results['distances'] else []
        ids = raw_results['ids'][0] if raw_results['ids'] else []
        
        for i in range(len(documents)):
            result = {
                'id': ids[i],
                'text': documents[i],
                'metadata': metadatas[i],
                'rank': i + 1
            }
            
            if include_distance and distances:
                # Convert distance to similarity score (0-1, higher is better)
                # ChromaDB uses L2 distance, lower is better
                # Similarity = 1 / (1 + distance)
                distance = distances[i]
                similarity = 1 / (1 + distance)
                
                result['distance'] = round(distance, 4)
                result['similarity_score'] = round(similarity, 4)
            
            formatted.append(result)
        
        return formatted
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 10,
        context_window: int = 1
    ) -> List[Dict]:
        """
        Search and include surrounding chunks for context
        
        Args:
            query: Search query
            top_k: Number of initial results
            context_window: Number of chunks before/after to include
            
        Returns:
            Results with surrounding context
        """
        # Get initial results
        results = self.search(query, top_k=top_k)
        
        # Add surrounding chunks
        enhanced_results = []
        for result in results:
            # Get paper_id and chunk_id from metadata
            paper_id = result['metadata'].get('paper_id')
            chunk_id = result['metadata'].get('chunk_id')
            
            if paper_id and chunk_id is not None:
                # Get surrounding chunks
                context_chunks = self._get_surrounding_chunks(
                    paper_id, 
                    chunk_id, 
                    window=context_window
                )
                
                result['context_before'] = context_chunks['before']
                result['context_after'] = context_chunks['after']
            
            enhanced_results.append(result)
        
        return enhanced_results
    
    def _get_surrounding_chunks(
        self, 
        paper_id: str, 
        chunk_id: int, 
        window: int = 1
    ) -> Dict:
        """
        Get chunks before and after a specific chunk
        
        Args:
            paper_id: Paper identifier
            chunk_id: Target chunk ID
            window: Number of chunks before/after
            
        Returns:
            Dictionary with 'before' and 'after' lists
        """
        # Get all chunks for paper
        all_chunks = self.database.get_by_paper_id(paper_id)
        
        # Sort by chunk_id
        chunks_with_ids = list(zip(
            all_chunks['documents'],
            all_chunks['metadatas']
        ))
        chunks_with_ids.sort(key=lambda x: x[1].get('chunk_id', 0))
        
        # Find target chunk index
        target_idx = None
        for i, (_, metadata) in enumerate(chunks_with_ids):
            if metadata.get('chunk_id') == chunk_id:
                target_idx = i
                break
        
        if target_idx is None:
            return {'before': [], 'after': []}
        
        # Get surrounding chunks
        before = []
        after = []
        
        for i in range(max(0, target_idx - window), target_idx):
            before.append(chunks_with_ids[i][0])
        
        for i in range(target_idx + 1, min(len(chunks_with_ids), target_idx + window + 1)):
            after.append(chunks_with_ids[i][0])
        
        return {'before': before, 'after': after}
    
    def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 5,
        aggregate_method: str = "max"
    ) -> List[Dict]:
        """
        Search with multiple query variations
        
        Args:
            queries: List of query variations
            top_k: Results per query
            aggregate_method: How to combine scores ('max', 'mean', 'sum')
            
        Returns:
            Aggregated and deduplicated results
        """
        print(f"\n🔍 Multi-query search with {len(queries)} queries")
        
        # Search with each query
        all_results = {}
        for query in queries:
            results = self.search(query, top_k=top_k)
            
            for result in results:
                chunk_id = result['id']
                score = result.get('similarity_score', 0)
                
                if chunk_id not in all_results:
                    all_results[chunk_id] = {
                        'result': result,
                        'scores': [score]
                    }
                else:
                    all_results[chunk_id]['scores'].append(score)
        
        # Aggregate scores
        final_results = []
        for chunk_id, data in all_results.items():
            result = data['result']
            scores = data['scores']
            
            if aggregate_method == "max":
                final_score = max(scores)
            elif aggregate_method == "mean":
                final_score = sum(scores) / len(scores)
            elif aggregate_method == "sum":
                final_score = sum(scores)
            else:
                final_score = max(scores)
            
            result['aggregated_score'] = round(final_score, 4)
            result['num_hits'] = len(scores)
            final_results.append(result)
        
        # Sort by aggregated score
        final_results.sort(key=lambda x: x['aggregated_score'], reverse=True)
        
        # Re-rank
        for i, result in enumerate(final_results):
            result['rank'] = i + 1
        
        print(f"✅ Found {len(final_results)} unique results")
        
        return final_results[:top_k]
    
    def visualize_results(self, results: List[Dict], max_text_length: int = 200):
        """
        Print formatted search results
        
        Args:
            results: Search results to display
            max_text_length: Max characters to show per result
        """
        if not results:
            print("No results to display")
            return
        
        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS ({len(results)} found)")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"Rank {result['rank']}")
            print(f"─" * 80)
            
            # Show similarity score if available
            if 'similarity_score' in result:
                score_pct = result['similarity_score'] * 100
                print(f"Similarity: {score_pct:.1f}% | Distance: {result.get('distance', 'N/A')}")
            
            # Show metadata
            metadata = result['metadata']
            print(f"Paper: {metadata.get('title', 'Unknown')[:60]}")
            print(f"Section: {metadata.get('section_title', 'Unknown')}")
            print(f"Chunk: {metadata.get('chunk_id', 'N/A')}")
            
            # Show text
            text = result['text']
            if len(text) > max_text_length:
                text = text[:max_text_length] + "..."
            print(f"\nText: {text}")
            
            # Show context if available
            if 'context_before' in result and result['context_before']:
                print(f"\n[Context before: {len(result['context_before'])} chunks]")
            if 'context_after' in result and result['context_after']:
                print(f"[Context after: {len(result['context_after'])} chunks]")
            
            print(f"\n{'='*80}\n")


def test_semantic_search():
    """Test semantic search"""
    
    # Initialize
    searcher = SemanticSearcher()
    
    # Check if database has data
    if searcher.database.collection.count() == 0:
        print("\n⚠️ Database is empty!")
        print("   Please run the ingestion pipeline first:")
        print("   python src/ingestion/pipeline.py")
        return
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How does BERT work?",
        "Explain transformer architecture",
        "What are the results?"
    ]
    
    print("\n" + "="*80)
    print("TESTING SEMANTIC SEARCH")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"Query: {query}")
        print(f"{'─'*80}")
        
        # Search
        results = searcher.search(query, top_k=3)
        
        # Visualize
        searcher.visualize_results(results, max_text_length=150)
    
    # Test multi-query search
    print("\n" + "="*80)
    print("TESTING MULTI-QUERY SEARCH")
    print("="*80)
    
    query_variations = [
        "What is transformer model?",
        "Explain transformer architecture",
        "How do transformers work?"
        "what is OS?"
    ]
    
    results = searcher.multi_query_search(query_variations, top_k=5)
    searcher.visualize_results(results, max_text_length=150)


if __name__ == "__main__":
    test_semantic_search()