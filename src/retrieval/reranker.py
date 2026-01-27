"""
Re-ranking Module
Improve retrieval results using cross-encoder models
It does not return vectors. It returns the semantic text content that the Answer Generator needs to actually write the response
"""

import sys
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("⚠️ Installing sentence-transformers for CrossEncoder...")
    import os
    os.system("pip install sentence-transformers")
    from sentence_transformers import CrossEncoder


class Reranker:
    """Re-rank search results using cross-encoder"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None
    ):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        print("🔄 Initializing Reranker...")
        print(f"   Model: {model_name}")
        
        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        print(f"   Device: {device}")
        
        # Load cross-encoder model
        self.model = CrossEncoder(model_name, device=device)
        
        print("✅ Reranker ready")
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = None,
        score_key: str = 'rerank_score'
    ) -> List[Dict]:
        """
        Re-rank search results using cross-encoder
        
        Args:
            query: Original search query
            results: Initial search results
            top_k: Number of results to return (None = all)
            score_key: Key to store rerank score
            
        Returns:
            Re-ranked results
        """
        if not results:
            print("⚠️ No results to rerank")
            return []
        
        print(f"\n🔄 Re-ranking {len(results)} results...")
        
        # Prepare query-document pairs
        pairs = [[query, result['text']] for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        for result, score in zip(results, scores):
            result[score_key] = round(float(score), 4)
            # Keep original score for comparison
            if 'original_score' not in result:
                result['original_score'] = result.get('hybrid_score', result.get('similarity_score', 0))
        
        # Sort by rerank score
        results.sort(key=lambda x: x[score_key], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results, 1):
            result['rerank_rank'] = i
            result['original_rank'] = result.get('rank', i)
        
        # Trim to top_k if specified
        if top_k:
            results = results[:top_k]
        
        print(f"✅ Re-ranking complete")
        
        return results
    
    def rerank_with_threshold(
        self,
        query: str,
        results: List[Dict],
        threshold: float = 0.5,
        top_k: int = None
    ) -> List[Dict]:
        """
        Re-rank and filter by score threshold
        
        Args:
            query: Search query
            results: Initial results
            threshold: Minimum score to keep
            top_k: Max results to return
            
        Returns:
            Filtered and re-ranked results
        """
        # Re-rank
        reranked = self.rerank(query, results)
        
        # Filter by threshold
        filtered = [r for r in reranked if r.get('rerank_score', 0) >= threshold]
        
        print(f"   Kept {len(filtered)}/{len(reranked)} results above threshold {threshold}")
        
        if top_k:
            filtered = filtered[:top_k]
        
        return filtered
    
    def compare_rankings(
        self,
        query: str,
        results: List[Dict],
        top_n: int = 5
    ):
        """
        Compare original vs reranked order
        
        Args:
            query: Search query
            results: Results to compare
            top_n: Number of top results to show
        """
        # Store original order
        original_order = [(r.get('rank', i), r['text'][:100]) for i, r in enumerate(results, 1)]
        
        # Rerank
        reranked = self.rerank(query, results.copy())
        
        print(f"\n{'='*80}")
        print(f"RANKING COMPARISON (Top {top_n})")
        print(f"{'='*80}\n")
        
        print("ORIGINAL ORDER:")
        print("─" * 80)
        for i, (rank, text) in enumerate(original_order[:top_n], 1):
            print(f"{i}. [Original Rank: {rank}] {text}...")
        
        print(f"\n{'='*80}")
        print("AFTER RE-RANKING:")
        print("─" * 80)
        for i, result in enumerate(reranked[:top_n], 1):
            original_rank = result.get('original_rank', 'N/A')
            rerank_score = result.get('rerank_score', 0)
            print(f"{i}. [Was: {original_rank}, Score: {rerank_score:.3f}] {result['text'][:100]}...")
        
        print(f"\n{'='*80}\n")
    
    def analyze_score_distribution(self, results: List[Dict]):
        """
        Analyze distribution of rerank scores
        
        Args:
            results: Results with rerank scores
        """
        if not results or 'rerank_score' not in results[0]:
            print("⚠️ No rerank scores found")
            return
        
        scores = [r['rerank_score'] for r in results]
        
        print(f"\n{'='*80}")
        print("RERANK SCORE DISTRIBUTION")
        print(f"{'='*80}")
        print(f"Count: {len(scores)}")
        print(f"Min: {min(scores):.4f}")
        print(f"Max: {max(scores):.4f}")
        print(f"Mean: {sum(scores)/len(scores):.4f}")
        print(f"Median: {sorted(scores)[len(scores)//2]:.4f}")
        
        # Score ranges
        high = sum(1 for s in scores if s > 5.0)
        medium = sum(1 for s in scores if 2.0 <= s <= 5.0)
        low = sum(1 for s in scores if s < 2.0)
        
        print(f"\nScore Ranges:")
        print(f"  High (>5.0): {high} ({high/len(scores)*100:.1f}%)")
        print(f"  Medium (2-5): {medium} ({medium/len(scores)*100:.1f}%)")
        print(f"  Low (<2.0): {low} ({low/len(scores)*100:.1f}%)")
        print(f"{'='*80}\n")


class EnsembleReranker:
    """
    Use multiple reranking models for ensemble
    """
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize ensemble reranker
        
        Args:
            model_names: List of cross-encoder models
        """
        if model_names is None:
            model_names = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2"
            ]
        
        print(f"🔄 Initializing Ensemble Reranker with {len(model_names)} models...")
        
        self.rerankers = []
        for model_name in model_names:
            print(f"   Loading: {model_name}")
            self.rerankers.append(Reranker(model_name=model_name))
        
        print("✅ Ensemble reranker ready")
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        aggregate_method: str = "mean",
        top_k: int = None
    ) -> List[Dict]:
        """
        Re-rank using ensemble of models
        
        Args:
            query: Search query
            results: Initial results
            aggregate_method: How to combine scores ('mean', 'max', 'weighted')
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        print(f"\n🔄 Ensemble re-ranking with {len(self.rerankers)} models...")
        
        # Get scores from each model
        all_scores = []
        for i, reranker in enumerate(self.rerankers):
            print(f"   Model {i+1}/{len(self.rerankers)}...")
            reranked = reranker.rerank(query, results.copy(), score_key=f'score_{i}')
            scores = [r[f'score_{i}'] for r in reranked]
            all_scores.append(scores)
        
        # Aggregate scores
        if aggregate_method == "mean":
            ensemble_scores = [sum(s[i] for s in all_scores) / len(all_scores) for i in range(len(results))]
        elif aggregate_method == "max":
            ensemble_scores = [max(s[i] for s in all_scores) for i in range(len(results))]
        elif aggregate_method == "weighted":
            # Weight first model more
            weights = [0.6, 0.4] if len(all_scores) == 2 else [1.0/len(all_scores)] * len(all_scores)
            ensemble_scores = [sum(s[i] * w for s, w in zip(all_scores, weights)) for i in range(len(results))]
        else:
            ensemble_scores = [sum(s[i] for s in all_scores) / len(all_scores) for i in range(len(results))]
        
        # Add ensemble scores
        for result, score in zip(results, ensemble_scores):
            result['ensemble_score'] = round(score, 4)
        
        # Sort by ensemble score
        results.sort(key=lambda x: x['ensemble_score'], reverse=True)
        
        # Update ranks
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        if top_k:
            results = results[:top_k]
        
        print(f"✅ Ensemble re-ranking complete")
        
        return results


def test_reranker():
    """Test reranking"""
    
    # Create mock results
    mock_results = [
        {
            'rank': 1,
            'text': 'Transformers use self-attention mechanisms to process sequences.',
            'similarity_score': 0.85,
            'id': 'chunk_1',
            'metadata': {'title': 'Attention Paper'}
        },
        {
            'rank': 2,
            'text': 'BERT is trained using masked language modeling.',
            'similarity_score': 0.75,
            'id': 'chunk_2',
            'metadata': {'title': 'BERT Paper'}
        },
        {
            'rank': 3,
            'text': 'The attention mechanism computes weighted sums of values.',
            'similarity_score': 0.70,
            'id': 'chunk_3',
            'metadata': {'title': 'Attention Paper'}
        },
        {
            'rank': 4,
            'text': 'Neural networks consist of layers of interconnected nodes.',
            'similarity_score': 0.65,
            'id': 'chunk_4',
            'metadata': {'title': 'General NN Paper'}
        },
        {
            'rank': 5,
            'text': 'Attention weights determine which parts of input are important.',
            'similarity_score': 0.60,
            'id': 'chunk_5',
            'metadata': {'title': 'Attention Paper'}
        }
    ]
    
    query = "How does attention mechanism work in transformers?"
    
    print("\n" + "="*80)
    print("TESTING RERANKER")
    print("="*80)
    
    # Initialize reranker
    reranker = Reranker()
    
    # Compare rankings
    reranker.compare_rankings(query, mock_results, top_n=5)
    
    # Rerank
    reranked = reranker.rerank(query, mock_results.copy())
    
    # Analyze scores
    reranker.analyze_score_distribution(reranked)
    
    # Test threshold filtering
    print("\n" + "="*80)
    print("TESTING THRESHOLD FILTERING")
    print("="*80)
    filtered = reranker.rerank_with_threshold(
        query,
        mock_results.copy(),
        threshold=3.0,
        top_k=3
    )
    
    print(f"\nFiltered to {len(filtered)} high-quality results:")
    for result in filtered:
        print(f"  Score: {result['rerank_score']:.3f} - {result['text'][:80]}...")


if __name__ == "__main__":
    test_reranker()