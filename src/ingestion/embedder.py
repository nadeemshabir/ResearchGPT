"""
Embedding Generator Module
Converts text chunks into vector embeddings
"""

import os
from typing import List
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of sentence-transformer model (default from .env)
            device: Device to use ('cpu', 'cuda', 'mps') - auto-detected if None
        """
        self.model_name = model_name or os.getenv('EMBEDDING_MODEL', 'allenai/specter')
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon
            else:
                device = 'cpu'
        
        self.device = device
        
        print(f"🧠 Loading embedding model: {self.model_name}")
        print(f"   Device: {self.device}")
        
        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"✅ Model loaded")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        print(f"\n🔢 Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        
        # Convert to list of lists for storage
        return embeddings.tolist()
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_chunks(self, chunks: List[dict], batch_size: int = 32) -> List[dict]:
        """
        Generate embeddings for chunks and add to chunk dictionaries
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            batch_size: Batch size for encoding
            
        Returns:
            Chunks with added 'embedding' field
        """
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        return chunks
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model"""
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': self.device,
            'max_seq_length': self.model.max_seq_length
        }


class MultiModelEmbedder:
    """
    Use multiple embedding models for comparison or ensemble
    """
    
    def __init__(self, model_names: List[str], device: str = None):
        """
        Initialize multiple embedding models
        
        Args:
            model_names: List of model names to use
            device: Device to use
        """
        self.models = {}
        
        for model_name in model_names:
            print(f"\nLoading model: {model_name}")
            self.models[model_name] = EmbeddingGenerator(model_name, device)
    
    def generate_embeddings(self, texts: List[str], model_name: str = None) -> List[List[float]]:
        """
        Generate embeddings using specified model
        
        Args:
            texts: List of texts to embed
            model_name: Which model to use (uses first if None)
            
        Returns:
            List of embeddings
        """
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        return self.models[model_name].generate_embeddings(texts)
    
    def compare_models(self, sample_text: str):
        """
        Compare embeddings from different models
        
        Args:
            sample_text: Text to embed with all models
        """
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"Sample text: {sample_text[:100]}...")
        print()
        
        embeddings = {}
        for model_name, generator in self.models.items():
            embedding = generator.generate_single_embedding(sample_text)
            embeddings[model_name] = embedding
            print(f"{model_name}:")
            print(f"  Dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print()


def test_embedder():
    """Test the embedding generator"""
    
    # Sample texts
    sample_texts = [
        "Transformers are neural network architectures based on attention mechanisms.",
        "BERT uses bidirectional attention to understand context.",
        "GPT models generate text using autoregressive language modeling.",
        "Attention mechanisms allow models to focus on relevant information."
    ]
    
    print("="*60)
    print("TEST 1: Single Model")
    print("="*60)
    
    # Test single model
    embedder = EmbeddingGenerator(model_name='all-MiniLM-L6-v2')  # Faster for testing, check other models in notes/.env
    #if we dont want to specify model_name, we can just do embedder = EmbeddingGenerator()
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(sample_texts)
    
    print(f"\nGenerated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First embedding (first 5 values): {embeddings[0][:5]}")
    
    # Test with chunks
    print("\n" + "="*60)
    print("TEST 2: Embed Chunks")
    print("="*60)
    
    chunks = [
        {'text': text, 'chunk_id': i}
        for i, text in enumerate(sample_texts)
    ]
    
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    
    print(f"\nProcessed {len(chunks_with_embeddings)} chunks")
    print(f"Chunk 0 has embedding: {'embedding' in chunks_with_embeddings[0]}")
    print(f"Embedding length: {len(chunks_with_embeddings[0]['embedding'])}")
    print(type(embeddings))
    print(type(embeddings[0]))
    print(len(embeddings))
    print(len(embeddings[0]))

    # Model info
    print("\n" + "="*60)
    print("MODEL INFO")
    print("="*60)
    info = embedder.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_embedder()