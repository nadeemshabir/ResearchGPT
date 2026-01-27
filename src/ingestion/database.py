"""
Vector Database Module
Store and retrieve embeddings using ChromaDB
both text and embeddings are stored in the database
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    """Manage vector database operations"""
    
    def __init__(
        self,
        db_path: str = None,
        collection_name: str = None
    ):
        """
        Initialize vector database
        
        Args:
            db_path: Path to store database (default from .env)
            collection_name: Name of collection (default from .env)
        """
        self.db_path = db_path or os.getenv('CHROMA_DB_PATH', './data/chroma_db')
        self.collection_name = collection_name or os.getenv('COLLECTION_NAME', 'research_papers')
        
        # Create db directory if needed
        os.makedirs(self.db_path, exist_ok=True)
        
        print(f"🗄️  Initializing vector database...")
        print(f"   Path: {self.db_path}")
        print(f"   Collection: {self.collection_name}")
        
        # Initialize ChromaDB client,  This means data will be saved to disk and persist between restarts.
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✅ Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Research papers embeddings"}
            )
            print(f"✅ Created new collection")
    
    def add_chunks(
        self,
        chunks: List[Dict],
        paper_id: str,
        paper_metadata: Dict = None
    ) -> int:
        """
        Add chunks to database
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'embedding' fields
            paper_id: Unique identifier for the paper
            paper_metadata: Metadata about the paper
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            print("⚠️  No chunks to add")
            return 0
        
        print(f"\n💾 Adding {len(chunks)} chunks to database...")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Create unique ID for chunk
            chunk_id = f"{paper_id}_chunk_{chunk['chunk_id']}"
            ids.append(chunk_id)
            
            # Add embedding
            embeddings.append(chunk['embedding'])
            
            # Add text
            documents.append(chunk['text'])
            
            # Prepare metadata
            metadata = {
                'paper_id': paper_id,
                'chunk_id': chunk['chunk_id'],
                'num_tokens': chunk.get('num_tokens', 0),
                'section_title': chunk.get('section_title', 'Unknown')
            }
            
            # Add paper metadata
            if paper_metadata:
                metadata.update(paper_metadata)
            
            # Add chunk metadata if exists
            if 'metadata' in chunk:
                metadata.update(chunk['metadata'])
            
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks")
        print(f"   Total documents in DB: {self.collection.count()}")
        
        return len(chunks)
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filter_dict: Dict = None
    ) -> Dict:
        """
        Query database for similar chunks
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Query results with documents, distances, and metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def get_by_paper_id(self, paper_id: str) -> Dict:
        """
        Get all chunks for a specific paper
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            All chunks for the paper
        """
        results = self.collection.get(
            where={"paper_id": paper_id}
        )
        
        return results
    
    def delete_paper(self, paper_id: str) -> int:
        """
        Delete all chunks for a paper
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Number of chunks deleted
        """
        # Get all IDs for this paper
        results = self.get_by_paper_id(paper_id)
        ids = results['ids']
        
        if ids:
            self.collection.delete(ids=ids)
            print(f"🗑️  Deleted {len(ids)} chunks for paper: {paper_id}")
        else:
            print(f"⚠️  No chunks found for paper: {paper_id}")
        
        return len(ids)
    
    def list_papers(self) -> List[Dict]:
        """
        List all papers in database
        
        Returns:
            List of paper metadata
        """
        # Get all documents
        all_docs = self.collection.get()
        
        # Extract unique papers
        papers = {}
        for metadata in all_docs['metadatas']:
            paper_id = metadata.get('paper_id', 'unknown')
            if paper_id not in papers:
                papers[paper_id] = {
                    'paper_id': paper_id,
                    'title': metadata.get('title', 'Unknown'),
                    'author': metadata.get('author', 'Unknown'),
                    'num_chunks': 0
                }
            papers[paper_id]['num_chunks'] += 1
        
        return list(papers.values())
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        total_docs = self.collection.count()
        papers = self.list_papers()
        
        stats = {
            'total_chunks': total_docs,
            'total_papers': len(papers),
            'collection_name': self.collection_name,
            'db_path': self.db_path
        }
        
        if papers:
            stats['avg_chunks_per_paper'] = total_docs / len(papers)
        
        return stats
    
    def print_stats(self):
        """Print database statistics"""
        stats = self.get_stats()
        
        print(f"\n{'='*60}")
        print("DATABASE STATISTICS")
        print(f"{'='*60}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\nPapers in database:")
        papers = self.list_papers()
        for paper in papers[:10]:  # Show first 10
            print(f"  • {paper['title'][:50]}... ({paper['num_chunks']} chunks)")
        
        if len(papers) > 10:
            print(f"  ... and {len(papers) - 10} more papers")
    
    def reset_database(self):
        """Delete all data in collection"""
        print(f"⚠️  Resetting database...")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Research papers embeddings"}
        )
        print(f"✅ Database reset complete")


def test_database():
    """Test the vector database"""
    
    print("="*60)
    print("TEST: Vector Database")
    print("="*60)
    
    # Initialize database
    db = VectorDatabase()
    
    # Create sample chunks with embeddings
    sample_chunks = [
        {
            'text': 'Transformers use attention mechanisms.',
            'chunk_id': 0,
            'num_tokens': 10,
            'embedding': [0.1, 0.2, 0.3, 0.4, 0.5] * 10  # Mock embedding
        },
        {
            'text': 'BERT is a bidirectional transformer.',
            'chunk_id': 1,
            'num_tokens': 8,
            'embedding': [0.2, 0.3, 0.4, 0.5, 0.6] * 10
        },
        {
            'text': 'GPT models generate text autoregressively.',
            'chunk_id': 2,
            'num_tokens': 9,
            'embedding': [0.3, 0.4, 0.5, 0.6, 0.7] * 10
        }
    ]
    
    paper_metadata = {
        'title': 'Test Paper on Transformers',
        'author': 'Test Author',
        'year': '2024'
    }
    
    # Add chunks
    db.add_chunks(
        chunks=sample_chunks,
        paper_id='test_paper_001',
        paper_metadata=paper_metadata
    )
    
    # Query
    print("\n" + "="*60)
    print("TEST: Querying")
    print("="*60)
    
    query_embedding = [0.15, 0.25, 0.35, 0.45, 0.55] * 10
    results = db.query(query_embedding, n_results=2)
    
    print(f"\nQuery results:")
    print(f"  Found {len(results['documents'][0])} results")
    for i, doc in enumerate(results['documents'][0]):
        print(f"  {i+1}. {doc[:50]}...")
        print(f"     Distance: {results['distances'][0][i]:.4f}")
    
    # Stats
    db.print_stats()
    
    # Optional: Reset for clean slate
    # db.reset_database()


if __name__ == "__main__":
    test_database()