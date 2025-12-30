"""
Complete Ingestion Pipeline
Orchestrates PDF processing from upload to database storage
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv

from .pdf_parser import PDFParser
from .chunker import TextChunker
from .embedder import EmbeddingGenerator
from .database import VectorDatabase

load_dotenv()


class IngestionPipeline:
    """Complete pipeline for processing and storing research papers"""
    
    def __init__(
        self,
        pdf_method: str = "pymupdf",
        chunk_size: int = None,
        chunk_overlap: int = None,
        embedding_model: str = None
    ):
        """
        Initialize ingestion pipeline
        
        Args:
            pdf_method: PDF parsing method
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            embedding_model: Embedding model name
        """
        print("🚀 Initializing Ingestion Pipeline...")
        print("="*60)
        
        # Initialize components
        self.pdf_parser = PDFParser(method=pdf_method)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        self.database = VectorDatabase()
        
        print("="*60)
        print("✅ Pipeline ready!")
    
    def process_paper(
        self,
        pdf_path: str,
        paper_id: Optional[str] = None,
        use_context: bool = True
    ) -> Dict:
        """
        Process a single paper through the complete pipeline
        
        Args:
            pdf_path: Path to PDF file
            paper_id: Unique identifier (uses filename if None)
            use_context: Whether to use context-aware chunking
            
        Returns:
            Processing statistics
        """
        start_time = time.time()
        
        # Generate paper ID if not provided
        if paper_id is None:
            paper_id = Path(pdf_path).stem
        
        print(f"\n{'='*60}")
        print(f"PROCESSING PAPER: {paper_id}")
        print(f"{'='*60}")
        
        # Step 1: Extract text from PDF
        print("\n📄 Step 1: Extracting text from PDF...")
        pdf_result = self.pdf_parser.extract_text(pdf_path)
        
        # Clean text
        text = self.pdf_parser.clean_text(pdf_result['text'])
        
        # Step 2: Chunk text
        print(f"\n🔪 Step 2: Chunking text...")
        if use_context:
            chunks = self.chunker.chunk_with_context(
                text=text,
                metadata=pdf_result['metadata']
            )
        else:
            chunks = self.chunker.chunk_text(
                text=text,
                metadata=pdf_result['metadata']
            )
        
        # Step 3: Generate embeddings
        print(f"\n🧠 Step 3: Generating embeddings...")
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 4: Store in database
        print(f"\n💾 Step 4: Storing in database...")
        paper_metadata = {
            'title': pdf_result['metadata'].get('title', Path(pdf_path).stem),
            'author': pdf_result['metadata'].get('author', 'Unknown'),
            'file_name': pdf_result['file_name'],
            'num_pages': pdf_result['num_pages']
        }
        
        num_stored = self.database.add_chunks(
            chunks=chunks_with_embeddings,
            paper_id=paper_id,
            paper_metadata=paper_metadata
        )
        
        # Calculate stats
        processing_time = time.time() - start_time
        
        stats = {
            'paper_id': paper_id,
            'file_name': pdf_result['file_name'],
            'num_pages': pdf_result['num_pages'],
            'num_chunks': len(chunks),
            'num_stored': num_stored,
            'processing_time_seconds': round(processing_time, 2),
            'metadata': paper_metadata
        }
        
        print(f"\n{'='*60}")
        print("✅ PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Paper ID: {paper_id}")
        print(f"Pages: {stats['num_pages']}")
        print(f"Chunks created: {stats['num_chunks']}")
        print(f"Chunks stored: {stats['num_stored']}")
        print(f"Processing time: {stats['processing_time_seconds']}s")
        print(f"{'='*60}\n")
        
        return stats
    
    def process_directory(
        self,
        directory_path: str,
        file_pattern: str = "*.pdf"
    ) -> list[Dict]:
        """
        Process all PDFs in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            file_pattern: File pattern to match (default: *.pdf)
            
        Returns:
            List of processing statistics for each paper
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all PDFs
        pdf_files = list(directory.glob(file_pattern))
        
        if not pdf_files:
            print(f"⚠️  No PDF files found in {directory}")
            return []
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING: {len(pdf_files)} PDFs")
        print(f"{'='*60}")
        
        all_stats = []
        successful = 0
        failed = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
            
            try:
                stats = self.process_paper(str(pdf_file))
                all_stats.append(stats)
                successful += 1
            except Exception as e:
                print(f"❌ ERROR processing {pdf_file.name}: {str(e)}")
                failed += 1
                all_stats.append({
                    'file_name': pdf_file.name,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*60}\n")
        
        return all_stats
    
    def get_pipeline_stats(self) -> Dict:
        """Get statistics about the pipeline and database"""
        db_stats = self.database.get_stats()
        model_info = self.embedder.get_model_info()
        
        return {
            'database': db_stats,
            'embedding_model': model_info,
            'chunker': {
                'chunk_size': self.chunker.chunk_size,
                'chunk_overlap': self.chunker.chunk_overlap
            }
        }


def main():
    """Example usage of the ingestion pipeline"""
    
    # Initialize pipeline
    pipeline = IngestionPipeline(
        pdf_method="pymupdf",
        chunk_size=1000,
        chunk_overlap=200,
        embedding_model="all-MiniLM-L6-v2"  # Fast model for testing
    )
    
    # Test with single PDF
    # test_pdf = "data/raw/test_paper.pdf"

    # Test with directory of PDFs
    test_pdf = "data/raw/"
    
    if os.path.exists(test_pdf):
        print("\n🧪 Processing single test paper...")
        # stats = pipeline.process_paper(test_pdf)

        stats = pipeline.process_directory(test_pdf)
        
        # Show database stats
        pipeline.database.print_stats()
    else:
        print(f"\n⚠️  Test PDF not found: {test_pdf}")
        print("   Please place a PDF in data/raw/test_paper.pdf")
        print("\n📂 To process multiple PDFs:")
        print("   1. Place PDFs in data/raw/")
        print("   2. Run: pipeline.process_directory('data/raw/')")


if __name__ == "__main__":
    main()