"""
Text Chunking Module
Splits text into optimal chunks for embedding and retrieval
"""

import os
from typing import List, Dict
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


class TextChunker:
    """Smart text chunking for academic papers"""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        encoding_name: str = "cl100k_base" #same tokenizer as used in openai gpt-4 model
    ):
        """
        Initialize text chunker
        
        Args:
            chunk_size: Target size of each chunk in tokens (default from .env)
            chunk_overlap: Overlap between chunks in tokens (default from .env)
            encoding_name: Tokenizer encoding to use, this actually counts tokens while chunking
        """
        self.chunk_size = chunk_size or int(os.getenv('CHUNK_SIZE', 1000))# IF USER DOESNT PROVIDE CHUNK SIZE, IT WILL TAKE FROM .ENV FILE and wwhen we dont provide anything in .env file, it will take 1000 as default
        self.chunk_overlap = chunk_overlap or int(os.getenv('CHUNK_OVERLAP', 200))
        
        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize text splitter, Split long text into chunks of a given size, but split it nicely
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=[
                "\n\n\n",  # Multiple newlines (section breaks)
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentences
                " ",       # Words
                ""         # Characters (fallback)
            ]
        )
        
        print(f"📐 Chunker initialized:")
        print(f"   Chunk size: {self.chunk_size} tokens")
        print(f"   Overlap: {self.chunk_overlap} tokens")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks
        
        Args:
            text: Full text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        print(f"\n🔪 Chunking text...")
        print(f"   Input length: {len(text)} characters")
        print(f"   Input tokens: {self._count_tokens(text)}")
        
        # Split text
        raw_chunks = self.splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunk = {
                'text': chunk_text,
                'chunk_id': idx,
                'num_tokens': self._count_tokens(chunk_text),
                'num_characters': len(chunk_text)
            }
            
            # Add metadata if provided
            if metadata:
                chunk['metadata'] = metadata.copy()
                chunk['metadata']['chunk_id'] = idx
            
            chunks.append(chunk)
        
        print(f"✅ Created {len(chunks)} chunks")
        print(f"   Avg tokens per chunk: {sum(c['num_tokens'] for c in chunks) / len(chunks):.0f}")
        
        return chunks
    
    def chunk_with_context(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Advanced chunking with section context
        Tries to identify and preserve section headers
        
        Args:
            text: Full text to chunk
            metadata: Optional metadata
            
        Returns:
            List of chunks with added context
        """
        # First, try to identify sections
        sections = self._identify_sections(text)
        
        if len(sections) > 1:
            print(f"   Found {len(sections)} sections")
            # Chunk each section separately
            all_chunks = []
            for section in sections:
                section_chunks = self.chunk_text(section['text'], metadata)
                # Add section context to each chunk
                for chunk in section_chunks:
                    chunk['section_title'] = section.get('title', 'Unknown')
                all_chunks.extend(section_chunks)
            return all_chunks
        else:
            # No clear sections, use regular chunking
            return self.chunk_text(text, metadata)
    
    def _identify_sections(self, text: str) -> List[Dict]:
        """
        Identify sections in academic paper
        Looks for common section headers
        
        Returns:
            List of sections with title and text
        """
        common_headers = [
            'abstract', 'introduction', 'background', 'related work',
            'methodology', 'methods', 'approach', 'system', 'model',
            'experiments', 'results', 'evaluation', 'discussion',
            'conclusion', 'future work', 'references', 'acknowledgments'
        ]
        
        sections = []
        lines = text.split('\n')
        
        current_section = {'title': 'Introduction', 'text': ''}
        
        for line in lines:
            line_lower = line.strip().lower()
            
            # Check if line is a section header
            is_header = False
            for header in common_headers:
                if line_lower.startswith(header) and len(line.strip()) < 50:
                    is_header = True
                    # Save previous section
                    if current_section['text'].strip():
                        sections.append(current_section)
                    # Start new section
                    current_section = {
                        'title': line.strip(),
                        'text': ''
                    }
                    break
            
            if not is_header:
                current_section['text'] += line + '\n'
        
        # Add last section
        if current_section['text'].strip():
            sections.append(current_section)
        
        return sections if len(sections) > 1 else [{'title': 'Full Text', 'text': text}]
    
    def visualize_chunks(self, chunks: List[Dict], max_display: int = 5):
        """
        Display chunk information for debugging
        
        Args:
            chunks: List of chunks to visualize
            max_display: Maximum number of chunks to display in detail
        """
        print(f"\n{'='*60}")
        print(f"CHUNK VISUALIZATION")
        print(f"{'='*60}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Total tokens: {sum(c['num_tokens'] for c in chunks)}")
        print(f"Avg tokens: {sum(c['num_tokens'] for c in chunks) / len(chunks):.0f}")
        print(f"Min tokens: {min(c['num_tokens'] for c in chunks)}")
        print(f"Max tokens: {max(c['num_tokens'] for c in chunks)}")
        
        print(f"\n📝 First {min(max_display, len(chunks))} chunks:\n")
        for i, chunk in enumerate(chunks[:max_display]):
            print(f"Chunk {i}:")
            print(f"  Tokens: {chunk['num_tokens']}")
            print(f"  Preview: {chunk['text'][:100]}...")
            if 'section_title' in chunk:
                print(f"  Section: {chunk['section_title']}")
            print()


def test_chunker():
    """Test the chunker"""
    # Sample text (you can replace with actual paper text)
    sample_text = """
    Abstract
    
    This paper introduces a new approach to natural language processing
    using transformer architectures. We demonstrate significant improvements
    over previous methods.
    
    Introduction
    
    Natural language processing has seen remarkable advances in recent years.
    The introduction of attention mechanisms has revolutionized the field.
    In this work, we propose a novel architecture that builds upon these
    foundations to achieve state-of-the-art results.
    
    Methods
    
    Our approach consists of three main components: the encoder, the decoder,
    and the attention mechanism. The encoder processes the input sequence,
    while the decoder generates the output. The attention mechanism allows
    the model to focus on relevant parts of the input.
    
    Results
    
    We evaluated our model on three benchmark datasets. On dataset A, we
    achieved an accuracy of 95.2%, outperforming the previous best by 3.1%.
    On dataset B, our F1 score was 0.89, and on dataset C, we reached 92.8%
    accuracy.
    
    Conclusion
    
    We have presented a new architecture for natural language processing.
    Our results demonstrate the effectiveness of our approach. Future work
    will explore applications to other domains.
    """ * 10  # Repeat to make it longer
    
    # Initialize chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    # Test basic chunking
    print("\n" + "="*60)
    print("TEST 1: Basic Chunking")
    print("="*60)
    chunks = chunker.chunk_text(sample_text)
    chunker.visualize_chunks(chunks, max_display=3)
    
    # Test context-aware chunking
    print("\n" + "="*60)
    print("TEST 2: Context-Aware Chunking")
    print("="*60)
    chunks_with_context = chunker.chunk_with_context(sample_text)
    chunker.visualize_chunks(chunks_with_context, max_display=3)


if __name__ == "__main__":
    test_chunker()