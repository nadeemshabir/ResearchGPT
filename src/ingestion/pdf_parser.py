"""
PDF Parser Module
Extracts text and metadata from academic PDFs
"""

import os
from pathlib import Path
from typing import Dict, Optional
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from tqdm import tqdm


class PDFParser:
    """Parse PDF files and extract text with metadata"""
    
    def __init__(self, method: str = "pymupdf"):
        """
        Initialize PDF parser
        
        Args:
            method: Parsing method ('pypdf2', 'pdfplumber', 'pymupdf')
                   'pymupdf' is recommended for best quality. if we dont mention anything, it uses 'pymupdf'
        """
        self.method = method
        
    def extract_text(self, pdf_path: str) -> Dict[str, any]:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text, metadata, and stats
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        print(f"📄 Processing: {pdf_path.name}")
        print(f"   Method: {self.method}")
        
        # Choose parsing method
        if self.method == "pypdf2":
            result = self._extract_pypdf2(pdf_path) #here call is happeing to _extract_pypdf2
        elif self.method == "pdfplumber":
            result = self._extract_pdfplumber(pdf_path)
        elif self.method == "pymupdf":
            result = self._extract_pymupdf(pdf_path)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Add file metadata
        result['file_name'] = pdf_path.name
        result['file_path'] = str(pdf_path.absolute())
        result['file_size_mb'] = pdf_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Extracted {len(result['text'])} characters from {result['num_pages']} pages")
        
        return result
    
    def _extract_pypdf2(self, pdf_path: Path) -> Dict:
        """Extract using PyPDF2 (basic, fast)"""
        text = ""
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            # Extract metadata
            metadata = reader.metadata
            
            # Extract text from all pages
            for page_num in tqdm(range(num_pages), desc="Extracting pages"):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return {
            'text': text,
            'num_pages': num_pages,
            'metadata': {
                'title': metadata.get('/Title', ''),
                'author': metadata.get('/Author', ''),
                'subject': metadata.get('/Subject', ''),
                'creator': metadata.get('/Creator', '')
            } if metadata else {}
        }
    
    def _extract_pdfplumber(self, pdf_path: Path) -> Dict:
        """Extract using pdfplumber (better quality, slower)"""
        text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            metadata = pdf.metadata
            
            # Extract text from all pages
            for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        
        return {
            'text': text,
            'num_pages': num_pages,
            'metadata': metadata or {}
        }
    
    def _extract_pymupdf(self, pdf_path: Path) -> Dict:
        """Extract using PyMuPDF/fitz (best quality, recommended)"""
        text = ""
        
        # Open PDF
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        
        # Extract metadata
        metadata = doc.metadata
        
        # Extract text from all pages
        for page_num in tqdm(range(num_pages), desc="Extracting pages"):
            page = doc[page_num]
            text += page.get_text() + "\n\n"
        
        doc.close()
        
        return {
            'text': text,
            'num_pages': num_pages,
            'metadata': {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
            }
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace('\x00', '')  # Null characters
        text = text.replace('\uf0b7', '')  # Bullet points
        
        # Fix common spacing issues
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        
        return text


def test_parser():
    """Test the PDF parser"""
    # Test with a sample PDF
    test_pdf = "data/raw/test_paper.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"⚠️ Test PDF not found: {test_pdf}")
        print("   Please place a test PDF in data/raw/test_paper.pdf")
        return
    
    # Test all methods
    for method in ['pypdf2', 'pdfplumber', 'pymupdf']:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print('='*60)
        
        parser = PDFParser(method=method)
        result = parser.extract_text(test_pdf)
        
        print(f"\n📊 Results:")
        print(f"   Pages: {result['num_pages']}")
        print(f"   Characters: {len(result['text'])}")
        print(f"   File size: {result['file_size_mb']:.2f} MB")
        print(f"\n📝 First 200 characters:")
        print(result['text'][:200])


if __name__ == "__main__":
    test_parser()