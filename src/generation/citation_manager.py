"""
Citation Manager Module
Handle citation formatting and tracking
"""

import re
from typing import List, Dict, Set


class CitationManager:
    """Manage citations in generated text"""
    
    def __init__(self, citation_style: str = "inline"):
        """
        Initialize citation manager
        
        Args:
            citation_style: Citation style ('inline', 'numbered', 'apa')
        """
        self.citation_style = citation_style
        self.citations_used = []
        
        print(f"📚 Citation Manager initialized (style: {citation_style})")
    
    def format_citation(self, source: Dict) -> str:
        """
        Format a single citation
        
        Args:
            source: Source metadata dict
            
        Returns:
            Formatted citation string
        """
        title = source.get('title', 'Unknown')
        year = source.get('year', 'n.d.')
        author = source.get('author', 'Unknown')
        
        if self.citation_style == 'inline':
            # [Title, Year]
            return f"[{title}, {year}]"
        
        elif self.citation_style == 'numbered':
            # [1]
            if source not in self.citations_used:
                self.citations_used.append(source)
            idx = self.citations_used.index(source) + 1
            return f"[{idx}]"
        
        elif self.citation_style == 'apa':
            # (Author, Year)
            return f"({author}, {year})"
        
        else:
            return f"[{title}, {year}]"
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract existing citations from text
        
        Args:
            text: Text with citations
            
        Returns:
            List of citation strings
        """
        # Match citations in formats: [X, YYYY], [N], (X, YYYY)
        patterns = [
            r'\[([^\]]+, \d{4})\]',  # [Title, 2023]
            r'\[(\d+)\]',             # [1]
            r'\(([^)]+, \d{4})\)'    # (Author, 2023)
        ]
        
        citations = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return citations
    
    def add_citations_to_text(
        self,
        text: str,
        sources: List[Dict],
        claims: List[str] = None
    ) -> str:
        """
        Add citations to text
        
        Args:
            text: Text needing citations
            sources: Available sources
            claims: Specific claims to cite (optional)
            
        Returns:
            Text with citations
        """
        # Simple approach: add citations at sentence ends
        sentences = text.split('. ')
        cited_sentences = []
        
        for i, sentence in enumerate(sentences):
            # Add citation to sentences with specific claims
            if i < len(sources):
                citation = self.format_citation(sources[i])
                cited_sentences.append(f"{sentence} {citation}")
            else:
                cited_sentences.append(sentence)
        
        return '. '.join(cited_sentences)
    
    def generate_bibliography(self, sources: List[Dict]) -> str:
        """
        Generate bibliography from sources
        
        Args:
            sources: List of source metadata
            
        Returns:
            Formatted bibliography
        """
        if not sources:
            return ""
        
        bibliography = ["## References\n"]
        
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown')
            author = source.get('author', 'Unknown')
            year = source.get('year', 'n.d.')
            
            if self.citation_style == 'numbered':
                entry = f"{i}. {author} ({year}). {title}."
            elif self.citation_style == 'apa':
                entry = f"{author} ({year}). {title}."
            else:
                entry = f"• {author} ({year}). {title}."
            
            bibliography.append(entry)
        
        return "\n".join(bibliography)
    
    def validate_citations(self, text: str, sources: List[Dict]) -> Dict:
        """
        Validate citations in text
        
        Args:
            text: Text with citations
            sources: Available sources
            
        Returns:
            Validation report
        """
        extracted = self.extract_citations(text)
        
        # Check for uncited sources
        cited_titles = set()
        for cit in extracted:
            # Extract title from citation
            if ',' in cit:
                title = cit.split(',')[0].strip('[]() ')
                cited_titles.add(title.lower())
        
        available_titles = {s.get('title', '').lower() for s in sources}
        
        uncited = available_titles - cited_titles
        
        return {
            'total_citations': len(extracted),
            'unique_sources_cited': len(cited_titles),
            'available_sources': len(sources),
            'uncited_sources': list(uncited),
            'citation_coverage': len(cited_titles) / len(sources) if sources else 0
        }
    
    def remove_citations(self, text: str) -> str:
        """
        Remove citations from text
        
        Args:
            text: Text with citations
            
        Returns:
            Text without citations
        """
        # Remove all citation patterns
        patterns = [
            r'\[([^\]]+, \d{4})\]',
            r'\[(\d+)\]',
            r'\(([^)]+, \d{4})\)'
        ]
        
        clean_text = text
        for pattern in patterns:
            clean_text = re.sub(pattern, '', clean_text)
        
        # Clean up extra spaces
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    def merge_duplicate_citations(self, text: str) -> str:
        """
        Merge duplicate consecutive citations
        
        Args:
            text: Text with citations
            
        Returns:
            Text with merged citations
        """
        # Find patterns like [A, 2023] [B, 2024] and merge to [A, 2023; B, 2024]
        pattern = r'\[([^\]]+)\]\s*\[([^\]]+)\]'
        
        def merge_match(match):
            return f"[{match.group(1)}; {match.group(2)}]"
        
        merged_text = re.sub(pattern, merge_match, text)
        
        return merged_text


def test_citation_manager():
    """Test citation manager"""
    
    print("\n" + "="*80)
    print("TESTING CITATION MANAGER")
    print("="*80)
    
    # Test sources
    sources = [
        {'title': 'Attention Is All You Need', 'author': 'Vaswani et al.', 'year': '2017'},
        {'title': 'BERT Paper', 'author': 'Devlin et al.', 'year': '2018'},
        {'title': 'GPT-3 Paper', 'author': 'Brown et al.', 'year': '2020'}
    ]
    
    # Test different citation styles
    for style in ['inline', 'numbered', 'apa']:
        print(f"\n{'─'*80}")
        print(f"Style: {style}")
        print(f"{'─'*80}")
        
        manager = CitationManager(citation_style=style)
        
        # Format citations
        print("\n1. Format citations:")
        for source in sources:
            citation = manager.format_citation(source)
            print(f"   {citation}")
        
        # Generate bibliography
        print("\n2. Bibliography:")
        bib = manager.generate_bibliography(sources)
        print(bib)
    
    # Test citation extraction
    print(f"\n{'─'*80}")
    print("Extract Citations")
    print(f"{'─'*80}")
    
    manager = CitationManager()
    test_text = "Transformers use attention [Attention Is All You Need, 2017]. BERT is bidirectional [BERT Paper, 2018]."
    
    extracted = manager.extract_citations(test_text)
    print(f"\nText: {test_text}")
    print(f"\nExtracted citations: {extracted}")
    
    # Test validation
    print(f"\n{'─'*80}")
    print("Validate Citations")
    print(f"{'─'*80}")
    
    validation = manager.validate_citations(test_text, sources)
    print("\nValidation report:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    # Test remove citations
    print(f"\n{'─'*80}")
    print("Remove Citations")
    print(f"{'─'*80}")
    
    clean_text = manager.remove_citations(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Cleaned: {clean_text}")


if __name__ == "__main__":
    test_citation_manager()