"""
Query Router Module
Intelligently routes user queries to appropriate generation methods
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class QueryType(Enum):
    """Types of queries the system can handle"""
    QA = "qa"                           # Standard question-answering
    COMPARISON = "comparison"            # Compare multiple items
    LITERATURE_REVIEW = "review"         # Generate literature review
    EXTRACTION = "extraction"            # Extract specific information
    DEFINITION = "definition"            # Define a term/concept
    SUMMARY = "summary"                  # Summarize papers/topics


class QueryRouter:
    """Routes queries to appropriate generation methods"""
    
    def __init__(self):
        """Initialize query router with patterns"""
        
        # Comparison patterns
        self.comparison_patterns = [
            r'\b(compare|comparison|versus|vs\.?)\b',
            r'\b(difference|differ|different)\b.*\b(between|from)\b',
            r'\b(similar|similarity)\b.*\b(to|with)\b',
            r'\b(contrast|contrasting)\b',
            r'\bwhich is better\b',
            r'\b(\w+)\s+(vs\.?|versus)\s+(\w+)\b',
            r'\b(\w+)\s+and\s+(\w+)\s+(comparison|difference)\b'
        ]
        
        # Literature review patterns
        self.review_patterns = [
            r'\b(review|overview|survey)\b',
            r'\b(summarize|summary)\s+(all|the)\b',
            r'\bliterature review\b',
            r'\bstate of (the )?art\b',
            r'\bcurrent research\b',
            r'\brecent (work|research|advances|developments)\b',
            r'\bwhat (has been|is) done\b',
            r'\bprogress in\b'
        ]
        
        # Extraction patterns
        self.extraction_patterns = [
            r'\bextract\b',
            r'\blist (all|the)\b',
            r'\bfind (all|the)\b.*\b(methods|techniques|approaches)\b',
            r'\bwhat (methods|techniques|approaches|algorithms)\b',
            r'\benumerate\b',
            r'\bidentify (all|the)\b'
        ]
        
        # Definition patterns
        self.definition_patterns = [
            r'^\s*what is\b',
            r'^\s*define\b',
            r'^\s*definition of\b',
            r'\bmeaning of\b',
            r'\bexplain\s+(\w+)\s*$',  # "Explain BERT"
            r'^\s*(\w+)\s*\??\s*$'      # Single word question
        ]
        
        # Summary patterns
        self.summary_patterns = [
            r'\bsummarize\b',
            r'\bsummary of\b',
            r'\bmain (points|findings|contributions)\b',
            r'\bkey (points|findings|takeaways)\b',
            r'\bin brief\b',
            r'\bin short\b'
        ]
        
        # Comparison keywords for extraction
        self.comparison_keywords = [
            'compare', 'comparison', 'versus', 'vs', 'vs.',
            'difference', 'different', 'differ',
            'contrast', 'similar', 'similarity',
            'better', 'worse'
        ]
        
        # Review keywords
        self.review_keywords = [
            'review', 'overview', 'survey', 'literature',
            'state of the art', 'recent', 'current',
            'progress', 'advances', 'developments'
        ]
    
    def route_query(self, query: str) -> Dict:
        """
        Route query to appropriate method
        
        Args:
            query: User's question
            
        Returns:
            Dict with routing information:
            {
                'query_type': QueryType,
                'method': str,  # Method name to call
                'params': dict,  # Extracted parameters
                'confidence': float  # Confidence score (0-1)
            }
        """
        query_lower = query.lower().strip()
        
        # Check each query type in order of specificity
        
        # 1. Check for comparison
        comparison_result = self._check_comparison(query, query_lower)
        if comparison_result['confidence'] > 0.7:
            return comparison_result
        
        # 2. Check for literature review
        review_result = self._check_review(query, query_lower)
        if review_result['confidence'] > 0.7:
            return review_result
        
        # 3. Check for extraction
        extraction_result = self._check_extraction(query, query_lower)
        if extraction_result['confidence'] > 0.7:
            return extraction_result
        
        # 4. Check for summary
        summary_result = self._check_summary(query, query_lower)
        if summary_result['confidence'] > 0.6:
            return summary_result
        
        # 5. Check for definition
        definition_result = self._check_definition(query, query_lower)
        if definition_result['confidence'] > 0.5:
            return definition_result
        
        # 6. Default to Q&A
        return {
            'query_type': QueryType.QA,
            'method': 'answer_question',
            'params': {'question': query},
            'confidence': 0.8
        }
    
    def _check_comparison(self, query: str, query_lower: str) -> Dict:
        """Check if query is a comparison request"""
        confidence = 0.0
        items = []
        
        # Check patterns
        for pattern in self.comparison_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.3
        
        # Extract items to compare
        items = self._extract_comparison_items(query)
        if len(items) >= 2:
            confidence += 0.5
        
        # Check for comparison keywords
        keyword_count = sum(1 for kw in self.comparison_keywords if kw in query_lower)
        confidence += min(keyword_count * 0.1, 0.3)
        
        confidence = min(confidence, 1.0)
        
        return {
            'query_type': QueryType.COMPARISON,
            'method': 'compare_papers',
            'params': {
                'items': items if items else [],
                'query': query
            },
            'confidence': confidence
        }
    
    def _check_review(self, query: str, query_lower: str) -> Dict:
        """Check if query is a literature review request"""
        confidence = 0.0
        
        # Check patterns
        for pattern in self.review_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.4
        
        # Check for review keywords
        keyword_count = sum(1 for kw in self.review_keywords if kw in query_lower)
        confidence += min(keyword_count * 0.15, 0.4)
        
        # Extract topic
        topic = self._extract_topic(query)
        
        confidence = min(confidence, 1.0)
        
        return {
            'query_type': QueryType.LITERATURE_REVIEW,
            'method': 'generate_literature_review',
            'params': {
                'topic': topic,
                'query': query
            },
            'confidence': confidence
        }
    
    def _check_extraction(self, query: str, query_lower: str) -> Dict:
        """Check if query is an extraction request"""
        confidence = 0.0
        
        # Check patterns
        for pattern in self.extraction_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.5
        
        confidence = min(confidence, 1.0)
        
        return {
            'query_type': QueryType.EXTRACTION,
            'method': 'answer_question',  # Use Q&A with extraction focus
            'params': {
                'question': query,
                'extract_mode': True
            },
            'confidence': confidence
        }
    
    def _check_summary(self, query: str, query_lower: str) -> Dict:
        """Check if query is a summary request"""
        confidence = 0.0
        
        # Check patterns
        for pattern in self.summary_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.4
        
        confidence = min(confidence, 1.0)
        
        return {
            'query_type': QueryType.SUMMARY,
            'method': 'answer_question',  # Use Q&A for summaries
            'params': {
                'question': query,
                'summary_mode': True
            },
            'confidence': confidence
        }
    
    def _check_definition(self, query: str, query_lower: str) -> Dict:
        """Check if query is asking for a definition"""
        confidence = 0.0
        
        # Check patterns
        for pattern in self.definition_patterns:
            if re.search(pattern, query_lower):
                confidence += 0.3
        
        # Short queries are often definitions
        word_count = len(query.split())
        if word_count <= 5:
            confidence += 0.2
        
        confidence = min(confidence, 1.0)
        
        return {
            'query_type': QueryType.DEFINITION,
            'method': 'answer_question',
            'params': {
                'question': query,
                'definition_mode': True
            },
            'confidence': confidence
        }
    
    def _extract_comparison_items(self, query: str) -> List[str]:
        """
        Extract items to compare from query
        
        Examples:
            "Compare BERT and GPT" -> ["BERT", "GPT"]
            "BERT vs GPT" -> ["BERT", "GPT"]
            "Difference between RNN and LSTM" -> ["RNN", "LSTM"]
        """
        items = []
        
        # Pattern 1: "X and Y"
        and_pattern = r'(\b[A-Z][A-Za-z0-9-]*\b)\s+and\s+(\b[A-Z][A-Za-z0-9-]*\b)'
        match = re.search(and_pattern, query)
        if match:
            items = [match.group(1), match.group(2)]
            return items
        
        # Pattern 2: "X vs Y" or "X versus Y"
        vs_pattern = r'(\b[A-Z][A-Za-z0-9-]*\b)\s+(?:vs\.?|versus)\s+(\b[A-Z][A-Za-z0-9-]*\b)'
        match = re.search(vs_pattern, query)
        if match:
            items = [match.group(1), match.group(2)]
            return items
        
        # Pattern 3: "between X and Y"
        between_pattern = r'between\s+(\b[A-Z][A-Za-z0-9-]*\b)\s+and\s+(\b[A-Z][A-Za-z0-9-]*\b)'
        match = re.search(between_pattern, query, re.IGNORECASE)
        if match:
            items = [match.group(1), match.group(2)]
            return items
        
        # Pattern 4: Extract all capitalized words (fallback)
        capitalized = re.findall(r'\b[A-Z][A-Za-z0-9-]+\b', query)
        if len(capitalized) >= 2:
            items = capitalized[:2]  # Take first two
        
        return items
    
    def _extract_topic(self, query: str) -> str:
        """Extract topic from review/summary request"""
        # Remove common keywords
        topic = query.lower()
        
        remove_words = [
            'review', 'overview', 'survey', 'summarize', 'summary',
            'literature', 'state of the art', 'recent', 'current',
            'give me', 'provide', 'show me', 'tell me about',
            'what is', 'what are', 'the', 'a', 'an', 'of', 'on', 'about'
        ]
        
        for word in remove_words:
            topic = re.sub(r'\b' + word + r'\b', '', topic, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        topic = ' '.join(topic.split())
        
        return topic.strip() or query  # Return original if nothing left
    
    def get_query_type_description(self, query_type: QueryType) -> str:
        """Get human-readable description of query type"""
        descriptions = {
            QueryType.QA: "Question & Answer",
            QueryType.COMPARISON: "Comparison",
            QueryType.LITERATURE_REVIEW: "Literature Review",
            QueryType.EXTRACTION: "Information Extraction",
            QueryType.DEFINITION: "Definition",
            QueryType.SUMMARY: "Summary"
        }
        return descriptions.get(query_type, "Unknown")


def test_query_router():
    """Test the query router with various queries"""
    
    router = QueryRouter()
    
    test_queries = [
        "What is BERT?",
        "Compare BERT and GPT",
        "BERT vs GPT",
        "What is the difference between RNN and LSTM?",
        "Review all papers on transformers",
        "Give me an overview of attention mechanisms",
        "Summarize the main findings",
        "List all methods used in the paper",
        "What are the key contributions?",
        "How does self-attention work?",
        "Recent advances in NLP",
        "State of the art in computer vision"
    ]
    
    print("\n" + "="*80)
    print("TESTING QUERY ROUTER")
    print("="*80)
    
    for query in test_queries:
        result = router.route_query(query)
        
        print(f"\n📝 Query: {query}")
        print(f"   Type: {router.get_query_type_description(result['query_type'])}")
        print(f"   Method: {result['method']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Params: {result['params']}")
        print("   " + "-"*70)
    
    print("\n" + "="*80)
    print("✅ Test complete!")
    print("="*80)


if __name__ == "__main__":
    test_query_router()
