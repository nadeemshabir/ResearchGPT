"""
Query Processing Module
 is designed to "upgrade" the user's raw input before it gets sent to the search engine. Its goal is to make the search more robust and more likely to find relevant results.
Clean, expand, and improve search queries
"""

import re
from typing import List, Dict, Optional


class QueryProcessor: 
    """Process and enhance search queries"""
    
    def __init__(self):
        """Initialize query processor"""
        print("🔍 Initializing Query Processor...")
        
        # Common stopwords to remove
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with'
        }
        
        # Common academic synonyms
        self.synonyms = {
            'method': ['approach', 'technique', 'methodology'],
            'model': ['architecture', 'framework', 'system'],
            'result': ['outcome', 'finding', 'conclusion'],
            'improve': ['enhance', 'optimize', 'boost'],
            'use': ['utilize', 'employ', 'apply'],
            'show': ['demonstrate', 'indicate', 'reveal'],
            'propose': ['introduce', 'present', 'suggest'],
        }
        
        # Question word patterns
        self.question_patterns = {
            'what': 'definition OR explanation',
            'how': 'method OR process OR mechanism',
            'why': 'reason OR cause OR rationale',
            'when': 'time OR timeline OR history',
            'where': 'location OR context OR application'
        }
        
        print("✅ Query processor ready")
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query
        """
        if not query:
            return ""
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove special characters (keep alphanumeric, spaces, hyphens)
        query = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', query)
        
        # Normalize whitespace again
        query = ' '.join(query.split())
        
        return query.strip()
    
    def remove_stopwords(self, query: str, preserve_phrases: bool = True) -> str:
        """
        Remove common stopwords from query
        
        Args:
            query: Input query
            preserve_phrases: Keep stopwords in quoted phrases
            
        Returns:
            Query without stopwords
        """
        if preserve_phrases and '"' in query:
            # Extract phrases
            phrases = re.findall(r'"([^"]*)"', query)
            # Remove phrases temporarily
            temp_query = re.sub(r'"[^"]*"', 'PHRASE_PLACEHOLDER', query)
            # Remove stopwords from non-phrase parts
            words = temp_query.split()
            filtered = [w for w in words if w.lower() not in self.stopwords or w == 'PHRASE_PLACEHOLDER']
            result = ' '.join(filtered)
            # Restore phrases
            for phrase in phrases:
                result = result.replace('PHRASE_PLACEHOLDER', f'"{phrase}"', 1)
            return result
        else:
            # Remove all stopwords
            words = query.split()
            filtered = [w for w in words if w.lower() not in self.stopwords]
            return ' '.join(filtered)
    
    def expand_query(
        self,
        query: str,
        use_synonyms: bool = True,
        max_expansions: int = 2
    ) -> str:
        """
        Expand query with synonyms
        
        Args:
            query: Original query
            use_synonyms: Whether to add synonyms
            max_expansions: Max synonyms per word
            
        Returns:
            Expanded query
        """
        if not use_synonyms:
            return query
        
        words = query.lower().split()
        expanded = []
        
        for word in words:
            expanded.append(word)
            
            # Check if word has synonyms
            if word in self.synonyms:
                # Add synonyms (limited)
                for syn in self.synonyms[word][:max_expansions]:
                    expanded.append(syn)
        
        return ' '.join(expanded)
    
    def extract_key_terms(self, query: str, top_n: int = 5) -> List[str]:
        """
        Extract key terms from query
        
        Args:
            query: Input query
            top_n: Number of terms to extract
            
        Returns:
            List of key terms
        """
        # Clean query
        clean = self.clean_query(query)
        
        # Remove stopwords
        words = [w for w in clean.lower().split() if w not in self.stopwords]
        
        # Simple heuristic: longer words are often more important
        words.sort(key=len, reverse=True)
        
        return words[:top_n]
    
    def generate_query_variations(
        self,
        query: str,
        num_variations: int = 3
    ) -> List[str]:
        """
        Generate multiple query variations for multi-query search
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        variations = [query]  # Original
        
        # Variation 1: Remove stopwords
        if num_variations >= 2:
            no_stopwords = self.remove_stopwords(query)
            if no_stopwords != query:
                variations.append(no_stopwords)
        
        # Variation 2: Expand with synonyms
        if num_variations >= 3:
            expanded = self.expand_query(query)
            if expanded != query:
                variations.append(expanded)
        
        # Variation 3: Key terms only
        if num_variations >= 4:
            key_terms = ' '.join(self.extract_key_terms(query, top_n=3))
            if key_terms and key_terms != query:
                variations.append(key_terms)
        
        return variations[:num_variations]
    
    def enhance_question(self, query: str) -> str:
        """
        Enhance question-based queries
        
        Args:
            query: Question query
            
        Returns:
            Enhanced query
        """
        query_lower = query.lower()
        
        # Check for question words
        for q_word, expansion in self.question_patterns.items():
            if query_lower.startswith(q_word):
                # Remove question word and add helpful terms
                rest_of_query = query[len(q_word):].strip()
                return f"{rest_of_query} {expansion}"
        
        return query
    
    def detect_query_intent(self, query: str) -> Dict:
        """
        Detect the intent/type of query
        
        Args:
            query: Input query
            
        Returns:
            Dictionary with intent information
        """
        query_lower = query.lower()
        
        intent = {
            'type': 'general',
            'is_question': False,
            'is_definition': False,
            'is_comparison': False,
            'is_howto': False,
            'has_technical_terms': False
        }
        
        # Question detection
        if any(query_lower.startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who']):
            intent['is_question'] = True
            intent['type'] = 'question'
        
        # Definition query
        if any(word in query_lower for word in ['what is', 'define', 'definition of', 'explain']):
            intent['is_definition'] = True
            intent['type'] = 'definition'
        
        # Comparison query
        if any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'better']):
            intent['is_comparison'] = True
            intent['type'] = 'comparison'
        
        # How-to query
        if query_lower.startswith('how to') or 'step' in query_lower or 'process' in query_lower:
            intent['is_howto'] = True
            intent['type'] = 'howto'
        
        # Technical terms
        technical_indicators = ['algorithm', 'model', 'architecture', 'function', 'method']
        if any(term in query_lower for term in technical_indicators):
            intent['has_technical_terms'] = True
        
        return intent
    
    def process_query(
        self,
        query: str,
        remove_stops: bool = False,
        expand: bool = True,
        enhance_questions: bool = True
    ) -> Dict:
        """
        Complete query processing pipeline
        
        Args:
            query: Raw query
            remove_stops: Remove stopwords
            expand: Expand with synonyms
            enhance_questions: Enhance question queries
            
        Returns:
            Dictionary with processed query and metadata
        """
        print(f"\n🔍 Processing query: '{query}'")
        
        # Step 1: Clean
        cleaned = self.clean_query(query)
        print(f"   1️⃣ Cleaned: '{cleaned}'")
        
        # Step 2: Detect intent
        intent = self.detect_query_intent(cleaned)
        print(f"   2️⃣ Intent: {intent['type']}")
        
        # Step 3: Remove stopwords (optional)
        if remove_stops:
            cleaned = self.remove_stopwords(cleaned)
            print(f"   3️⃣ Removed stopwords: '{cleaned}'")
        
        # Step 4: Enhance questions
        if enhance_questions and intent['is_question']:
            cleaned = self.enhance_question(cleaned)
            print(f"   4️⃣ Enhanced: '{cleaned}'")
        
        # Step 5: Expand (optional)
        expanded = cleaned
        if expand:
            expanded = self.expand_query(cleaned)
            if expanded != cleaned:
                print(f"   5️⃣ Expanded: '{expanded}'")
        
        # Step 6: Generate variations
        variations = self.generate_query_variations(cleaned, num_variations=3)
        print(f"   6️⃣ Generated {len(variations)} variations")
        
        # Step 7: Extract key terms
        key_terms = self.extract_key_terms(cleaned)
        print(f"   7️⃣ Key terms: {key_terms}")
        
        result = {
            'original': query,
            'cleaned': cleaned,
            'expanded': expanded,
            'variations': variations,
            'key_terms': key_terms,
            'intent': intent
        }
        
        print("✅ Query processing complete")
        
        return result


def test_query_processor():
    """Test query processing"""
    
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "What is attention mechanism?",
        "How does BERT work?",
        "Compare transformer and RNN architectures",
        "the quick brown fox jumps over the lazy dog",
        "explain the method used in this paper",
        "What are the results of the experiment?"
    ]
    
    print("\n" + "="*80)
    print("TESTING QUERY PROCESSOR")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'─'*80}")
        print(f"ORIGINAL: {query}")
        print(f"{'─'*80}")
        
        # Process query
        result = processor.process_query(query, remove_stops=True, expand=True)
        
        print(f"\nProcessing Results:")
        print(f"  Cleaned: {result['cleaned']}")
        print(f"  Expanded: {result['expanded']}")
        print(f"  Intent: {result['intent']['type']}")
        print(f"  Key Terms: {result['key_terms']}")
        print(f"\n  Variations:")
        for i, var in enumerate(result['variations'], 1):
            print(f"    {i}. {var}")
        print(f"{'─'*80}")
    
    # Test specific functions
    print("\n" + "="*80)
    print("TESTING SPECIFIC FUNCTIONS")
    print("="*80)
    
    # Stopword removal
    print("\n1. Stopword Removal:")
    test = "the attention mechanism is used in transformers"
    print(f"   Original: {test}")
    print(f"   Filtered: {processor.remove_stopwords(test)}")
    
    # Query expansion
    print("\n2. Query Expansion:")
    test = "method to improve model"
    print(f"   Original: {test}")
    print(f"   Expanded: {processor.expand_query(test)}")
    
    # Intent detection
    print("\n3. Intent Detection:")
    test_intents = [
        "What is BERT?",
        "Compare BERT and GPT",
        "How to train transformer?",
        "architecture of neural networks"
    ]
    for test in test_intents:
        intent = processor.detect_query_intent(test)
        print(f"   '{test}' → {intent['type']}")


if __name__ == "__main__":
    test_query_processor()