"""
Prompt Templates Module
High-quality prompts for different generation tasks
"""

from typing import List, Dict


class PromptTemplates:
    """Collection of prompt templates for RAG system"""
    
    # System prompts for different agent roles
    SYSTEM_PROMPTS = {
        'analyzer': """You are a research paper analyzer. Your job is to extract key information from paper excerpts.

Focus on:
- Main findings and conclusions
- Methodologies used
- Key technical terms and concepts
- Important results and metrics

Be concise and factual. Only extract information present in the text.""",
        
        'synthesizer': """You are a research synthesis expert. Your job is to combine information from multiple sources into a coherent answer.

Guidelines:
- Synthesize information logically
- Remove redundancy
- Maintain accuracy
- Organize ideas clearly
- Connect related concepts

Never add information not present in the sources.""",
        
        'citation_agent': """You are a citation expert. Your job is to add proper citations to text.

Rules:
- Use format: [Paper Title, Year]
- Cite specific claims, not general statements
- One citation per specific fact
- Place citations at end of relevant sentence

Example: "BERT uses bidirectional attention [BERT: Pre-training of Deep Bidirectional Transformers, 2018].\"
""",
        
        'critic': """You are a quality critic. Your job is to improve and refine answers.

Check for:
- Accuracy (claims match sources)
- Clarity (easy to understand)
- Completeness (answers the question)
- Citations (properly formatted)
- Coherence (logical flow)

Suggest specific improvements."""
    }
    
    @staticmethod
    def build_qa_prompt(question: str, context: str, citations: List[Dict] = None) -> str:
        """
        Build Question-Answering prompt
        
        Args:
            question: User question
            context: Retrieved context from papers
            citations: Optional citation information
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following research paper excerpts, answer the question.

RESEARCH EXCERPTS:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer ONLY based on the provided excerpts
2. Be comprehensive but concise
3. Include specific details and findings
4. If excerpts don't fully answer the question, say so clearly
5. Use citations in format [Paper Title, Year] after relevant claims

ANSWER:"""
        
        return prompt
    
    @staticmethod
    def build_extraction_prompt(text: str, extract_type: str = "key_findings") -> str:
        """
        Build information extraction prompt
        
        Args:
            text: Source text
            extract_type: What to extract ('key_findings', 'methodology', 'results')
            
        Returns:
            Formatted prompt
        """
        extraction_guides = {
            'key_findings': "Extract the main findings and conclusions.",
            'methodology': "Extract the methodology, approach, or techniques used.",
            'results': "Extract experimental results, metrics, and outcomes.",
            'contributions': "Extract the key contributions and innovations."
        }
        
        guide = extraction_guides.get(extract_type, "Extract key information.")
        
        prompt = f"""From the following text, {guide}

TEXT:
{text}

EXTRACTED INFORMATION:"""
        
        return prompt
    
    @staticmethod
    def build_synthesis_prompt(
        question: str,
        extractions: List[Dict],
        max_length: int = 300
    ) -> str:
        """
        Build synthesis prompt from multiple extractions
        
        Args:
            question: Original question
            extractions: List of extracted information from different sources
            max_length: Max words in response
            
        Returns:
            Formatted prompt
        """
        # Format extractions
        formatted_extractions = []
        for i, ext in enumerate(extractions, 1):
            source = ext.get('source', f'Source {i}')
            content = ext.get('content', '')
            formatted_extractions.append(f"Source {i} ({source}):\n{content}")
        
        extractions_text = "\n\n".join(formatted_extractions)
        
        prompt = f"""Synthesize the following information to answer the question.

QUESTION: {question}

INFORMATION FROM SOURCES:
{extractions_text}

INSTRUCTIONS:
1. Combine information from all sources
2. Create a coherent, unified answer
3. Remove redundancy
4. Keep answer under {max_length} words
5. Maintain accuracy - only use provided information

SYNTHESIZED ANSWER:"""
        
        return prompt
    
    @staticmethod
    def build_citation_prompt(text: str, sources: List[Dict]) -> str:
        """
        Build prompt to add citations to text
        
        Args:
            text: Text needing citations
            sources: List of source documents with metadata
            
        Returns:
            Formatted prompt
        """
        # Format sources
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', 'Unknown')
            year = source.get('year', 'n.d.')
            author = source.get('author', 'Unknown')
            formatted_sources.append(f"{i}. {title} ({author}, {year})")
        
        sources_text = "\n".join(formatted_sources)
        
        prompt = f"""Add proper citations to the following text using the provided sources.

TEXT TO CITE:
{text}

AVAILABLE SOURCES:
{sources_text}

INSTRUCTIONS:
1. Add citations after specific claims
2. Use format: [Title, Year]
3. Only cite claims that come from sources
4. Don't cite general knowledge
5. One citation per specific fact

TEXT WITH CITATIONS:"""
        
        return prompt
    
    @staticmethod
    def build_critique_prompt(question: str, answer: str, sources: str) -> str:
        """
        Build prompt for answer critique
        
        Args:
            question: Original question
            answer: Generated answer
            sources: Source context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Review and improve the following answer.

QUESTION: {question}

CURRENT ANSWER:
{answer}

SOURCES:
{sources}

REVIEW CHECKLIST:
1. Accuracy: Does answer match sources?
2. Completeness: Fully answers question?
3. Clarity: Easy to understand?
4. Citations: Properly formatted and placed?
5. Coherence: Logical flow?

PROVIDE:
1. Issues found (if any)
2. Improved version of answer

REVIEW:"""
        
        return prompt
    
    @staticmethod
    def build_literature_review_prompt(
        topic: str,
        papers_summary: List[Dict],
        max_length: int = 500
    ) -> str:
        """
        Build prompt for literature review generation
        
        Args:
            topic: Research topic
            papers_summary: Summaries of relevant papers
            max_length: Max words
            
        Returns:
            Formatted prompt
        """
        # Format paper summaries
        formatted_papers = []
        for i, paper in enumerate(papers_summary, 1):
            title = paper.get('title', 'Unknown')
            summary = paper.get('summary', '')
            formatted_papers.append(f"{i}. {title}\n{summary}")
        
        papers_text = "\n\n".join(formatted_papers)
        
        prompt = f"""Generate a literature review on the topic: {topic}

PAPERS:
{papers_text}

INSTRUCTIONS:
1. Provide an overview of the research area
2. Discuss main approaches and methodologies
3. Compare and contrast different papers
4. Identify trends and gaps
5. Keep under {max_length} words
6. Include citations

LITERATURE REVIEW:"""
        
        return prompt
    
    @staticmethod
    def build_comparison_prompt(
        items: List[str],
        context: str,
        aspects: List[str] = None
    ) -> str:
        """
        Build prompt for comparing multiple items
        
        Args:
            items: Items to compare (e.g., ["BERT", "GPT"])
            context: Relevant context about items
            aspects: Specific aspects to compare
            
        Returns:
            Formatted prompt
        """
        items_text = " vs ".join(items)
        
        aspects_text = ""
        if aspects:
            aspects_text = f"\nCompare on these aspects: {', '.join(aspects)}"
        
        prompt = f"""Compare: {items_text}

CONTEXT:
{context}
{aspects_text}

INSTRUCTIONS:
1. Compare similarities and differences
2. Use specific details from context
3. Organize comparison clearly
4. Include citations for specific claims

COMPARISON:"""
        
        return prompt


def test_prompt_templates():
    """Test prompt templates"""
    
    print("\n" + "="*80)
    print("TESTING PROMPT TEMPLATES")
    print("="*80)
    
    # Test QA prompt
    print("\n1️⃣ Question-Answering Prompt:")
    print("─" * 80)
    qa_prompt = PromptTemplates.build_qa_prompt(
        question="What is attention mechanism?",
        context="[Source 1]: The attention mechanism computes weighted sums...\n[Source 2]: Attention allows models to focus..."
    )
    print(qa_prompt)
    
    # Test extraction prompt
    print("\n2️⃣ Extraction Prompt:")
    print("─" * 80)
    ext_prompt = PromptTemplates.build_extraction_prompt(
        text="This paper introduces BERT, a bidirectional transformer...",
        extract_type="key_findings"
    )
    print(ext_prompt)
    
    # Test synthesis prompt
    print("\n3️⃣ Synthesis Prompt:")
    print("─" * 80)
    synth_prompt = PromptTemplates.build_synthesis_prompt(
        question="How does BERT work?",
        extractions=[
            {'source': 'BERT Paper', 'content': 'BERT uses masked language modeling...'},
            {'source': 'Tutorial', 'content': 'BERT is pretrained on large corpora...'}
        ]
    )
    print(synth_prompt)
    
    # Test system prompts
    print("\n4️⃣ System Prompts:")
    print("─" * 80)
    for role, prompt in PromptTemplates.SYSTEM_PROMPTS.items():
        print(f"\n{role.upper()}:")
        print(prompt[:150] + "...")


if __name__ == "__main__":
    test_prompt_templates()