"""
Complete Answer Generator
Integrates retrieval and generation into end-to-end system
"""

import time
from typing import Dict, List, Optional

from ..retrieval.retrieval_system import RetrievalSystem
from .llm_client import LLMClient
from .agents import AgentOrchestrator
from .citation_manager import CitationManager
from .prompt_templates import PromptTemplates
from .query_router import QueryRouter


class AnswerGenerator:
    """
    Complete end-to-end answer generation system
    Combines retrieval and generation
    """
    
    def __init__(
        self,
        use_multi_agent: bool = True,
        use_citations: bool = True,
        llm_provider: str = "groq",
        llm_model: str = None,
        use_smart_routing: bool = True
    ):
        """
        Initialize answer generator
        
        Args:
            use_multi_agent: Use multi-agent system (vs simple generation)
            use_citations: Add citations to answers
            llm_provider: LLM provider ('groq', 'openai', 'gemini')
            llm_model: Specific model name
            use_smart_routing: Use smart query routing
        """
        print("\n" + "="*80)
        print("🚀 INITIALIZING RESEARCHGPT COMPLETE SYSTEM")
        print("="*80)
        
        self.use_multi_agent = use_multi_agent
        self.use_citations = use_citations
        self.use_smart_routing = use_smart_routing
        
        # Initialize Query Router
        if use_smart_routing:
            print("\n1️⃣ Initializing Smart Query Router...")
            self.query_router = QueryRouter()
            print("   ✅ Query router ready")
        else:
            self.query_router = None
        
        # Initialize Retrieval System
        print("\n2️⃣ Initializing Retrieval System...")
        self.retrieval_system = RetrievalSystem(
            use_reranking=True,
            use_query_processing=True
        )
        
        # Initialize LLM Client
        print("\n3️⃣ Initializing LLM Client...")
        self.llm_client = LLMClient(
            provider=llm_provider,
            model=llm_model
        )
        
        # Initialize Generation Components
        if use_multi_agent:
            print("\n4️⃣ Initializing Multi-Agent System...")
            self.agent_orchestrator = AgentOrchestrator(self.llm_client)
        else:
            self.agent_orchestrator = None
        
        if use_citations:
            print("\n5️⃣ Initializing Citation Manager...")
            self.citation_manager = CitationManager(citation_style='inline')
        else:
            self.citation_manager = None
        
        print("\n" + "="*80)
        print("✅ RESEARCHGPT SYSTEM READY!")
        print("="*80)
    
    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        max_context_tokens: int = 4000,
        include_sources: bool = True
    ) -> Dict:
        """
        Complete question-answering pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            max_context_tokens: Max tokens for LLM context
            include_sources: Include source chunks in response
            
        Returns:
            Complete response with answer, sources, metadata
        """
        start_time = time.time()
        
        print(f"\n{'='*80}")
        print(f"QUESTION: {question}")
        print(f"{'='*80}")
        
        # Step 1: Retrieve relevant chunks
        print("\n📚 Step 1: Retrieving relevant papers...")
        retrieval_result = self.retrieval_system.get_relevant_chunks(
            query=question,
            max_tokens=max_context_tokens,
            min_score=0.5 if self.retrieval_system.use_reranking else 0.6
        )
        
        chunks = retrieval_result['chunks']
        context = retrieval_result['context']
        
        if not chunks:
            return {
                'answer': "I couldn't find relevant information in your papers to answer this question.",
                'sources': [],
                'metadata': {
                    'question': question,
                    'num_sources': 0,
                    'processing_time': time.time() - start_time,
                    'error': 'No relevant chunks found'
                }
            }
        
        print(f"   ✅ Retrieved {len(chunks)} relevant chunks ({retrieval_result['metadata']['total_tokens']} tokens)")
        
        # Step 2: Generate answer
        print("\n🤖 Step 2: Generating answer...")
        
        if self.use_multi_agent:
            # Multi-agent approach
            generation_result = self.agent_orchestrator.generate_answer(
                query=question,
                chunks=chunks,
                use_citations=self.use_citations,
                use_critique=True
            )
            answer = generation_result['answer']
        else:
            # Simple approach (faster)
            prompt = PromptTemplates.build_qa_prompt(question, context)
            answer = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You are a helpful research assistant.",
                temperature=0.5
            )
        
        print(f"   ✅ Generated answer ({len(answer)} chars)")
        
        # Step 3: Format response
        processing_time = time.time() - start_time
        
        response = {
            'answer': answer,
            'question': question,
            'sources': self._format_sources(chunks) if include_sources else [],
            'metadata': {
                'num_sources': len(chunks),
                'processing_time': round(processing_time, 2),
                'total_tokens': retrieval_result['metadata']['total_tokens'],
                'retrieval_method': 'hybrid + rerank',
                'generation_method': 'multi-agent' if self.use_multi_agent else 'simple',
                'used_citations': self.use_citations
            }
        }
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETE (took {processing_time:.2f}s)")
        print(f"{'='*80}\n")
        
        return response
    
    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        """Format source chunks for response"""
        sources = []
        
        seen_papers = set()
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            paper_id = metadata.get('paper_id', 'unknown')
            
            # Only include each paper once
            if paper_id not in seen_papers:
                sources.append({
                    'title': metadata.get('title', 'Unknown'),
                    'author': metadata.get('author', 'Unknown'),
                    'chunk_preview': chunk.get('text', '')[:200] + "...",
                    'relevance_score': chunk.get('rerank_score', chunk.get('hybrid_score', 0))
                })
                seen_papers.add(paper_id)
        
        return sources
    
    def generate_literature_review(
        self,
        topic: str,
        max_papers: int = 10,
        max_length: int = 500
    ) -> Dict:
        """
        Generate literature review on topic
        
        Args:
            topic: Research topic
            max_papers: Max papers to include
            max_length: Max words in review
            
        Returns:
            Literature review with sources
        """
        print(f"\n{'='*80}")
        print(f"GENERATING LITERATURE REVIEW: {topic}")
        print(f"{'='*80}")
        
        # Retrieve papers on topic
        print("\n📚 Retrieving relevant papers...")
        retrieval_result = self.retrieval_system.search(
            query=topic,
            top_k=max_papers * 3  # Get more candidates
        )
        
        chunks = retrieval_result['results']
        
        if not chunks:
            return {
                'review': f"No papers found on topic: {topic}",
                'papers': []
            }
        
        # Group by paper
        papers_dict = {}
        for chunk in chunks:
            paper_id = chunk['metadata'].get('paper_id')
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {
                    'title': chunk['metadata'].get('title', 'Unknown'),
                    'author': chunk['metadata'].get('author', 'Unknown'),
                    'year': chunk['metadata'].get('year', 'n.d.'),
                    'chunks': []
                }
            papers_dict[paper_id]['chunks'].append(chunk['text'])
        
        # Limit to max_papers
        papers = list(papers_dict.values())[:max_papers]
        
        # Generate summaries for each paper
        print(f"\n🤖 Generating review from {len(papers)} papers...")
        
        papers_summary = []
        for paper in papers:
            # Combine chunks for this paper
            paper_text = " ".join(paper['chunks'][:3])  # Use first 3 chunks
            
            # Extract key info
            extract_prompt = PromptTemplates.build_extraction_prompt(
                text=paper_text,
                extract_type='key_findings'
            )
            
            summary = self.llm_client.generate(
                prompt=extract_prompt,
                temperature=0.3
            )
            
            papers_summary.append({
                'title': paper['title'],
                'author': paper['author'],
                'year': paper['year'],
                'summary': summary
            })
        
        # Generate literature review
        review_prompt = PromptTemplates.build_literature_review_prompt(
            topic=topic,
            papers_summary=papers_summary,
            max_length=max_length
        )
        
        review = self.llm_client.generate(
            prompt=review_prompt,
            system_prompt="You are an expert at writing literature reviews.",
            temperature=0.6
        )
        
        print(f"\n✅ Literature review generated ({len(review)} chars)")
        
        return {
            'review': review,
            'papers': papers_summary,
            'topic': topic,
            'num_papers': len(papers)
        }
    
    def compare_papers(
        self,
        items: List[str],
        aspects: List[str] = None
    ) -> Dict:
        """
        Compare multiple papers or concepts
        
        Args:
            items: Items to compare (e.g., ["BERT", "GPT"])
            aspects: Aspects to compare on
            
        Returns:
            Comparison with sources
        """
        print(f"\n{'='*80}")
        print(f"COMPARING: {' vs '.join(items)}")
        print(f"{'='*80}")
        
        # Retrieve context for each item
        all_chunks = []
        for item in items:
            print(f"\n📚 Retrieving info on {item}...")
            result = self.retrieval_system.search(
                query=item,
                top_k=5
            )
            all_chunks.extend(result['results'])
        
        # Build context
        context = "\n\n".join([
            f"[Source: {chunk['metadata'].get('title', 'Unknown')}]\n{chunk['text']}"
            for chunk in all_chunks
        ])
        
        # Generate comparison
        print(f"\n🤖 Generating comparison...")
        
        comparison_prompt = PromptTemplates.build_comparison_prompt(
            items=items,
            context=context,
            aspects=aspects
        )
        
        comparison = self.llm_client.generate(
            prompt=comparison_prompt,
            system_prompt="You are an expert at comparing and contrasting research concepts.",
            temperature=0.5
        )
        
        print(f"\n✅ Comparison generated")
        
        return {
            'comparison': comparison,
            'items': items,
            'aspects': aspects,
            'sources': self._format_sources(all_chunks)
        }
    
    def visualize_response(self, response: Dict):
        """
        Pretty print response
        
        Args:
            response: Response from answer_question()
        """
        print(f"\n{'='*80}")
        print("ANSWER")
        print(f"{'='*80}\n")
        
        print(response['answer'])
        
        if response.get('sources'):
            print(f"\n{'='*80}")
            print(f"SOURCES ({len(response['sources'])})")
            print(f"{'='*80}\n")
            
            for i, source in enumerate(response['sources'], 1):
                print(f"{i}. {source['title']}")
                print(f"   Author: {source['author']}")
                print(f"   Relevance: {source.get('relevance_score', 'N/A')}")
                print(f"   Preview: {source['chunk_preview'][:150]}...")
                print()
        
        if response.get('metadata'):
            print(f"{'='*80}")
            print("METADATA")
            print(f"{'='*80}")
            meta = response['metadata']
            print(f"Sources used: {meta.get('num_sources', 0)}")
            print(f"Processing time: {meta.get('processing_time', 0)}s")
            print(f"Total tokens: {meta.get('total_tokens', 0)}")
            print(f"Method: {meta.get('generation_method', 'N/A')}")
            print(f"{'='*80}\n")
    
    def smart_answer(self, user_query: str, **kwargs) -> Dict:
        """
        Smart answer generation with automatic query routing
        
        Args:
            user_query: User's question
            **kwargs: Additional arguments passed to specific methods
            
        Returns:
            Response dict with answer and metadata
        """
        if not self.use_smart_routing or not self.query_router:
            # Fallback to standard Q&A
            return self.answer_question(user_query, **kwargs)
        
        # Route the query
        routing_result = self.query_router.route_query(user_query)
        
        print(f"\n🤖 Smart Routing:")
        print(f"   Detected Type: {self.query_router.get_query_type_description(routing_result['query_type'])}")
        print(f"   Confidence: {routing_result['confidence']:.2f}")
        print(f"   Method: {routing_result['method']}")
        
        # Route to appropriate method
        method_name = routing_result['method']
        params = routing_result['params']
        
        if method_name == 'compare_papers':
            # Comparison query
            items = params.get('items', [])
            if not items:
                # Couldn't extract items, fallback to Q&A
                print("   ⚠️ Couldn't extract items to compare, using Q&A")
                return self.answer_question(user_query, **kwargs)
            
            aspects = kwargs.get('aspects', None)
            response = self.compare_papers(items, aspects)
            
            # Format response to match standard structure
            return {
                'answer': response['comparison'],
                'question': user_query,
                'sources': response.get('sources', []),
                'metadata': {
                    'query_type': 'comparison',
                    'items': items,
                    'aspects': aspects,
                    'routing_confidence': routing_result['confidence']
                }
            }
        
        elif method_name == 'generate_literature_review':
            # Literature review query
            topic = params.get('topic', user_query)
            max_papers = kwargs.get('max_papers', 10)
            max_length = kwargs.get('max_length', 500)
            
            response = self.generate_literature_review(topic, max_papers, max_length)
            
            # Format response to match standard structure
            return {
                'answer': response['review'],
                'question': user_query,
                'sources': [{'title': p['title'], 'author': p['author']} for p in response['papers']],
                'metadata': {
                    'query_type': 'literature_review',
                    'topic': topic,
                    'num_papers': response['num_papers'],
                    'routing_confidence': routing_result['confidence']
                }
            }
        
        else:
            # Standard Q&A (default)
            response = self.answer_question(user_query, **kwargs)
            response['metadata']['query_type'] = routing_result['query_type'].value
            response['metadata']['routing_confidence'] = routing_result['confidence']
            return response


def test_answer_generator():
    """Test complete answer generation system"""
    
    print("\n" + "="*80)
    print("TESTING RESEARCHGPT COMPLETE SYSTEM")
    print("="*80)
    
    try:
        # Initialize system
        generator = AnswerGenerator(
            use_multi_agent=True,
            use_citations=True,
            llm_provider='groq'
        )
        
        # Test questions
        test_questions = [
            "What is attention mechanism in transformers?",
            "How does BERT work?",
            "What are the main results of the paper?"
        ]
        
        for question in test_questions:
            print(f"\n{'─'*80}")
            print(f"Testing: {question}")
            print(f"{'─'*80}")
            
            # Generate answer
            response = generator.answer_question(
                question=question,
                top_k=5,
                include_sources=True
            )
            
            # Visualize
            generator.visualize_response(response)
            
            # Pause between questions
            print("\n" + "─"*80)
            input("Press Enter to continue to next question...")
        
        print("\n✅ All tests complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. You have papers in database (run ingestion)")
        print("2. GROQ_API_KEY is set in .env")
        print("3. All dependencies installed")


if __name__ == "__main__":
    test_answer_generator()