"""
Multi-Agent System Module
Specialized AI agents for different generation tasks
"""

import sys
from typing import List, Dict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from generation.llm_client import LLMClient
from generation.prompt_templates import PromptTemplates


class BaseAgent:
    """Base class for all agents"""
    
    def __init__(self, llm_client: LLMClient, role: str):
        """
        Initialize agent
        
        Args:
            llm_client: LLM client instance
            role: Agent role name
        """
        self.llm = llm_client
        self.role = role
        self.system_prompt = PromptTemplates.SYSTEM_PROMPTS.get(role, "")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Process input (to be overridden by subclasses)
        
        Args:
            input_data: Input dictionary
            
        Returns:
            Output dictionary
        """
        raise NotImplementedError("Subclasses must implement process()")


class AnalyzerAgent(BaseAgent):
    """Agent for analyzing and extracting information from text"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, 'analyzer')
        print("   📊 Analyzer Agent initialized")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Extract key information from chunks
        
        Args:
            input_data: {
                'chunks': List of text chunks,
                'query': Original question
            }
            
        Returns:
            {'extractions': List of extracted information}
        """
        chunks = input_data.get('chunks', [])
        query = input_data.get('query', '')
        
        print(f"\n   📊 Analyzing {len(chunks)} chunks...")
        
        extractions = []
        for i, chunk in enumerate(chunks, 1):
            # Build extraction prompt
            prompt = PromptTemplates.build_extraction_prompt(
                text=chunk.get('text', ''),
                extract_type='key_findings'
            )
            
            # Extract information
            try:
                extraction = self.llm.generate(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    temperature=0.3  # Lower for factual extraction
                )
                
                extractions.append({
                    'chunk_id': chunk.get('id'),
                    'source': chunk.get('metadata', {}).get('title', f'Source {i}'),
                    'content': extraction,
                    'original_text': chunk.get('text', '')
                })
                
            except Exception as e:
                print(f"      ⚠️ Error analyzing chunk {i}: {str(e)}")
                continue
        
        print(f"   ✅ Extracted information from {len(extractions)} chunks")
        
        return {
            'extractions': extractions,
            'query': query
        }


class SynthesizerAgent(BaseAgent):
    """Agent for synthesizing information into coherent answers"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, 'synthesizer')
        print("   🧬 Synthesizer Agent initialized")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Synthesize extracted information into answer
        
        Args:
            input_data: {
                'extractions': List of extractions,
                'query': Original question
            }
            
        Returns:
            {'answer': Synthesized answer}
        """
        extractions = input_data.get('extractions', [])
        query = input_data.get('query', '')
        
        print(f"\n   🧬 Synthesizing information...")
        
        if not extractions:
            return {'answer': "No information available to answer the question."}
        
        # Build synthesis prompt
        prompt = PromptTemplates.build_synthesis_prompt(
            question=query,
            extractions=extractions,
            max_length=300
        )
        
        # Generate synthesis
        try:
            answer = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.5  # Balanced creativity
            )
            
            print(f"   ✅ Generated synthesized answer ({len(answer)} chars)")
            
            return {
                'answer': answer,
                'query': query,
                'extractions': extractions
            }
            
        except Exception as e:
            print(f"   ❌ Error synthesizing: {str(e)}")
            return {'answer': "Error generating answer.", 'error': str(e)}


class CitationAgent(BaseAgent):
    """Agent for adding proper citations"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, 'citation_agent')
        print("   📚 Citation Agent initialized")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Add citations to answer
        
        Args:
            input_data: {
                'answer': Text to cite,
                'extractions': Source information
            }
            
        Returns:
            {'cited_answer': Answer with citations}
        """
        answer = input_data.get('answer', '')
        extractions = input_data.get('extractions', [])
        
        print(f"\n   📚 Adding citations...")
        
        if not extractions:
            return {'cited_answer': answer}
        
        # Prepare source information
        sources = []
        for ext in extractions:
            source_info = {
                'title': ext.get('source', 'Unknown'),
                'year': ext.get('year', 'n.d.'),
                'author': ext.get('author', 'Unknown')
            }
            sources.append(source_info)
        
        # Build citation prompt
        prompt = PromptTemplates.build_citation_prompt(
            text=answer,
            sources=sources
        )
        
        # Add citations
        try:
            cited_answer = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.2  # Very factual
            )
            
            print(f"   ✅ Citations added")
            
            return {
                'cited_answer': cited_answer,
                'sources': sources
            }
            
        except Exception as e:
            print(f"   ⚠️ Error adding citations: {str(e)}")
            return {'cited_answer': answer}


class CriticAgent(BaseAgent):
    """Agent for critiquing and improving answers"""
    
    def __init__(self, llm_client: LLMClient):
        super().__init__(llm_client, 'critic')
        print("   🔍 Critic Agent initialized")
    
    def process(self, input_data: Dict) -> Dict:
        """
        Critique and improve answer
        
        Args:
            input_data: {
                'answer': Current answer,
                'query': Original question,
                'sources': Source context
            }
            
        Returns:
            {'improved_answer': Improved answer, 'critique': Critique notes}
        """
        answer = input_data.get('answer', '')
        query = input_data.get('query', '')
        sources = input_data.get('sources', '')
        
        print(f"\n   🔍 Critiquing answer...")
        
        # Build critique prompt
        prompt = PromptTemplates.build_critique_prompt(
            question=query,
            answer=answer,
            sources=str(sources)
        )
        
        # Generate critique
        try:
            critique_response = self.llm.generate(
                prompt=prompt,
                system_prompt=self.system_prompt,
                temperature=0.4
            )
            
            # Try to extract improved answer (simple parsing)
            improved_answer = answer  # Default to original
            critique_notes = critique_response
            
            # Look for "IMPROVED:" or similar markers
            if "IMPROVED" in critique_response:
                parts = critique_response.split("IMPROVED")
                if len(parts) > 1:
                    improved_answer = parts[1].strip()
                    critique_notes = parts[0].strip()
            
            print(f"   ✅ Critique complete")
            
            return {
                'improved_answer': improved_answer,
                'critique': critique_notes,
                'original_answer': answer
            }
            
        except Exception as e:
            print(f"   ⚠️ Error critiquing: {str(e)}")
            return {'improved_answer': answer, 'critique': ''}


class AgentOrchestrator:
    """Orchestrates multiple agents in a workflow"""
    
    def __init__(self, llm_client: LLMClient = None):
        """
        Initialize orchestrator
        
        Args:
            llm_client: LLM client (creates new if None)
        """
        print("\n🎭 Initializing Multi-Agent System...")
        print("="*60)
        
        # Initialize LLM client
        if llm_client is None:
            self.llm = LLMClient(provider='groq')
        else:
            self.llm = llm_client
        
        # Initialize agents
        self.agents = {
            'analyzer': AnalyzerAgent(self.llm),
            'synthesizer': SynthesizerAgent(self.llm),
            'citation': CitationAgent(self.llm),
            'critic': CriticAgent(self.llm)
        }
        
        print("="*60)
        print("✅ Multi-Agent System Ready!")
    
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict],
        use_citations: bool = True,
        use_critique: bool = True
    ) -> Dict:
        """
        Generate answer using multi-agent workflow
        
        Args:
            query: User question
            chunks: Retrieved chunks
            use_citations: Whether to add citations
            use_critique: Whether to critique/improve
            
        Returns:
            Complete answer with metadata
        """
        print(f"\n{'='*80}")
        print(f"GENERATING ANSWER: {query}")
        print(f"{'='*80}")
        
        # Stage 1: Analyze chunks
        analysis_result = self.agents['analyzer'].process({
            'chunks': chunks,
            'query': query
        })
        
        # Stage 2: Synthesize answer
        synthesis_result = self.agents['synthesizer'].process({
            'extractions': analysis_result['extractions'],
            'query': query
        })
        
        answer = synthesis_result['answer']
        
        # Stage 3: Add citations (optional)
        if use_citations:
            citation_result = self.agents['citation'].process({
                'answer': answer,
                'extractions': analysis_result['extractions']
            })
            answer = citation_result.get('cited_answer', answer)
        
        # Stage 4: Critique and improve (optional)
        if use_critique:
            critique_result = self.agents['critic'].process({
                'answer': answer,
                'query': query,
                'sources': analysis_result['extractions']
            })
            answer = critique_result.get('improved_answer', answer)
        
        print(f"\n{'='*80}")
        print("✅ ANSWER GENERATION COMPLETE")
        print(f"{'='*80}\n")
        
        return {
            'answer': answer,
            'query': query,
            'sources': analysis_result['extractions'],
            'num_chunks': len(chunks)
        }
    
    def simple_generate(self, query: str, context: str) -> str:
        """
        Simple generation without multi-agent (faster)
        
        Args:
            query: User question
            context: Retrieved context
            
        Returns:
            Answer string
        """
        prompt = PromptTemplates.build_qa_prompt(query, context)
        
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a helpful research assistant.",
            temperature=0.5
        )
        
        return answer


def test_agents():
    """Test multi-agent system"""
    
    # Mock chunks
    mock_chunks = [
        {
            'id': 'chunk_1',
            'text': 'The attention mechanism allows transformers to focus on relevant parts of the input sequence. It computes weighted sums of values.',
            'metadata': {'title': 'Attention Is All You Need', 'author': 'Vaswani et al.', 'year': '2017'}
        },
        {
            'id': 'chunk_2',
            'text': 'Self-attention enables parallel processing and captures long-range dependencies better than RNNs.',
            'metadata': {'title': 'Transformer Tutorial', 'author': 'Smith', 'year': '2020'}
        }
    ]
    
    query = "What is attention mechanism in transformers?"
    
    print("\n" + "="*80)
    print("TESTING MULTI-AGENT SYSTEM")
    print("="*80)
    
    try:
        # Initialize orchestrator
        orchestrator = AgentOrchestrator()
        
        # Generate answer
        result = orchestrator.generate_answer(
            query=query,
            chunks=mock_chunks,
            use_citations=True,
            use_critique=True
        )
        
        # Display result
        print("\nFINAL ANSWER:")
        print("─" * 80)
        print(result['answer'])
        print("─" * 80)
        print(f"\nSources used: {result['num_chunks']}")
        
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure:")
        print("1. GROQ_API_KEY is set in .env")
        print("2. You have internet connection")


if __name__ == "__main__":
    test_agents()