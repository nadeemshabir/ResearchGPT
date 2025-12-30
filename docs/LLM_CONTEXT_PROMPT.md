# Prompt for LLMs - ResearchGPT Project Context

## Copy-Paste This to Any LLM:

---

I'm working on a project called **ResearchGPT** - a Retrieval-Augmented Generation (RAG) system for research paper analysis. Here's the complete context:

### **Project Purpose:**
Users upload PDF research papers and ask questions in natural language. The system retrieves relevant information and generates AI answers with citations.

### **Architecture (3 Pipelines):**

**1. INGESTION PIPELINE:**
- `pdf_parser.py` - Extracts text from PDFs (supports PyPDF2, pdfplumber, PyMuPDF)
- `chunker.py` - Splits text into 1000-token chunks with 200-token overlap
- `embedder.py` - Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
- `database.py` - Stores chunks in ChromaDB vector database
- `pipeline.py` - Orchestrates the entire ingestion process

**2. RETRIEVAL PIPELINE:**
- `query_processor.py` - Enhances queries (expansion, spell correction)
- `semantic_search.py` - Vector similarity search (cosine similarity)
- `keyword_search.py` - BM25 keyword search
- `hybrid_search.py` - Combines semantic (70%) + keyword (30%) using RRF
- `reranker.py` - Cross-encoder reranking for better relevance
- `retrieval_system.py` - Orchestrates the entire retrieval process

**3. GENERATION PIPELINE:**
- `llm_client.py` - Unified interface for Groq/OpenAI/Gemini LLMs
- `prompt_templates.py` - 7 pre-built prompt builders (Q&A, comparison, review, etc.)
- `query_router.py` - Automatically detects query type (Q&A, comparison, review)
- `agents.py` - Multi-agent system with 4 specialized agents:
  - AnalyzerAgent: Extracts key info from chunks
  - SynthesizerAgent: Combines info into coherent answer
  - CitationAgent: Adds academic citations
  - CriticAgent: Reviews and improves answer
- `citation_manager.py` - Manages citations and bibliography
- `answer_generator.py` - **MAIN ORCHESTRATOR** - ties everything together

### **Main Entry Point:**
```python
from src.generation.answer_generator import AnswerGenerator

generator = AnswerGenerator(use_smart_routing=True)
response = generator.smart_answer(user_query)
```

### **Smart Routing Feature:**
The `query_router.py` automatically detects query type:
- **Q&A**: "What is BERT?" → Routes to `answer_question()`
- **Comparison**: "Compare BERT and GPT" → Routes to `compare_papers()`
- **Literature Review**: "Review papers on transformers" → Routes to `generate_literature_review()`

Uses pattern matching and confidence scoring to route queries automatically.

### **Data Flow:**
```
User Query
  ↓
[query_router] Detect type (Q&A/Comparison/Review)
  ↓
[retrieval_system] 
  ├─ Enhance query
  ├─ Hybrid search (semantic + keyword)
  └─ Rerank with cross-encoder
  ↓
Top-K Relevant Chunks
  ↓
[Multi-Agent Pipeline]
  ├─ Analyzer: Extract key info
  ├─ Synthesizer: Combine info
  ├─ Citation: Add citations
  └─ Critic: Improve quality
  ↓
High-Quality Answer with Citations
```

### **Technology Stack:**
- **PDFs**: PyPDF2, pdfplumber, PyMuPDF
- **Embeddings**: sentence-transformers, PyTorch
- **Database**: ChromaDB
- **Search**: BM25 (rank-bm25), Cosine Similarity, Cross-Encoder
- **LLMs**: Groq (Llama 3.3 70B), OpenAI (GPT-4), Gemini (1.5 Flash)
- **Text Processing**: LangChain, tiktoken

### **File Count:**
- **Total**: 19 Python files
- **Ingestion**: 5 files
- **Retrieval**: 6 files
- **Generation**: 7 files
- **Tests**: 1 file

### **Key Design Patterns:**
1. **Modular Architecture**: Each component is independent
2. **Orchestrator Pattern**: Pipeline classes coordinate components
3. **Multi-Agent System**: Specialized agents for quality
4. **Hybrid Search**: Combines semantic and keyword search
5. **Smart Routing**: Automatic query type detection

### **Main Features:**
✅ Hybrid search (semantic + keyword)
✅ Multi-agent generation (4-stage quality pipeline)
✅ Smart query routing (automatic)
✅ Citation management (academic style)
✅ Multiple LLM support (Groq/OpenAI/Gemini)
✅ Literature review generation
✅ Paper comparison

### **Usage Example:**
```python
# 1. Ingest papers
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
pipeline.process_directory("data/raw/")

# 2. Ask questions (automatic routing!)
from src.generation.answer_generator import AnswerGenerator
generator = AnswerGenerator(use_smart_routing=True)

# All these work automatically:
response1 = generator.smart_answer("What is BERT?")
response2 = generator.smart_answer("Compare BERT and GPT")
response3 = generator.smart_answer("Review papers on transformers")
```

### **Response Format:**
```python
{
    'answer': str,              # Generated answer
    'question': str,            # Original question
    'sources': List[Dict],      # Source papers
    'metadata': {
        'query_type': str,      # Detected type
        'routing_confidence': float,
        'num_sources': int,
        'processing_time': float
    }
}
```

### **Environment Setup:**
- API keys in `.env`: GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
- Cache on D drive to save C drive space
- Virtual environment with all dependencies

### **Project Structure:**
```
ResearchGPT/
├── src/
│   ├── ingestion/      # PDF → Database
│   ├── retrieval/      # Search chunks
│   └── generation/     # Generate answers
├── data/raw/           # Upload PDFs here
├── docs/               # Documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── QUICK_REFERENCE.md
│   └── SMART_ROUTING_GUIDE.md
└── .env                # API keys
```

### **Current Status:**
✅ All modules implemented and functional
✅ Smart routing system added
✅ Multi-agent pipeline working
✅ Documentation complete
✅ Ready for UI development

### **Next Steps:**
Building a UI (Streamlit/Flask) where users can:
1. Upload PDFs
2. Ask questions in natural language
3. Get AI-generated answers with citations

The backend is complete and ready to integrate with any frontend.

---

**Now you have complete context about my ResearchGPT project. Feel free to ask me anything about it!**
