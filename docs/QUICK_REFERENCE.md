# ResearchGPT - Quick Reference

## 🎯 One-Line Summary
ResearchGPT is a RAG system that lets users upload research papers and ask questions in natural language, getting AI-generated answers with citations.

## 📂 Project Structure (Simplified)

```
ResearchGPT/
├── src/
│   ├── ingestion/      # PDF → Database (6 files)
│   ├── retrieval/      # Search chunks (6 files)
│   └── generation/     # Generate answers (7 files)
├── data/raw/           # Upload PDFs here
├── docs/               # Documentation
└── .env                # API keys
```

## 🔄 Three Main Pipelines

### 1. INGESTION
```
PDF → Extract Text → Chunk → Embed → Store in DB
```

### 2. RETRIEVAL
```
Query → Enhance → Search (Semantic + Keyword) → Rerank → Top-K Chunks
```

### 3. GENERATION
```
Chunks + Query → LLM → Answer + Citations
```

## 📋 All Files Explained

### **INGESTION (src/ingestion/)**
| File | What It Does |
|------|--------------|
| `pdf_parser.py` | Extract text from PDFs |
| `chunker.py` | Split text into chunks |
| `embedder.py` | Generate embeddings |
| `database.py` | Store in ChromaDB |
| `pipeline.py` | Orchestrate ingestion |

### **RETRIEVAL (src/retrieval/)**
| File | What It Does |
|------|--------------|
| `semantic_search.py` | Vector similarity search |
| `keyword_search.py` | BM25 keyword search |
| `hybrid_search.py` | Combine both searches |
| `reranker.py` | Rerank with cross-encoder |
| `query_processor.py` | Enhance queries |
| `retrieval_system.py` | Orchestrate retrieval |

### **GENERATION (src/generation/)**
| File | What It Does |
|------|--------------|
| `llm_client.py` | Talk to LLMs (Groq/OpenAI/Gemini) |
| `prompt_templates.py` | Pre-built prompts |
| `agents.py` | 4-agent quality pipeline |
| `citation_manager.py` | Handle citations |
| `query_router.py` | Auto-detect query type |
| `answer_generator.py` | **MAIN FILE** - Orchestrate everything |
| `smart_answer_example.py` | Demo script |

## 🚀 Quick Start

### **1. Setup**
```bash
pip install -r requirements.txt
# Add GROQ_API_KEY to .env
```

### **2. Ingest Papers**
```python
from src.ingestion.pipeline import IngestionPipeline
pipeline = IngestionPipeline()
pipeline.process_directory("data/raw/")
```

### **3. Ask Questions**
```python
from src.generation.answer_generator import AnswerGenerator
generator = AnswerGenerator(use_smart_routing=True)
response = generator.smart_answer("What is BERT?")
print(response['answer'])
```

## 🎯 Main Method to Use

```python
# This ONE method handles EVERYTHING!
generator.smart_answer(user_query)

# Examples:
generator.smart_answer("What is BERT?")           # → Q&A
generator.smart_answer("Compare BERT and GPT")    # → Comparison
generator.smart_answer("Review transformers")     # → Literature Review
```

## 🔑 Key Features

✅ Hybrid Search (Semantic + Keyword)
✅ Multi-Agent Generation (4 stages)
✅ Smart Query Routing (automatic)
✅ Citation Management
✅ Multiple LLM Support (Groq/OpenAI/Gemini)

## 📊 File Dependencies

```
answer_generator.py (MAIN)
    ├── query_router.py (detect query type)
    ├── retrieval_system.py (get chunks)
    │   ├── hybrid_search.py
    │   ├── reranker.py
    │   └── query_processor.py
    ├── agents.py (multi-agent pipeline)
    │   ├── llm_client.py
    │   └── prompt_templates.py
    └── citation_manager.py (add citations)
```

## 💡 For LLMs to Understand

**Tell the LLM:**

"ResearchGPT is a 3-pipeline RAG system:

1. **Ingestion**: Converts PDFs to searchable chunks in ChromaDB
   - Files: pdf_parser → chunker → embedder → database → pipeline

2. **Retrieval**: Finds relevant chunks using hybrid search
   - Files: semantic_search + keyword_search → hybrid_search → reranker → retrieval_system

3. **Generation**: Creates answers using LLMs
   - Files: query_router → answer_generator → agents → llm_client
   - Uses prompt_templates and citation_manager

**Main Entry Point:** `answer_generator.py` with `smart_answer()` method

**Smart Routing:** Automatically detects if query is Q&A, comparison, or literature review

**Multi-Agent:** 4-stage pipeline (Analyzer → Synthesizer → Citation → Critic) for quality"

## 🎓 Learning Order

1. Read `PROJECT_OVERVIEW.md` (full details)
2. Look at `answer_generator.py` (main orchestrator)
3. Run `smart_answer_example.py` (see it work)
4. Explore individual modules as needed

## 📝 Common Tasks

### **Add a new paper:**
```python
pipeline.process_paper("path/to/paper.pdf")
```

### **Ask any question:**
```python
response = generator.smart_answer("your question")
```

### **Change LLM provider:**
```python
generator = AnswerGenerator(llm_provider='openai')
```

### **Disable multi-agent (faster):**
```python
generator = AnswerGenerator(use_multi_agent=False)
```

## 🔧 Configuration

**In `.env`:**
```
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
```

**In code:**
```python
AnswerGenerator(
    use_multi_agent=True,      # Quality (slower)
    use_citations=True,         # Add citations
    llm_provider='groq',        # LLM to use
    use_smart_routing=True      # Auto-detect query type
)
```

## 📊 Tech Stack Summary

- **PDFs**: PyPDF2, pdfplumber, PyMuPDF
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Database**: ChromaDB
- **Search**: BM25 + Cosine Similarity + Cross-Encoder
- **LLMs**: Groq (Llama 3.3), OpenAI (GPT-4), Gemini

## ✅ Summary

**19 Python files total:**
- 5 ingestion files
- 6 retrieval files  
- 7 generation files
- 1 test file

**Main file:** `answer_generator.py`
**Main method:** `smart_answer()`
**Main feature:** Automatic query routing

**That's it!** 🚀
