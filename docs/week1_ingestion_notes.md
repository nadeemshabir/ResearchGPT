# 📚 ResearchGPT - Ingestion Pipeline Notes

## 🎯 Overview

The **Ingestion Pipeline** is the first and most critical part of ResearchGPT. It transforms PDF research papers into searchable vector embeddings stored in a database.

### Pipeline Flow
```
PDF File → Extract Text → Clean → Chunk → Generate Embeddings → Store in DB
   ↓           ↓           ↓       ↓            ↓                    ↓
30 sec    5-10 sec    instant  instant      5-10 sec             instant
```

**Total Time per Paper:** ~30-60 seconds  
**Processing:** Done ONCE per paper  
**Querying:** Instant after upload

---

## 📁 Project Structure

```
researchgpt/
├── venv/                          # Virtual environment
├── data/
│   ├── raw/                       # Original PDFs go here
│   ├── processed/                 # Processed data
│   └── chroma_db/                 # Vector database storage
├── models/                        # Saved models, configs
├── src/
│   ├── __init__.py
│   └── ingestion/
│       ├── __init__.py
│       ├── pdf_parser.py          # Extract text from PDFs
│       ├── chunker.py             # Split text into chunks
│       ├── embedder.py            # Generate embeddings
│       ├── database.py            # ChromaDB operations
│       └── pipeline.py            # Complete orchestration
├── tests/
├── requirements.txt
├── .env                           # Environment variables
└── README.md
```

---

## 🔧 Installation & Setup

### Step 1: Environment Setup
```bash
# Create project folder
mkdir researchgpt && cd researchgpt

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Environment Variables (.env)
```bash
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_key_here  # Optional

# Database Settings
CHROMA_DB_PATH=./data/chroma_db
COLLECTION_NAME=research_papers

# Embedding Model
EMBEDDING_MODEL=allenai/specter  # Best for scientific papers
# Alternative: all-MiniLM-L6-v2 (faster, general purpose)

# Chunking Settings
CHUNK_SIZE=1000          # Target chunk size in tokens
CHUNK_OVERLAP=200        # Overlap between chunks

# Processing Settings
MAX_TOKENS=4000          # Max context window
DEVICE=cpu               # cpu, cuda, or mps (Apple Silicon)
```

---

## 📄 Component 1: PDF Parser

### Purpose
Extract text and metadata from PDF files

### Key Features
- **3 Parsing Methods:**
  - `pypdf2` - Basic, fastest
  - `pdfplumber` - Better quality, slower
  - `pymupdf` - **Recommended** (best quality)

### Usage
```python
from src.ingestion.pdf_parser import PDFParser

# Initialize
parser = PDFParser(method="pymupdf")

# Extract text
result = parser.extract_text("paper.pdf")

# Result contains:
# - text: Full text content
# - num_pages: Number of pages
# - metadata: Title, author, etc.
# - file_name: Original filename
# - file_size_mb: File size
```

### Key Methods
```python
# Extract text from PDF
result = parser.extract_text(pdf_path)

# Clean extracted text
cleaned = parser.clean_text(result['text'])
```

### Common Issues
- **Empty text:** PDF might be scanned images (needs OCR)
- **Garbled text:** Try different parsing methods
- **Missing metadata:** Not all PDFs have metadata

---

## 🔪 Component 2: Text Chunker

### Purpose
Split long text into optimal chunks for embedding and retrieval

### Why Chunking?
- Embedding models have max token limits (typically 512-8192)
- Smaller chunks = more precise retrieval
- Larger chunks = more context
- **Sweet spot:** 1000 tokens with 200 overlap

### Key Concepts

**Token vs Character:**
- Tokens ≠ Words
- "Hello" = 1 token
- "Transformer" = 1-2 tokens
- Average: 1 token ≈ 4 characters

**Overlap:**
```
Chunk 1: [------------------]
Chunk 2:        [------------------]
              ↑ Overlap prevents losing context at boundaries
```

### Usage
```python
from src.ingestion.chunker import TextChunker

# Initialize
chunker = TextChunker(
    chunk_size=1000,     # Target tokens per chunk
    chunk_overlap=200    # Overlap between chunks
)

# Basic chunking
chunks = chunker.chunk_text(text, metadata={"title": "Paper Title"})

# Context-aware chunking (preserves section headers)
chunks = chunker.chunk_with_context(text, metadata={...})

# Visualize chunks
chunker.visualize_chunks(chunks, max_display=5)
```

### Chunk Structure
```python
chunk = {
    'text': 'The actual chunk text...',
    'chunk_id': 0,                    # Unique ID within paper
    'num_tokens': 987,                # Token count
    'num_characters': 4231,           # Character count
    'section_title': 'Introduction',  # If context-aware
    'metadata': {...}                 # Paper metadata
}
```

### Advanced: Section Detection
Automatically identifies common sections:
- Abstract
- Introduction
- Methods/Methodology
- Results
- Discussion
- Conclusion
- References

---

## 🧠 Component 3: Embedding Generator

### Purpose
Convert text chunks into numerical vectors (embeddings)

### What are Embeddings?
```
Text: "Transformers use attention"
      ↓
Embedding: [0.23, -0.45, 0.67, ..., 0.12]  (384-1024 dimensions)
```

Similar text → Similar vectors (close in vector space)

### Best Models for Research Papers

| Model | Dimension | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `allenai/specter` | 768 | Slow | Excellent | **Scientific papers** ⭐ |
| `allenai/specter-2` | 768 | Slow | Excellent | Scientific papers (newer) |
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose, testing |
| `all-mpnet-base-v2` | 768 | Medium | Very Good | General purpose |

**Recommendation:** Use `allenai/specter` for production, `all-MiniLM-L6-v2` for testing

### Usage
```python
from src.ingestion.embedder import EmbeddingGenerator

# Initialize
embedder = EmbeddingGenerator(
    model_name='allenai/specter',
    device='cpu'  # or 'cuda' if you have GPU
)

# Generate embeddings for chunks
chunks_with_embeddings = embedder.embed_chunks(chunks)

# Single text embedding
embedding = embedder.generate_single_embedding("Some text")

# Get model info
info = embedder.get_model_info()
# Returns: model_name, embedding_dimension, device, max_seq_length
```

### Performance Tips
- **GPU:** 10-20x faster than CPU
- **Batch size:** Larger = faster (but more memory)
- **Device auto-detection:**
  - CUDA (NVIDIA GPU)
  - MPS (Apple Silicon M1/M2)
  - CPU (fallback)

### Memory Requirements
- `allenai/specter` (768d): ~500MB RAM
- `all-MiniLM-L6-v2` (384d): ~80MB RAM
- Processing 100 chunks: ~1-2GB RAM

---

## 🗄️ Component 4: Vector Database (ChromaDB)

### Purpose
Store embeddings and enable fast similarity search

### Why ChromaDB?
- ✅ Free and open-source
- ✅ Runs locally (no API needed)
- ✅ Persistent storage (data saved to disk)
- ✅ Easy to use
- ✅ Fast similarity search

### Key Concepts

**Collection:** Container for documents (like a table)
**Document:** Text chunk + embedding + metadata
**Query:** Find similar documents using embedding similarity

### Usage
```python
from src.ingestion.database import VectorDatabase

# Initialize
db = VectorDatabase(
    db_path='./data/chroma_db',
    collection_name='research_papers'
)

# Add chunks
db.add_chunks(
    chunks=chunks_with_embeddings,
    paper_id='paper_001',
    paper_metadata={'title': 'Paper Title', 'author': 'Author Name'}
)

# Query (requires query embedding)
results = db.query(
    query_embedding=[0.1, 0.2, ...],  # 384 or 768 dimensions
    n_results=5                        # Top 5 results
)

# Get paper by ID
paper_chunks = db.get_by_paper_id('paper_001')

# Delete paper
db.delete_paper('paper_001')

# List all papers
papers = db.list_papers()

# Database statistics
db.print_stats()
```

### Database Operations

**Add:**
```python
# Adds chunks with unique IDs: paper_id_chunk_0, paper_id_chunk_1, etc.
db.add_chunks(chunks, paper_id='paper_123', paper_metadata={...})
```

**Query:**
```python
# Returns most similar chunks
results = db.query(query_embedding, n_results=10)

# Result structure:
{
    'documents': [[chunk1_text, chunk2_text, ...]],
    'distances': [[0.23, 0.45, ...]],  # Lower = more similar
    'metadatas': [[{...}, {...}, ...]],
    'ids': [['id1', 'id2', ...]]
}
```

**Filter:**
```python
# Query with metadata filter
results = db.query(
    query_embedding,
    n_results=5,
    filter_dict={'author': 'John Doe'}  # Only from this author
)
```

### Data Persistence
- Stored in: `./data/chroma_db/` (default)
- Automatically saved to disk
- Survives restarts
- Can be backed up by copying folder

---

## 🚀 Component 5: Complete Pipeline

### Purpose
Orchestrate all components into a single, easy-to-use interface

### Usage

**Process Single Paper:**
```python
from src.ingestion.pipeline import IngestionPipeline

# Initialize
pipeline = IngestionPipeline(
    pdf_method='pymupdf',
    chunk_size=1000,
    chunk_overlap=200,
    embedding_model='allenai/specter'
)

# Process one paper
stats = pipeline.process_paper('data/raw/paper.pdf')

# Stats returned:
{
    'paper_id': 'paper',
    'file_name': 'paper.pdf',
    'num_pages': 15,
    'num_chunks': 23,
    'num_stored': 23,
    'processing_time_seconds': 34.5,
    'metadata': {...}
}
```

**Process Multiple Papers:**
```python
# Process entire folder
stats_list = pipeline.process_directory(
    directory_path='data/raw/',
    file_pattern='*.pdf'
)

# Returns list of stats for each paper
```

**Get Pipeline Info:**
```python
info = pipeline.get_pipeline_stats()
# Returns database stats, model info, chunker settings
```

### Pipeline Steps (Detailed)

**Step 1: PDF Extraction (10-15s)**
- Read PDF file
- Extract text from all pages
- Extract metadata (title, author, etc.)
- Clean text (remove artifacts)

**Step 2: Chunking (1-2s)**
- Count tokens in text
- Split into chunks with overlap
- Preserve section context if possible
- Add metadata to each chunk

**Step 3: Generate Embeddings (10-30s)**
- Convert each chunk to embedding vector
- Batch processing for efficiency
- Add embedding to chunk dictionary

**Step 4: Store in Database (1-2s)**
- Create unique IDs for chunks
- Add to ChromaDB collection
- Associate with paper metadata
- Persist to disk

---

## ⚙️ Configuration & Best Practices

### Chunking Strategy

**Small Chunks (500 tokens):**
- ✅ Precise retrieval
- ✅ Less noise
- ❌ Less context
- **Use for:** Specific fact-finding

**Medium Chunks (1000 tokens) ⭐ Recommended:**
- ✅ Good balance
- ✅ Enough context
- ✅ Not too noisy
- **Use for:** General Q&A

**Large Chunks (2000 tokens):**
- ✅ Maximum context
- ❌ Less precise
- ❌ More noise
- **Use for:** Summarization

### Overlap Strategy

**No Overlap (0 tokens):**
- ❌ Information lost at boundaries
- ❌ Sentences split mid-way

**Small Overlap (100 tokens):**
- ✅ Minimal redundancy
- ⚠️ May still miss context

**Medium Overlap (200 tokens) ⭐ Recommended:**
- ✅ Good context preservation
- ✅ Acceptable redundancy
- **Best for:** Most use cases

**Large Overlap (500 tokens):**
- ✅ Maximum context preservation
- ❌ High redundancy (storage waste)

### Performance Tuning

**Fast Processing (Testing):**
```python
pipeline = IngestionPipeline(
    embedding_model='all-MiniLM-L6-v2',  # Fast model
    chunk_size=500,                       # Smaller chunks
    pdf_method='pypdf2'                   # Fastest parser
)
# ~15s per paper
```

**High Quality (Production):**
```python
pipeline = IngestionPipeline(
    embedding_model='allenai/specter',    # Best for papers
    chunk_size=1000,                      # Optimal size
    pdf_method='pymupdf'                  # Best quality
)
# ~45s per paper
```

**GPU Acceleration:**
```python
# In .env file:
DEVICE=cuda  # If you have NVIDIA GPU

# Result: 5-10x faster embedding generation
```

---

## 🧪 Testing & Validation

### Test Individual Components

**Test PDF Parser:**
```bash
python src/ingestion/pdf_parser.py
```

**Test Chunker:**
```bash
python src/ingestion/chunker.py
```

**Test Embedder:**
```bash
python src/ingestion/embedder.py
```

**Test Database:**
```bash
python src/ingestion/database.py
```

**Test Full Pipeline:**
```bash
python src/ingestion/pipeline.py
```

### Validation Checklist

✅ **PDF Extraction:**
- Text extracted correctly (no garbled characters)
- Metadata captured (title, author)
- All pages processed

✅ **Chunking:**
- Chunks within token limits
- Overlap working correctly
- Section headers preserved

✅ **Embeddings:**
- Correct dimensions (384 or 768)
- All chunks have embeddings
- No NaN values

✅ **Database:**
- Chunks stored successfully
- Can query and retrieve
- Paper metadata attached

✅ **End-to-End:**
- Complete paper processing
- Stats accurate
- Database queryable

---

## 🐛 Common Issues & Solutions

### Issue 1: Empty Text from PDF
**Symptom:** `len(text) == 0` or very short

**Causes:**
- PDF is scanned images (no text layer)
- PDF uses custom fonts
- PDF is encrypted

**Solutions:**
```python
# Try different parser
parser = PDFParser(method='pdfplumber')  # or 'pypdf2'

# For scanned PDFs, need OCR:
# pip install pytesseract
# (Add OCR functionality - advanced)
```

### Issue 2: Out of Memory
**Symptom:** `MemoryError` or system freeze

**Causes:**
- Too many papers at once
- Large embedding model
- Large batch size

**Solutions:**
```python
# Process one at a time
for pdf_file in pdf_files:
    pipeline.process_paper(pdf_file)
    # Clears memory between papers

# Use smaller model
embedder = EmbeddingGenerator('all-MiniLM-L6-v2')  # Uses less RAM

# Reduce batch size
chunks_with_embeddings = embedder.embed_chunks(chunks, batch_size=8)
```

### Issue 3: Slow Processing
**Symptom:** Taking >2 minutes per paper

**Causes:**
- Using CPU instead of GPU
- Large embedding model
- Many pages

**Solutions:**
```python
# Enable GPU (if available)
# In .env: DEVICE=cuda

# Use faster model for testing
embedder = EmbeddingGenerator('all-MiniLM-L6-v2')

# Increase batch size (if memory allows)
embedder.embed_chunks(chunks, batch_size=64)
```

### Issue 4: Database Not Persisting
**Symptom:** Data disappears after restart

**Causes:**
- Using in-memory ChromaDB
- Wrong database path

**Solutions:**
```python
# Ensure using PersistentClient
db = VectorDatabase(db_path='./data/chroma_db')

# Check path exists
import os
os.makedirs('./data/chroma_db', exist_ok=True)

# Verify data persisted
db.print_stats()  # Should show papers even after restart
```

### Issue 5: Import Errors
**Symptom:** `ModuleNotFoundError`

**Solutions:**
```bash
# Ensure in correct directory
cd researchgpt

# Ensure venv activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:./src"  # Mac/Linux
set PYTHONPATH=%PYTHONPATH%;./src        # Windows
```

---

## 📊 Performance Benchmarks

### Processing Speed (Single Paper, 15 pages)

| Configuration | Time | Notes |
|--------------|------|-------|
| CPU + Fast Model | 15-20s | Good for testing |
| CPU + Quality Model | 40-60s | Production quality |
| GPU + Quality Model | 10-15s | Best performance |

### Database Size

| Papers | Chunks | Disk Space | RAM Usage |
|--------|--------|------------|-----------|
| 1 | 20 | ~2 MB | 100 MB |
| 10 | 200 | ~20 MB | 200 MB |
| 100 | 2000 | ~200 MB | 500 MB |
| 1000 | 20000 | ~2 GB | 2 GB |

### Token Counts (Average Research Paper)

- **Abstract:** 150-250 tokens
- **Introduction:** 500-1000 tokens
- **Methods:** 800-1500 tokens
- **Results:** 600-1200 tokens
- **Discussion:** 600-1000 tokens
- **Full Paper:** 6000-12000 tokens
- **Chunks (1000 tokens):** 8-15 chunks per paper

---

## 🎯 Next Steps

### Week 1 Complete ✅
You now have:
- Working PDF extraction
- Smart text chunking
- Embedding generation
- Vector database storage
- Complete ingestion pipeline

### Week 2: Retrieval System
Next, you'll build:
- Semantic search (vector similarity)
- Keyword search (BM25)
- Hybrid search (combining both)
- Re-ranking with cross-encoders
- Query processing

### Week 3: Multi-Agent Generation
Then:
- LLM integration (Groq API)
- Prompt engineering
- Multi-agent system
- Answer synthesis
- Citation formatting

---

## 📚 Key Takeaways

### What You Built
✅ Complete PDF → Database pipeline  
✅ Handles multiple papers automatically  
✅ Stores data persistently  
✅ Ready for querying (Week 2)  

### Critical Concepts
1. **Embeddings:** Text converted to vectors for similarity search
2. **Chunking:** Breaking text into optimal pieces
3. **Vector DB:** Fast similarity search on embeddings
4. **Pipeline:** Orchestrating multiple steps automatically

### Why This Matters
- Foundation for all RAG systems
- Reusable for any document processing task
- Scalable to thousands of papers
- Production-ready architecture

---

## 🔗 Resources

### Documentation
- **ChromaDB:** https://docs.trychroma.com/
- **Sentence Transformers:** https://www.sbert.net/
- **LangChain:** https://python.langchain.com/docs/

### Papers to Read
- "Attention Is All You Need" (Transformers)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "Sentence-BERT" (SBERT paper)

### Testing Papers
- Download from: https://arxiv.org/
- Recommended: CS, ML, NLP papers
- Format: PDF

---

## 💡 Pro Tips

1. **Start small:** Test with 1-2 papers first
2. **Monitor memory:** Watch RAM usage
3. **Backup database:** Copy `chroma_db` folder regularly
4. **Version control:** Commit after each working feature
5. **Document changes:** Keep notes on what works
6. **Profile code:** Use `time` to find bottlenecks
7. **Test edge cases:** Empty PDFs, huge PDFs, corrupted files
8. **Validate data:** Always check a few chunks manually

---

## ✅ Completion Checklist

Mark these off as you complete them:

- [ ] Environment setup (venv, dependencies)
- [ ] `.env` file configured
- [ ] PDF parser working with test PDF
- [ ] Chunker creating proper chunks
- [ ] Embedder generating vectors
- [ ] Database storing and retrieving
- [ ] Full pipeline processing 1 paper
- [ ] Batch processing multiple papers
- [ ] Database statistics showing correctly
- [ ] Code committed to Git
- [ ] Ready for Week 2 (Retrieval)

---

**🎉 Congratulations on completing Week 1!**

You've built a production-quality ingestion pipeline. This is the hardest part - everything else builds on this foundation.

**Next:** Week 2 - Retrieval System (Query → Find relevant chunks)