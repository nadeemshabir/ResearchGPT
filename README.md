# 🔬 ResearchGPT

**AI-Powered Research Paper Analysis with RAG, Smart Query Routing & Multi-Agent Generation**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ResearchGPT is a complete Retrieval-Augmented Generation (RAG) system that allows you to upload PDF research papers, ask questions in natural language, and get AI-generated answers with citations.

![ResearchGPT Demo](docs/demo.png)

---

## ✨ Features

- 📄 **PDF Ingestion** - Upload and process research papers automatically
- 🔍 **Hybrid Search** - Combines semantic (70%) + keyword (30%) search for best results
- 🤖 **Multi-Agent Pipeline** - 4-stage answer generation (Analyze → Synthesize → Cite → Critique)
- 🧠 **Smart Query Routing** - Automatically detects query type (Q&A, Comparison, Literature Review)
- 📚 **Literature Reviews** - Generate comprehensive reviews across papers
- ⚖️ **Paper Comparison** - Compare concepts, methodologies, or results
- 📝 **Academic Citations** - Automatic citation management with multiple styles

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/nadeemshabir/ResearchGPT.git
cd ResearchGPT
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> 💡 **Get a FREE Groq API key**: [console.groq.com](https://console.groq.com) (no credit card required!)

### 5. Run the App

**Windows:**
```bash
run_app.bat
```

**Mac/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Or manually:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage

### 1. Initialize the System
- Click **"🚀 Initialize System"** in the sidebar

### 2. Upload Papers
- Go to **"📄 Upload Papers"** tab
- Upload your PDF research papers
- Click **"🔄 Process Uploaded Papers"**

### 3. Ask Questions
- Go to **"💬 Chat & Ask Questions"** tab
- Type your question and click **"🚀 Ask Question"**

### Example Questions

| Query Type | Example |
|------------|---------|
| **Q&A** | "What is the main contribution of this paper?" |
| **Comparison** | "Compare BERT and GPT architectures" |
| **Literature Review** | "Review papers on attention mechanisms" |
| **Extraction** | "What methodology did the authors use?" |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RESEARCHGPT SYSTEM                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
│  │   INGESTION      │  │    RETRIEVAL     │  │   GENERATION    │   │
│  │   PIPELINE       │→ │     SYSTEM       │→ │     SYSTEM      │   │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘   │
│         ↓                      ↓                      ↓            │
│    PDF → Chunks          Search Chunks         Generate Answer     │
│    Embeddings            Rerank Results        Add Citations       │
│    Store in DB           Return Top-K          Quality Control     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Description |
|--------|-------------|
| **Ingestion** (`src/ingestion/`) | PDF parsing, text chunking, embedding generation, ChromaDB storage |
| **Retrieval** (`src/retrieval/`) | Semantic search, BM25 keyword search, hybrid search, cross-encoder reranking |
| **Generation** (`src/generation/`) | LLM client, multi-agent system, smart routing, citation management |

---

## ⚙️ Configuration

### LLM Providers

ResearchGPT supports multiple LLM providers:

| Provider | Models | API Key Env Variable |
|----------|--------|---------------------|
| **Groq** (default) | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **OpenAI** | `gpt-4o-mini`, `gpt-4` | `OPENAI_API_KEY` |
| **Google Gemini** | `gemini-1.5-flash`, `gemini-1.5-pro` | `GEMINI_API_KEY` |

### Search Settings

- **Semantic Weight**: 0.7 (default) - Importance of meaning-based search
- **Keyword Weight**: 0.3 (default) - Importance of exact keyword matching
- **Top-K**: 10 (default) - Number of chunks to retrieve

---

## 📁 Project Structure

```
ResearchGPT/
├── app.py                    # Streamlit web interface
├── requirements.txt          # Python dependencies
├── .env                      # API keys (create this)
├── run_app.bat              # Windows launcher
├── run_app.sh               # Mac/Linux launcher
│
├── src/
│   ├── ingestion/           # PDF processing pipeline
│   │   ├── pdf_parser.py    # PDF text extraction
│   │   ├── chunker.py       # Text chunking
│   │   ├── embedder.py      # Embedding generation
│   │   ├── database.py      # ChromaDB vector store
│   │   └── pipeline.py      # Ingestion orchestrator
│   │
│   ├── retrieval/           # Search & retrieval
│   │   ├── semantic_search.py
│   │   ├── keyword_search.py
│   │   ├── hybrid_search.py
│   │   ├── reranker.py
│   │   └── retrieval_system.py
│   │
│   └── generation/          # Answer generation
│       ├── llm_client.py    # Multi-provider LLM client
│       ├── agents.py        # Multi-agent system
│       ├── query_router.py  # Smart query routing
│       └── answer_generator.py
│
├── data/raw/                # Store your PDFs here
├── docs/                    # Documentation
└── test/                    # Test files
```

---

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLMs**: Groq/OpenAI/Gemini
- **PDF Processing**: PyMuPDF, pdfplumber, PyPDF2
- **Search**: Hybrid (Semantic + BM25)
- **Reranking**: Cross-Encoder

---

## 📚 Documentation

- [Quick Start Guide](QUICKSTART.md)
- [Smart Routing Guide](docs/SMART_ROUTING_GUIDE.md)
- [Web Interface Guide](docs/WEB_INTERFACE_GUIDE.md)
- [Project Overview](docs/PROJECT_OVERVIEW.md)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for RAG utilities
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for the web interface

---

**Made with ❤️ by [Nadeem Shabir](https://github.com/nadeemshabir)**
