# ResearchGPT Web Interface

A beautiful, feature-rich Streamlit web interface for your ResearchGPT RAG system.

## 🎯 Features

- **📄 PDF Upload**: Upload and process research papers directly through the UI
- **💬 Interactive Chat**: Ask questions about your papers in natural language
- **🎯 Smart Query Routing**: Automatically detects Q&A, comparisons, and literature reviews
- **🤖 Multi-Agent Pipeline**: 4-stage pipeline for high-quality answers
- **📚 Paper Management**: View, organize, and delete uploaded papers
- **⚙️ Flexible Configuration**: Choose your LLM provider (Groq, OpenAI, Gemini)
- **📊 Real-time Statistics**: Track processing times, sources, and confidence scores

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Virtual environment** (recommended)
3. **API Keys** for your chosen LLM provider:
   - Groq: Get from [console.groq.com](https://console.groq.com)
   - OpenAI: Get from [platform.openai.com](https://platform.openai.com)
   - Gemini: Get from [makersuite.google.com](https://makersuite.google.com)

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create or update `.env` file in the project root:
   ```env
   # Choose your LLM provider
   GROQ_API_KEY=your_groq_api_key_here
   # OPENAI_API_KEY=your_openai_api_key_here
   # GOOGLE_API_KEY=your_gemini_api_key_here
   
   # Optional: Customize settings
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **Run the application**:
   
   **Windows**:
   ```bash
   run_app.bat
   ```
   
   **Linux/Mac**:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```
   
   **Or manually**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**:
   The app will automatically open at `http://localhost:8501`

## 📖 How to Use

### Step 1: Configure Settings

1. Open the **sidebar** (left panel)
2. Choose your **LLM provider** (Groq, OpenAI, or Gemini)
3. Select your preferred **model**
4. Enable/disable features:
   - **Multi-Agent Pipeline**: Higher quality but slower
   - **Citations**: Add academic citations to answers
   - **Smart Routing**: Auto-detect query types
5. Adjust **retrieval settings**:
   - **Top-K Results**: Number of chunks to retrieve (3-20)
   - **Max Context Tokens**: Maximum context size (2000-8000)
6. Click **"🚀 Initialize System"**

### Step 2: Upload Papers

1. Go to the **"📄 Upload Papers"** tab
2. Click **"Browse files"** and select PDF research papers
3. Click **"🔄 Process Uploaded Papers"**
4. Wait for processing to complete (you'll see progress bars)

### Step 3: Ask Questions

1. Go to the **"💬 Chat & Ask Questions"** tab
2. Type your question in the text area
3. Click **"🚀 Ask Question"**
4. View the AI-generated answer with sources

**Example Questions**:
- "What is BERT?"
- "Compare BERT and GPT"
- "Review papers on transformers"
- "What are the main findings of this paper?"

### Step 4: Manage Papers

1. Go to the **"📚 View Papers"** tab
2. View all uploaded papers with statistics
3. Delete papers if needed

## 🎨 UI Features

### Smart Query Detection

The system automatically detects:
- **Q&A**: Direct questions → Uses standard answer generation
- **Comparison**: "Compare X and Y" → Uses comparison pipeline
- **Literature Review**: "Review papers on X" → Generates comprehensive review

### Conversation History

- All Q&A interactions are saved in the session
- View previous questions and answers
- See metadata (query type, confidence, sources)
- Export chat history to file

### Real-time Feedback

- Processing time for each query
- Number of sources used
- Confidence scores
- Query type detection

## ⚙️ Configuration Options

### LLM Providers

**Groq** (Recommended for free tier):
- `llama-3.3-70b-versatile` - Fast and capable
- `mixtral-8x7b-32768` - Large context window

**OpenAI**:
- `gpt-4o-mini` - Fast and affordable
- `gpt-4o` - Most capable
- `gpt-4` - Previous generation

**Google Gemini**:
- `gemini-1.5-flash` - Fast and efficient
- `gemini-1.5-pro` - Most capable

### Generation Settings

- **Multi-Agent Pipeline**: 4-stage process (Extract → Draft → Critique → Refine)
- **Simple Generation**: Single-pass generation (faster)
- **Citations**: Inline academic citations
- **Smart Routing**: Automatic query type detection

### Retrieval Settings

- **Top-K**: More chunks = better context but slower
- **Max Context Tokens**: Larger = more context but higher cost

## 🛠️ Troubleshooting

### "System Not Initialized"
- Click the **"🚀 Initialize System"** button in the sidebar
- Make sure your API key is set in `.env`

### "No papers uploaded yet"
- Go to **"📄 Upload Papers"** tab first
- Upload at least one PDF paper

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### API Key Errors
- Check your `.env` file has the correct API key
- Verify the key is valid on the provider's website
- Make sure you've selected the right provider in the UI

### Slow Processing
- Reduce **Top-K** value (try 5 instead of 10)
- Reduce **Max Context Tokens** (try 2000 instead of 4000)
- Disable **Multi-Agent Pipeline** for faster responses

## 📁 Project Structure

```
ResearchGPT/
├── app.py                      # Main Streamlit application
├── run_app.bat                 # Windows launcher
├── run_app.sh                  # Unix/Linux launcher
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
├── src/
│   ├── ingestion/
│   │   ├── pipeline.py         # PDF ingestion pipeline
│   │   ├── pdf_parser.py       # PDF text extraction
│   │   ├── chunker.py          # Text chunking
│   │   ├── embedder.py         # Embedding generation
│   │   └── database.py         # Vector database
│   ├── retrieval/
│   │   └── retrieval_system.py # Hybrid retrieval
│   └── generation/
│       ├── answer_generator.py # Main answer generation
│       ├── llm_client.py       # LLM API client
│       ├── query_router.py     # Smart query routing
│       ├── agents.py           # Multi-agent system
│       └── citation_manager.py # Citation formatting
└── data/
    └── raw/                    # Uploaded PDFs stored here
```

## 🎯 Advanced Usage

### Batch Processing

To process multiple PDFs at once:
1. Place all PDFs in `data/raw/` folder
2. Upload them all at once in the UI
3. Click "Process Uploaded Papers"

### Saving Chat History

1. Ask several questions
2. Click **"💾 Save Chat"** button
3. Find the exported file: `chat_history_YYYYMMDD_HHMMSS.txt`

### Custom Prompts

To customize prompts, edit:
- `src/generation/prompt_templates.py`

### Database Management

View database statistics:
```python
from src.ingestion.database import VectorDatabase
db = VectorDatabase()
db.print_stats()
```

## 🔒 Privacy & Security

- All processing happens **locally** (except LLM API calls)
- PDFs are stored in `data/raw/` on your machine
- Embeddings stored in local ChromaDB (`.cache/chroma/`)
- No data is sent to third parties except LLM providers

## 📊 Performance Tips

1. **Use Groq** for fastest free tier performance
2. **Start with small Top-K** (5) and increase if needed
3. **Disable Multi-Agent** for quick testing
4. **Use smaller models** (gpt-4o-mini, gemini-flash) for speed
5. **Process papers in batches** rather than one-by-one

## 🤝 Contributing

This is part of the ResearchGPT project. See main README for contribution guidelines.

## 📝 License

See main project LICENSE file.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main ResearchGPT documentation
3. Check your API keys and environment setup

---

**Built with ❤️ using Streamlit | ResearchGPT v1.0**
