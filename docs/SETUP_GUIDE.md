# 🚀 ResearchGPT - Complete Setup Guide

## ✅ Import Issues - FIXED!

All import path issues have been resolved. The app is ready to run once dependencies are installed.

---

## 📦 Step-by-Step Dependency Installation

### Option 1: Install All at Once (Recommended)

```bash
pip install streamlit python-dotenv pydantic
pip install PyPDF2 pdfplumber pymupdf
pip install langchain langchain-community tiktoken
pip install sentence-transformers torch
pip install chromadb
pip install groq openai google-generativeai
pip install tqdm rank-bm25
```

### Option 2: Install from requirements.txt

```bash
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes as it downloads large packages like PyTorch and sentence-transformers.

---

## 🔧 What Was Fixed

### Import Errors Resolved ✅

**Before** (Broken):
```python
from pdf_parser import PDFParser  # ❌ Doesn't work
```

**After** (Fixed):
```python
from .pdf_parser import PDFParser  # ✅ Works!
```

### Files Modified:
1. ✅ `src/ingestion/pipeline.py` - Relative imports
2. ✅ `src/generation/answer_generator.py` - Relative imports  
3. ✅ `src/retrieval/retrieval_system.py` - Relative imports

---

## 🎯 Quick Start (After Installation)

### 1. Run the App

**Windows:**
```bash
run_app.bat
```

**Mac/Linux:**
```bash
chmod +x run_app.sh
./run_app.sh
```

**Manual:**
```bash
streamlit run app.py
```

### 2. Access the App

Open your browser to:
- **http://localhost:8501** (or 8502, 8503 if port is busy)

### 3. Configure (First Time)

In the sidebar:
1. Select **Groq** as LLM provider (free!)
2. Choose model: **llama-3.3-70b-versatile**
3. Enable features:
   - ☑️ Multi-Agent Pipeline
   - ☑️ Add Citations
   - ☑️ Smart Routing
4. Click **"🚀 Initialize System"**

### 4. Upload Papers

1. Go to **"📄 Upload Papers"** tab
2. Click **"Browse files"**
3. Select PDF research papers
4. Click **"🔄 Process Uploaded Papers"**
5. Wait for processing (progress bar will show)

### 5. Ask Questions

1. Go to **"💬 Chat & Ask Questions"** tab
2. Type your question:
   - "What is this paper about?"
   - "Compare BERT and GPT"
   - "Review papers on transformers"
3. Click **"🚀 Ask Question"**
4. View AI-generated answer with sources!

---

## 🔍 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'X'"

**Solution**: Install the missing module
```bash
pip install X
```

### Common Missing Modules:

| Error | Solution |
|-------|----------|
| `No module named 'pdfplumber'` | `pip install pdfplumber` |
| `No module named 'dotenv'` | `pip install python-dotenv` |
| `No module named 'chromadb'` | `pip install chromadb` |
| `No module named 'sentence_transformers'` | `pip install sentence-transformers` |
| `No module named 'groq'` | `pip install groq` |
| `No module named 'tiktoken'` | `pip install tiktoken` |
| `No module named 'tqdm'` | `pip install tqdm` |

### Error: "No API key found for groq"

**Solution**: Add your API key to `.env` file
```env
GROQ_API_KEY=your_actual_key_here
```

Get a free key from: https://console.groq.com

### Error: "Port 8501 is already in use"

**Solution**: Streamlit will automatically use next available port (8502, 8503, etc.)

Or kill existing process:
```bash
# Windows
taskkill /F /IM streamlit.exe

# Mac/Linux
pkill -f streamlit
```

### App Loads but Shows Errors

**Solution**: Check that you've:
1. ✅ Set `GROQ_API_KEY` in `.env` file
2. ✅ Clicked "Initialize System" in sidebar
3. ✅ Uploaded at least one PDF paper

---

## 📊 Installation Progress

You can track installation progress:

```bash
# Check if a package is installed
pip show streamlit

# List all installed packages
pip list

# Check specific packages
pip list | grep -E "streamlit|chromadb|groq"
```

---

## 🎨 Features Overview

### Smart Query Routing 🎯
Automatically detects:
- **Q&A**: "What is BERT?" → Standard answer
- **Comparison**: "Compare BERT and GPT" → Side-by-side analysis
- **Review**: "Review papers on transformers" → Literature synthesis

### Multi-Agent Generation 🤖
4-stage pipeline:
1. **Extract** - Pull relevant info
2. **Draft** - Generate initial answer
3. **Critique** - Identify improvements
4. **Refine** - Polish final answer

### LLM Provider Support 🔮
- **Groq** (Free, fast) ⚡
- **OpenAI** (GPT-4o, GPT-4) 🤖
- **Google Gemini** (1.5 Flash/Pro) 🔮

### Hybrid Retrieval 🔍
- Semantic search (embeddings)
- Keyword search (BM25)
- Fusion algorithm
- Reranking for relevance

---

## 💡 Tips for Best Results

### 1. Start Simple
- Use Groq (free and fast)
- Start with 1-2 papers
- Try simple Q&A first

### 2. Optimize Settings
- **Top-K**: Start with 5, increase if needed
- **Max Tokens**: Start with 2000, increase for longer context
- **Multi-Agent**: Disable for speed, enable for quality

### 3. Ask Good Questions
- Be specific: "What is the attention mechanism in transformers?"
- Not vague: "Tell me about this"

### 4. Upload Quality Papers
- Use academic PDFs (not scanned images)
- Papers with clear text work best
- Multiple papers enable comparisons

---

## 📁 Project Structure

```
ResearchGPT/
├── app.py                    # ← Main Streamlit UI
├── run_app.bat              # ← Windows launcher
├── run_app.sh               # ← Unix launcher
├── requirements.txt         # ← Dependencies
├── .env                     # ← API keys
│
├── src/
│   ├── ingestion/          # ← PDF processing
│   ├── retrieval/          # ← Search system
│   └── generation/         # ← Answer generation
│
├── data/
│   └── raw/                # ← Uploaded PDFs
│
└── docs/                   # ← Documentation
```

---

## 🔐 Environment Variables

Required in `.env`:

```env
# LLM API Keys (at least one required)
GROQ_API_KEY=your_groq_key_here

# Optional
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Database
CHROMA_DB_PATH=./data/chroma_db
COLLECTION_NAME=research_papers

# Model Settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_TOKENS=4000
DEVICE=cpu
```

---

## ✅ Verification Checklist

Before using the app, verify:

- [ ] All dependencies installed (`pip list`)
- [ ] `.env` file exists with `GROQ_API_KEY`
- [ ] App launches without errors (`streamlit run app.py`)
- [ ] Browser opens to http://localhost:8501
- [ ] Sidebar shows settings
- [ ] Can click "Initialize System" without errors

---

## 🆘 Still Having Issues?

### Check Logs
The terminal where you ran `streamlit run app.py` shows detailed error messages.

### Common Issues:

**1. Import Errors** ✅ FIXED
- All import paths have been corrected

**2. Missing Dependencies** ⚠️ IN PROGRESS
- Install packages as shown above

**3. API Key Errors**
- Check `.env` file
- Verify key is valid on provider's website

**4. Database Errors**
- Delete `.cache/chroma/` folder
- Restart app

---

## 📚 Documentation

- **Quick Start**: `QUICKSTART.md`
- **Web Interface Guide**: `docs/WEB_INTERFACE_GUIDE.md`
- **Integration Summary**: `INTEGRATION_SUMMARY.md`
- **Import Fix Details**: `IMPORT_FIX_APPLIED.md`

---

## 🎉 Success Criteria

You'll know it's working when:

1. ✅ App launches without import errors
2. ✅ Browser shows ResearchGPT interface
3. ✅ "Initialize System" button works
4. ✅ Can upload and process PDFs
5. ✅ Can ask questions and get answers

---

## 🚀 You're Almost There!

**Current Status**: 
- ✅ Import issues fixed
- ⏳ Dependencies installing
- 🎯 Ready to use once installation completes

**Next Step**: Wait for dependency installation to complete, then run `streamlit run app.py`

---

**Happy Researching! 🎓**
