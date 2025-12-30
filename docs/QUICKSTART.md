# 🚀 Quick Start Guide - ResearchGPT Web Interface

Get up and running with ResearchGPT in 5 minutes!

## ⚡ Super Quick Start

### 1. Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key (30 seconds)

Open `.env` file and add your Groq API key:

```env
GROQ_API_KEY=your_key_here
```

**Get a FREE Groq API key**: [console.groq.com](https://console.groq.com) (no credit card required!)

### 3. Run the App (10 seconds)

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

### 4. Use the App (3 minutes)

1. **Configure** (in sidebar):
   - Click "🚀 Initialize System"
   
2. **Upload Papers**:
   - Go to "📄 Upload Papers" tab
   - Upload a PDF research paper
   - Click "🔄 Process Uploaded Papers"
   
3. **Ask Questions**:
   - Go to "💬 Chat & Ask Questions" tab
   - Type: "What is this paper about?"
   - Click "🚀 Ask Question"

**Done! 🎉** You're now using AI to analyze research papers!

---

## 📝 Example Questions to Try

Once you've uploaded a paper, try these:

### Simple Q&A
- "What is the main contribution of this paper?"
- "What methodology did the authors use?"
- "What are the key findings?"

### Comparisons (if you have multiple papers)
- "Compare BERT and GPT"
- "What's the difference between these two approaches?"

### Literature Reviews
- "Review papers on transformers"
- "Summarize research on attention mechanisms"

---

## 🎯 Tips for Best Results

1. **Start with Groq** - It's free and fast!
2. **Use PDF papers** - Works best with academic PDFs
3. **Be specific** - Ask clear, focused questions
4. **Try different settings** - Adjust Top-K and context size

---

## ❓ Troubleshooting

### "No API key found"
- Check your `.env` file has `GROQ_API_KEY=your_actual_key`
- Make sure there are no spaces around the `=`

### "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### "No papers uploaded"
- Go to "📄 Upload Papers" tab first
- Upload at least one PDF

### App won't start
```bash
# Reinstall streamlit
pip install streamlit --upgrade
```

---

## 🔥 Next Steps

Once you're comfortable with the basics:

1. **Try different LLM providers** (OpenAI, Gemini)
2. **Enable Multi-Agent Pipeline** for higher quality
3. **Upload multiple papers** for comparisons
4. **Adjust retrieval settings** for better context

---

## 📚 Full Documentation

For detailed information, see:
- [Web Interface Guide](docs/WEB_INTERFACE_GUIDE.md)
- [Smart Routing Guide](docs/SMART_ROUTING_GUIDE.md)
- [Main README](README.md)

---

**Need Help?** Check the troubleshooting section above or review the full documentation.

**Happy Researching! 🎓**
