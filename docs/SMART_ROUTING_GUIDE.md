# Smart Query Routing - Complete Guide

## 🎯 Overview

The ResearchGPT system now has **intelligent query routing** that automatically detects the type of question and routes it to the appropriate generation method.

---

## 📋 How It Works

### **Before (Manual):**
```python
generator = AnswerGenerator()

# You had to choose the right method manually
response1 = generator.answer_question("What is BERT?")
response2 = generator.compare_papers(["BERT", "GPT"])
response3 = generator.generate_literature_review("Transformers")
```

### **Now (Automatic):**
```python
generator = AnswerGenerator(use_smart_routing=True)

# Just use smart_answer() for everything!
response1 = generator.smart_answer("What is BERT?")
# ↑ Auto-detects: Q&A

response2 = generator.smart_answer("Compare BERT and GPT")
# ↑ Auto-detects: Comparison

response3 = generator.smart_answer("Review all papers on transformers")
# ↑ Auto-detects: Literature Review
```

---

## 🤖 Query Types Detected

The `QueryRouter` can detect 6 types of queries:

| Query Type | Examples | Routes To |
|------------|----------|-----------|
| **Q&A** | "What is X?", "How does Y work?" | `answer_question()` |
| **Comparison** | "Compare X and Y", "X vs Y", "Difference between..." | `compare_papers()` |
| **Literature Review** | "Review papers on X", "Overview of Y", "State of the art" | `generate_literature_review()` |
| **Extraction** | "List all methods", "Extract findings" | `answer_question()` (extract mode) |
| **Definition** | "What is BERT?", "Define attention" | `answer_question()` (definition mode) |
| **Summary** | "Summarize main points", "Key findings" | `answer_question()` (summary mode) |

---

## 🔍 Detection Patterns

### **Comparison Detection:**
- Keywords: `compare`, `versus`, `vs`, `difference`, `contrast`, `similar`
- Patterns:
  - "Compare X and Y"
  - "X vs Y"
  - "Difference between X and Y"
  - "X versus Y"
- Extracts items automatically: `["X", "Y"]`

### **Literature Review Detection:**
- Keywords: `review`, `overview`, `survey`, `state of the art`, `recent advances`
- Patterns:
  - "Review all papers on X"
  - "Overview of Y"
  - "State of the art in Z"
  - "Recent work on X"

### **Q&A Detection (Default):**
- Any question not matching other patterns
- "What is...", "How does...", "Why..."

---

## 💻 Usage in Your UI

### **Option 1: Fully Automatic (Recommended)**

```python
from src.generation.answer_generator import AnswerGenerator

# Initialize once
generator = AnswerGenerator(use_smart_routing=True)

# In your UI backend
def handle_user_query(user_input: str):
    """Handle any type of query automatically"""
    response = generator.smart_answer(user_input)
    return response['answer']

# Examples:
handle_user_query("What is BERT?")
# → Automatically uses Q&A

handle_user_query("Compare BERT and GPT")
# → Automatically uses comparison

handle_user_query("Review papers on transformers")
# → Automatically uses literature review
```

### **Option 2: With Manual Override**

```python
def handle_user_query(user_input: str, force_mode: str = None):
    """
    Handle query with optional manual override
    
    Args:
        user_input: User's question
        force_mode: Optional override ('qa', 'compare', 'review', None for auto)
    """
    generator = AnswerGenerator(use_smart_routing=True)
    
    if force_mode == 'qa':
        response = generator.answer_question(user_input)
    elif force_mode == 'compare':
        # Extract items from query
        items = extract_items(user_input)  # Your extraction logic
        response = generator.compare_papers(items)
    elif force_mode == 'review':
        response = generator.generate_literature_review(user_input)
    else:
        # Auto-detect
        response = generator.smart_answer(user_input)
    
    return response
```

---

## 🎨 UI Examples

### **Streamlit Example:**

```python
import streamlit as st
from src.generation.answer_generator import AnswerGenerator

# Initialize
if 'generator' not in st.session_state:
    st.session_state.generator = AnswerGenerator(use_smart_routing=True)

st.title("🎓 ResearchGPT")

# User input
user_query = st.text_input("Ask anything about your research papers:")

if st.button("Ask 🔍"):
    if user_query:
        with st.spinner("Thinking..."):
            # Automatic routing!
            response = st.session_state.generator.smart_answer(user_query)
            
            # Show detected type
            st.info(f"📊 Detected: {response['metadata'].get('query_type', 'Q&A')}")
            
            # Show answer
            st.markdown("### Answer:")
            st.write(response['answer'])
            
            # Show sources
            if response.get('sources'):
                with st.expander("📚 Sources"):
                    for source in response['sources']:
                        st.write(f"- {source['title']}")
```

### **Flask API Example:**

```python
from flask import Flask, request, jsonify
from src.generation.answer_generator import AnswerGenerator

app = Flask(__name__)
generator = AnswerGenerator(use_smart_routing=True)

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """API endpoint for asking questions"""
    data = request.json
    user_query = data.get('question', '')
    
    if not user_query:
        return jsonify({'error': 'No question provided'}), 400
    
    # Automatic routing!
    response = generator.smart_answer(user_query)
    
    return jsonify({
        'answer': response['answer'],
        'query_type': response['metadata'].get('query_type'),
        'confidence': response['metadata'].get('routing_confidence'),
        'sources': response.get('sources', [])
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 📊 Response Format

All responses from `smart_answer()` have a consistent format:

```python
{
    'answer': str,              # The generated answer
    'question': str,            # Original question
    'sources': List[Dict],      # Source papers/chunks
    'metadata': {
        'query_type': str,      # Detected type
        'routing_confidence': float,  # 0-1 confidence
        'num_sources': int,
        'processing_time': float,
        # ... other metadata
    }
}
```

---

## 🔧 Configuration

### **Enable/Disable Smart Routing:**

```python
# With smart routing (default)
generator = AnswerGenerator(use_smart_routing=True)

# Without smart routing (always uses Q&A)
generator = AnswerGenerator(use_smart_routing=False)
```

### **Adjust Detection Confidence:**

The router returns a confidence score (0-1). You can use it:

```python
response = generator.smart_answer(user_query)

confidence = response['metadata']['routing_confidence']

if confidence < 0.5:
    # Low confidence, maybe ask user to clarify
    print("I'm not sure what type of question this is. Can you rephrase?")
else:
    # High confidence, proceed
    print(response['answer'])
```

---

## ✅ Benefits

1. **User-Friendly**: Users don't need to know about different modes
2. **Automatic**: No manual routing logic needed
3. **Flexible**: Can still use specific methods if needed
4. **Transparent**: Shows detected type and confidence
5. **Fallback**: Defaults to Q&A if uncertain

---

## 🧪 Testing

Test the router:

```bash
# Test query router
python src/generation/query_router.py

# Test smart answer generation
python src/generation/smart_answer_example.py
```

---

## 📝 Example Queries

### **Q&A:**
- "What is BERT?"
- "How does attention mechanism work?"
- "Explain transformer architecture"

### **Comparison:**
- "Compare BERT and GPT"
- "BERT vs GPT"
- "What is the difference between RNN and LSTM?"
- "How does BERT differ from ELMo?"

### **Literature Review:**
- "Review all papers on transformers"
- "Give me an overview of attention mechanisms"
- "State of the art in NLP"
- "Recent advances in computer vision"

---

## 🎯 Summary

**For Your UI:**
1. Initialize: `generator = AnswerGenerator(use_smart_routing=True)`
2. Use: `response = generator.smart_answer(user_query)`
3. Done! The system handles everything automatically.

**No need to:**
- ❌ Ask users to select a mode
- ❌ Write routing logic
- ❌ Parse queries manually

**The system automatically:**
- ✅ Detects query type
- ✅ Routes to appropriate method
- ✅ Returns consistent response format
- ✅ Provides confidence scores

---

**You're all set!** 🚀
