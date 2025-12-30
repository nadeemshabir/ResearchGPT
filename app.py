# """
# ResearchGPT - Streamlit User Interface
# A full-featured UI for your RAG system
# """

# import streamlit as st
# import os
# import time
# from pathlib import Path
# from datetime import datetime

# # Import your backend modules
# from src.ingestion.pipeline import IngestionPipeline
# from src.generation.answer_generator import AnswerGenerator

# # Page configuration
# st.set_page_config(
#     page_title="ResearchGPT",
#     page_icon="🎓",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         font-weight: bold;
#         text-align: center;
#         color: #1f77b4;
#         margin-bottom: 1rem;
#     }
#     .sub-header {
#         font-size: 1.2rem;
#         text-align: center;
#         color: #666;
#         margin-bottom: 2rem;
#     }
#     .answer-box {
#         background-color: #f0f2f6;
#         padding: 20px;
#         border-radius: 10px;
#         border-left: 5px solid #1f77b4;
#     }
#     .source-box {
#         background-color: #e8f4f8;
#         padding: 15px;
#         border-radius: 8px;
#         margin: 10px 0;
#     }
#     .metric-card {
#         background-color: #ffffff;
#         padding: 15px;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#     }
#     .stButton>button {
#         width: 100%;
#         border-radius: 8px;
#         height: 3em;
#         font-weight: 600;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# if 'generator' not in st.session_state:
#     st.session_state.generator = None
# if 'ingestion_pipeline' not in st.session_state:
#     st.session_state.ingestion_pipeline = None
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'uploaded_papers' not in st.session_state:
#     st.session_state.uploaded_papers = []
# if 'initialized' not in st.session_state:
#     st.session_state.initialized = False

# # Sidebar for settings
# with st.sidebar:
#     st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
#     st.title("⚙️ Settings")
    
#     st.markdown("---")
    
#     # LLM Configuration
#     st.subheader("🤖 LLM Configuration")
#     llm_provider = st.selectbox(
#         "Provider",
#         ["groq", "openai", "gemini"],
#         index=0,
#         help="Choose your LLM provider"
#     )
    
#     # Model selection based on provider
#     if llm_provider == "groq":
#         llm_model = st.selectbox(
#             "Model",
#             ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
#             help="Groq models (fast & free tier available)"
#         )
#     elif llm_provider == "openai":
#         llm_model = st.selectbox(
#             "Model",
#             ["gpt-4o-mini", "gpt-4o", "gpt-4"],
#             help="OpenAI models"
#         )
#     else:  # gemini
#         llm_model = st.selectbox(
#             "Model",
#             ["gemini-1.5-flash", "gemini-1.5-pro"],
#             help="Google Gemini models"
#         )
    
#     st.markdown("---")
    
#     # Generation Settings
#     st.subheader("🎯 Generation Settings")
#     use_multi_agent = st.checkbox(
#         "Multi-Agent Pipeline",
#         value=True,
#         help="Use 4-stage pipeline for higher quality (slower)"
#     )
    
#     use_citations = st.checkbox(
#         "Add Citations",
#         value=True,
#         help="Include academic citations in answers"
#     )
    
#     use_smart_routing = st.checkbox(
#         "Smart Routing",
#         value=True,
#         help="Auto-detect query type (Q&A, comparison, review)"
#     )
    
#     st.markdown("---")
    
#     # Retrieval Settings
#     st.subheader("🔍 Retrieval Settings")
#     top_k = st.slider(
#         "Top-K Results",
#         min_value=3,
#         max_value=20,
#         value=10,
#         help="Number of chunks to retrieve"
#     )
    
#     max_context_tokens = st.slider(
#         "Max Context Tokens",
#         min_value=2000,
#         max_value=8000,
#         value=4000,
#         step=500,
#         help="Maximum context size for LLM"
#     )
    
#     st.markdown("---")
    
#     # Initialize button
#     if st.button("🚀 Initialize System", type="primary"):
#         with st.spinner("Initializing ResearchGPT..."):
#             try:
#                 # Initialize ingestion pipeline
#                 st.session_state.ingestion_pipeline = IngestionPipeline()
                
#                 # Initialize answer generator
#                 st.session_state.generator = AnswerGenerator(
#                     use_multi_agent=use_multi_agent,
#                     use_citations=use_citations,
#                     llm_provider=llm_provider,
#                     llm_model=llm_model,
#                     use_smart_routing=use_smart_routing
#                 )
                
#                 st.session_state.initialized = True
#                 st.success("✅ System initialized successfully!")
#                 time.sleep(1)
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"❌ Initialization failed: {str(e)}")
    
#     st.markdown("---")
    
#     # System status
#     st.subheader("📊 System Status")
#     if st.session_state.initialized:
#         st.success("🟢 Ready")
#         st.metric("Papers Uploaded", len(st.session_state.uploaded_papers))
#         st.metric("Conversations", len(st.session_state.chat_history))
#     else:
#         st.warning("🟡 Not Initialized")
#         st.info("👆 Click 'Initialize System' to start")

# # Main content area
# st.markdown('<div class="main-header">🎓 ResearchGPT</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-header">AI-Powered Research Paper Analysis</div>', unsafe_allow_html=True)

# # Check if system is initialized
# if not st.session_state.initialized:
#     st.info("👈 Please configure settings in the sidebar and click 'Initialize System' to begin")
    
#     # Show features
#     st.markdown("---")
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("### 📄 Upload Papers")
#         st.write("Upload PDF research papers and let AI analyze them")
    
#     with col2:
#         st.markdown("### 💬 Ask Questions")
#         st.write("Ask anything in natural language and get AI-generated answers")
    
#     with col3:
#         st.markdown("### 🎯 Smart Routing")
#         st.write("Automatically detects Q&A, comparisons, and literature reviews")
    
#     st.stop()

# # Tabs for different functionalities
# tab1, tab2, tab3 = st.tabs(["📄 Upload Papers", "💬 Chat & Ask Questions", "📚 View Papers"])

# # TAB 1: Upload Papers
# with tab1:
#     st.header("📄 Upload Research Papers")
#     st.write("Upload PDF files to add them to your knowledge base")
    
#     uploaded_files = st.file_uploader(
#         "Choose PDF files",
#         type=['pdf'],
#         accept_multiple_files=True,
#         help="Upload one or more PDF research papers"
#     )
    
#     if uploaded_files:
#         if st.button("🔄 Process Uploaded Papers", type="primary"):
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             for idx, uploaded_file in enumerate(uploaded_files):
#                 # Update progress
#                 progress = (idx + 1) / len(uploaded_files)
#                 progress_bar.progress(progress)
#                 status_text.text(f"Processing {uploaded_file.name}...")
                
#                 try:
#                     # Save uploaded file temporarily
#                     temp_path = Path("data/raw") / uploaded_file.name
#                     temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
#                     with open(temp_path, "wb") as f:
#                         f.write(uploaded_file.getbuffer())
                    
#                     # Process the paper
#                     stats = st.session_state.ingestion_pipeline.process_paper(str(temp_path))
                    
#                     # Store in session state
#                     st.session_state.uploaded_papers.append({
#                         'filename': uploaded_file.name,
#                         'paper_id': stats['paper_id'],
#                         'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'stats': stats
#                     })
                    
#                     st.success(f"✅ {uploaded_file.name} processed successfully!")
#                     st.json(stats)
                    
#                 except Exception as e:
#                     st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
#             progress_bar.progress(1.0)
#             status_text.text("✅ All papers processed!")
#             time.sleep(2)
#             st.rerun()

# # TAB 2: Chat & Ask Questions
# with tab2:
#     st.header("💬 Ask Questions About Your Papers")
    
#     if len(st.session_state.uploaded_papers) == 0:
#         st.warning("⚠️ No papers uploaded yet. Please upload papers in the 'Upload Papers' tab first.")
#     else:
#         # Display chat history
#         st.subheader("📝 Conversation History")
        
#         for chat in st.session_state.chat_history:
#             with st.container():
#                 st.markdown(f"**🧑 You:** {chat['question']}")
#                 st.markdown(f"**🤖 ResearchGPT:** {chat['answer']}")
                
#                 # Show metadata
#                 with st.expander("ℹ️ Details"):
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Query Type", chat['metadata'].get('query_type', 'N/A'))
#                     with col2:
#                         st.metric("Confidence", f"{chat['metadata'].get('routing_confidence', 0):.2%}")
#                     with col3:
#                         st.metric("Sources", chat['metadata'].get('num_sources', 0))
                    
#                     if chat.get('sources'):
#                         st.markdown("**📚 Sources:**")
#                         for source in chat['sources'][:3]:  # Show top 3
#                             st.markdown(f"- {source.get('title', 'Unknown')}")
                
#                 st.markdown("---")
        
#         # Input area
#         st.subheader("🔍 Ask a New Question")
        
#         query_input = st.text_area(
#             "Your question:",
#             placeholder="Example: What is BERT?\nCompare BERT and GPT\nReview papers on transformers",
#             height=100,
#             help="Ask anything! The system will auto-detect the type of question."
#         )
        
#         col1, col2, col3 = st.columns([2, 1, 1])
        
#         with col1:
#             ask_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
        
#         with col2:
#             clear_button = st.button("🗑️ Clear History", use_container_width=True)
        
#         with col3:
#             if st.button("💾 Save Chat", use_container_width=True):
#                 # Save chat history to file
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 filename = f"chat_history_{timestamp}.txt"
                
#                 with open(filename, "w", encoding="utf-8") as f:
#                     for chat in st.session_state.chat_history:
#                         f.write(f"Q: {chat['question']}\n")
#                         f.write(f"A: {chat['answer']}\n")
#                         f.write("-" * 80 + "\n\n")
                
#                 st.success(f"✅ Chat saved to {filename}")
        
#         if clear_button:
#             st.session_state.chat_history = []
#             st.rerun()
        
#         if ask_button and query_input:
#             with st.spinner("🤔 Thinking..."):
#                 try:
#                     start_time = time.time()
                    
#                     # Generate answer using smart routing
#                     response = st.session_state.generator.smart_answer(
#                         user_query=query_input,
#                         top_k=top_k,
#                         max_context_tokens=max_context_tokens
#                     )
                    
#                     processing_time = time.time() - start_time
                    
#                     # Add to chat history
#                     st.session_state.chat_history.append({
#                         'question': query_input,
#                         'answer': response['answer'],
#                         'sources': response.get('sources', []),
#                         'metadata': response.get('metadata', {}),
#                         'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         'processing_time': processing_time
#                     })
                    
#                     st.rerun()
                    
#                 except Exception as e:
#                     st.error(f"❌ Error generating answer: {str(e)}")
#                     st.exception(e)

# # TAB 3: View Papers
# with tab3:
#     st.header("📚 Uploaded Papers")
    
#     if len(st.session_state.uploaded_papers) == 0:
#         st.info("No papers uploaded yet")
#     else:
#         st.write(f"Total papers: **{len(st.session_state.uploaded_papers)}**")
        
#         for paper in st.session_state.uploaded_papers:
#             with st.expander(f"📄 {paper['filename']}"):
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.markdown(f"**Paper ID:** `{paper['paper_id']}`")
#                     st.markdown(f"**Upload Time:** {paper['upload_time']}")
                
#                 with col2:
#                     stats = paper['stats']
#                     st.markdown(f"**Pages:** {stats.get('num_pages', 'N/A')}")
#                     st.markdown(f"**Chunks:** {stats.get('num_chunks', 'N/A')}")
#                     st.markdown(f"**Processing Time:** {stats.get('processing_time_seconds', 0):.2f}s")
                
#                 if st.button(f"🗑️ Delete {paper['filename']}", key=f"delete_{paper['paper_id']}"):
#                     # Remove from list
#                     st.session_state.uploaded_papers.remove(paper)
#                     st.success(f"Removed {paper['filename']}")
#                     st.rerun()

# # Footer
# st.markdown("---")
# st.markdown(
#     "<div style='text-align: center; color: #666;'>"
#     "Built with ❤️ using Streamlit | ResearchGPT v1.0"
#     "</div>",
#     unsafe_allow_html=True
# )

"""
ResearchGPT - Streamlit User Interface
A full-featured UI for your RAG system
"""

import streamlit as st
import os
import time
from pathlib import Path
from datetime import datetime

# Import your backend modules
from src.ingestion.pipeline import IngestionPipeline
from src.generation.answer_generator import AnswerGenerator

# Page configuration
st.set_page_config(
    page_title="ResearchGPT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'ingestion_pipeline' not in st.session_state:
    st.session_state.ingestion_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_papers' not in st.session_state:
    st.session_state.uploaded_papers = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar for settings
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    # LLM Configuration
    st.subheader("🤖 LLM Configuration")
    llm_provider = st.selectbox(
        "Provider",
        ["groq", "openai", "gemini"],
        index=0,
        help="Choose your LLM provider"
    )
    
    # Model selection based on provider
    if llm_provider == "groq":
        llm_model = st.selectbox(
            "Model",
            ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
            help="Groq models (fast & free tier available)"
        )
    elif llm_provider == "openai":
        llm_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4"],
            help="OpenAI models"
        )
    else:  # gemini
        llm_model = st.selectbox(
            "Model",
            ["gemini-1.5-flash", "gemini-1.5-pro"],
            help="Google Gemini models"
        )
    
    st.markdown("---")
    
    # Generation Settings
    st.subheader("🎯 Generation Settings")
    use_multi_agent = st.checkbox(
        "Multi-Agent Pipeline",
        value=True,
        help="Use 4-stage pipeline for higher quality (slower)"
    )
    
    use_citations = st.checkbox(
        "Add Citations",
        value=True,
        help="Include academic citations in answers"
    )
    
    use_smart_routing = st.checkbox(
        "Smart Routing",
        value=True,
        help="Auto-detect query type (Q&A, comparison, review)"
    )
    
    st.markdown("---")
    
    # Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider(
        "Top-K Results",
        min_value=3,
        max_value=20,
        value=10,
        help="Number of chunks to retrieve"
    )
    
    max_context_tokens = st.slider(
        "Max Context Tokens",
        min_value=2000,
        max_value=8000,
        value=4000,
        step=500,
        help="Maximum context size for LLM"
    )
    
    st.markdown("---")
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing ResearchGPT..."):
            try:
                # Initialize ingestion pipeline
                st.session_state.ingestion_pipeline = IngestionPipeline()
                
                # Initialize answer generator
                st.session_state.generator = AnswerGenerator(
                    use_multi_agent=use_multi_agent,
                    use_citations=use_citations,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    use_smart_routing=use_smart_routing
                )
                
                st.session_state.initialized = True
                st.success("✅ System initialized successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
    
    st.markdown("---")
    
    # System status
    st.subheader("📊 System Status")
    if st.session_state.initialized:
        st.success("🟢 Ready")
        st.metric("Papers Uploaded", len(st.session_state.uploaded_papers))
        st.metric("Conversations", len(st.session_state.chat_history))
    else:
        st.warning("🟡 Not Initialized")
        st.info("👆 Click 'Initialize System' to start")

# Main content area
st.markdown('<div class="main-header">🎓 ResearchGPT</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Research Paper Analysis</div>', unsafe_allow_html=True)

# Check if system is initialized
if not st.session_state.initialized:
    st.info("👈 Please configure settings in the sidebar and click 'Initialize System' to begin")
    
    # Show features
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 Upload Papers")
        st.write("Upload PDF research papers and let AI analyze them")
    
    with col2:
        st.markdown("### 💬 Ask Questions")
        st.write("Ask anything in natural language and get AI-generated answers")
    
    with col3:
        st.markdown("### 🎯 Smart Routing")
        st.write("Automatically detects Q&A, comparisons, and literature reviews")
    
    st.stop()

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📄 Upload Papers", "💬 Chat & Ask Questions", "📚 View Papers"])

# TAB 1: Upload Papers
with tab1:
    st.header("📄 Upload Research Papers")
    st.write("Upload PDF files to add them to your knowledge base")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Uploaded Papers", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                # Update progress
                progress = (idx + 1) / len(uploaded_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Save uploaded file temporarily
                    temp_path = Path("data/raw") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Process the paper
                    stats = st.session_state.ingestion_pipeline.process_paper(str(temp_path))
                    
                    # Store in session state
                    st.session_state.uploaded_papers.append({
                        'filename': uploaded_file.name,
                        'paper_id': stats['paper_id'],
                        'upload_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'stats': stats
                    })
                    
                    st.success(f"✅ {uploaded_file.name} processed successfully!")
                    st.json(stats)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ All papers processed!")
            time.sleep(2)
            st.rerun()

# TAB 2: Chat & Ask Questions
with tab2:
    st.header("💬 Ask Questions About Your Papers")
    
    if len(st.session_state.uploaded_papers) == 0:
        st.warning("⚠️ No papers uploaded yet. Please upload papers in the 'Upload Papers' tab first.")
    else:
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📝 Conversation History")
            
            for chat in st.session_state.chat_history:
                with st.container():
                    st.markdown(f"**🧑 You:** {chat['question']}")
                    st.markdown(f"**🤖 ResearchGPT:** {chat['answer']}")
                    
                    # Show metadata
                    with st.expander("ℹ️ Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Query Type", chat['metadata'].get('query_type', 'N/A'))
                        with col2:
                            st.metric("Confidence", f"{chat['metadata'].get('routing_confidence', 0):.2%}")
                        with col3:
                            st.metric("Sources", chat['metadata'].get('num_sources', 0))
                        
                        if chat.get('sources'):
                            st.markdown("**📚 Sources:**")
                            for source in chat['sources'][:3]:  # Show top 3
                                st.markdown(f"- {source.get('title', 'Unknown')}")
                    
                    st.markdown("---")
            
            # Action buttons for chat history
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ Clear History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Chat to File", use_container_width=True):
                    # Save chat history to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"chat_history_{timestamp}.txt"
                    
                    with open(filename, "w", encoding="utf-8") as f:
                        for chat in st.session_state.chat_history:
                            f.write(f"Q: {chat['question']}\n")
                            f.write(f"A: {chat['answer']}\n")
                            f.write("-" * 80 + "\n\n")
                    
                    st.success(f"✅ Chat saved to {filename}")
            
            st.markdown("---")
        
        # Chat input
        st.subheader("🔍 Ask a Question")
        
        # Use st.text_input (works inside tabs, supports Enter to submit)
        query_input = st.text_input(
            "Your question:",
            placeholder="Ask anything: What is BERT? | Compare BERT and GPT | Review papers on transformers",
            key="chat_input"
        )
        
        # Process query when submitted (Enter pressed or send button clicked)
        if query_input:
            with st.spinner("🤔 Thinking..."):
                try:
                    start_time = time.time()
                    
                    # Generate answer using smart routing
                    response = st.session_state.generator.smart_answer(
                        user_query=query_input,
                        top_k=top_k,
                        max_context_tokens=max_context_tokens
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': query_input,
                        'answer': response['answer'],
                        'sources': response.get('sources', []),
                        'metadata': response.get('metadata', {}),
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'processing_time': processing_time
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")
                    st.exception(e)

# TAB 3: View Papers
with tab3:
    st.header("📚 Uploaded Papers")
    
    if len(st.session_state.uploaded_papers) == 0:
        st.info("No papers uploaded yet")
    else:
        st.write(f"Total papers: **{len(st.session_state.uploaded_papers)}**")
        
        for paper in st.session_state.uploaded_papers:
            with st.expander(f"📄 {paper['filename']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Paper ID:** `{paper['paper_id']}`")
                    st.markdown(f"**Upload Time:** {paper['upload_time']}")
                
                with col2:
                    stats = paper['stats']
                    st.markdown(f"**Pages:** {stats.get('num_pages', 'N/A')}")
                    st.markdown(f"**Chunks:** {stats.get('num_chunks', 'N/A')}")
                    st.markdown(f"**Processing Time:** {stats.get('processing_time_seconds', 0):.2f}s")
                
                if st.button(f"🗑️ Delete {paper['filename']}", key=f"delete_{paper['paper_id']}"):
                    try:
                        # Remove from database
                        from src.ingestion.database import VectorDatabase
                        db = VectorDatabase()
                        db.delete_paper(paper['paper_id'])
                        
                        # Remove from list
                        st.session_state.uploaded_papers.remove(paper)
                        st.success(f"✅ Removed {paper['filename']} from database and list")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using Streamlit | ResearchGPT v1.0"
    "</div>",
    unsafe_allow_html=True
)