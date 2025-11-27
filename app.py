try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-title {font-size: 2.5rem; color: #FF4B4B; text-align: center;}
    .sub-header {font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

# --- State Management Functions ---
def reset_application():
    """
    Callback to reset the vector store and chat history 
    when the user changes the uploaded files.
    """
    st.session_state["messages"] = []
    if "retriever" in st.session_state:
        del st.session_state["retriever"]
    if "vector_db" in st.session_state:
        del st.session_state["vector_db"]
    if "processed_files" in st.session_state:
        del st.session_state["processed_files"]

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key
    user_api_key = st.text_input("OpenAI API Key", type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
    
    st.divider()
    
    # 2. File Uploader (Modified for single file processing)
    uploaded_files = st.file_uploader(
        "Upload PDF Documents (one at a time)", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload PDFs one by one. Each file will be processed and added to the knowledge base."
    )
    
    st.divider()
    
    # 3. Control Buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    
    with col2:
        if st.button("Reset All", use_container_width=True):
            reset_application()
            st.rerun()

# --- Backend Logic ---

def process_single_pdf(uploaded_file, existing_vectorstore=None):
    """
    Processes a single PDF file and adds it to the vector store.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    try:
        # Load document
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()
        
        # Add filename metadata to each document
        for doc in docs:
            doc.metadata["source_file"] = uploaded_file.name
        
    finally:
        os.remove(temp_file_path)
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    doc_chunks = splitter.split_documents(docs)

    # Create or update vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if existing_vectorstore is None:
        # Create new vector store
        vectorstore = Chroma.from_documents(
            documents=doc_chunks,
            collection_name="rag_collection",
            embedding=embeddings,
        )
    else:
        # Add to existing vector store
        existing_vectorstore.add_documents(doc_chunks)
        vectorstore = existing_vectorstore
    
    return vectorstore

def create_vector_db(uploaded_files):
    """
    Processes the uploaded files and creates a Chroma Vector Store.
    """
    all_docs = []
    
    # Display progress
    progress_bar = st.progress(0, text="Reading documents...")
    total_files = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            # Add filename metadata
            for doc in docs:
                doc.metadata["source_file"] = uploaded_file.name
            all_docs.extend(docs)
        finally:
            os.remove(temp_file_path)
        
        # Update progress
        progress_bar.progress((i + 1) / total_files, text=f"Processed {i+1}/{total_files} files")

    progress_bar.progress(1.0, text="Splitting text and creating embeddings...")
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    doc_chunks = splitter.split_documents(all_docs)

    # Embed
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    
    progress_bar.empty() # Clear bar when done
    return vectorstore

def get_rag_chain(retriever):
    """Retrieval Augmented Generation Chain"""
    
    rag_prompt = """You are an expert assistant. Answer the question using ONLY the context provided below.
    If the answer is not in the context, say "I don't know based on the documents provided."
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(rag_prompt)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    chain = (
        {"context": (retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

# --- Main Page UI ---

st.markdown('<div class="main-title">RAG QA Chatbot 🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload your PDFs and ask questions about them.</div>', unsafe_allow_html=True)

# 1. Validation Checks
if not user_api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

if not uploaded_files:
    st.info("👈 Please upload PDF documents in the sidebar to start.")
    st.stop()

# 2. Process Documents (Handle incremental uploads)
# Initialize processed files tracker
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()

# Check for new files
current_files = {f.name for f in uploaded_files} if uploaded_files else set()
new_files = current_files - st.session_state["processed_files"]

if new_files:
    # Process new files one by one
    for uploaded_file in uploaded_files:
        if uploaded_file.name in new_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    existing_vectorstore = st.session_state.get("vector_db", None)
                    vector_db = process_single_pdf(uploaded_file, existing_vectorstore)
                    
                    st.session_state["vector_db"] = vector_db
                    st.session_state["retriever"] = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                    st.session_state["processed_files"].add(uploaded_file.name)
                    
                    st.toast(f"✅ {uploaded_file.name} processed successfully!", icon="📄")
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    continue

# Show processed files count
if st.session_state["processed_files"]:
    st.sidebar.success(f"📚 {len(st.session_state['processed_files'])} files processed")
    with st.sidebar.expander("View processed files"):
        for filename in sorted(st.session_state["processed_files"]):
            st.write(f"• {filename}")

# Check if we have a vector store
if "vector_db" not in st.session_state:
    st.info("👈 Please upload PDF documents to start asking questions.")
    st.stop()

# 3. Chat Interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I've analyzed your documents. What would you like to know?"}]

# Display history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle Input
if user_input := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retriever = st.session_state["retriever"]
                chain = get_rag_chain(retriever)
                
                response = chain.invoke(user_input)
                
                st.write(response)
                st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
