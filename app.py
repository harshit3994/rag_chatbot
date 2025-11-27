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

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. API Key
    user_api_key = st.text_input("OpenAI API Key", type="password")
    if user_api_key:
        os.environ["OPENAI_API_KEY"] = user_api_key
    
    st.divider()
    
    # 2. File Uploader (Linked to reset callback)
    uploaded_files = st.file_uploader(
        "Upload PDF Documents", 
        type=["pdf"], 
        accept_multiple_files=True,
        on_change=reset_application # <--- The Critical Fix
    )
    
    st.divider()
    
    # 3. Clear Chat Button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- Backend Logic ---

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

# 2. Process Documents (Only if not already processed)
if "vector_db" not in st.session_state:
    with st.spinner("Processing documents..."):
        try:
            vector_db = create_vector_db(uploaded_files)
            st.session_state["vector_db"] = vector_db
            st.session_state["retriever"] = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            st.toast("Documents processed successfully!", icon="✅")
        except Exception as e:
            st.error(f"Error processing documents: {e}")
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
