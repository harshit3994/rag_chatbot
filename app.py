try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------------

import streamlit as st
import os
import tempfile
import time

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Page Config ---
st.set_page_config(
    page_title="DocuChat AI", 
    page_icon="📚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for clean look ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4A90E2;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-text {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- State Management Helper ---
def reset_conversation():
    """Resets the chat history and vector store when new files are uploaded."""
    st.session_state["messages"] = []
    st.session_state["vector_db"] = None
    if "retriever" in st.session_state:
        del st.session_state["retriever"]

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/book.png", width=80)
    st.title("Settings")
    
    # 1. API Key
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("API Key accepted", icon="✅")
    else:
        st.warning("Please enter API Key", icon="⚠️")

    st.divider()

    # 2. File Uploader with on_change callback
    st.subheader("Document Source")
    uploaded_files = st.file_uploader(
        "Upload PDFs", 
        type=["pdf"], 
        accept_multiple_files=True,
        on_change=reset_conversation # Crucial for UX: Reset if files change
    )

    st.divider()
    
    # 3. Actions
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- Backend Logic (Cached) ---

@st.cache_resource(show_spinner=False)
def create_vector_db(uploaded_files):
    """
    Processes PDFs and returns a Chroma vector store. 
    Cached to prevent re-processing on every interaction.
    """
    all_docs = []
    
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress bar
        my_bar.progress((idx + 1) / total_files, text=f"Processing file {idx+1} of {total_files}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
        finally:
            os.remove(temp_file_path)

    my_bar.progress(1.0, text="Splitting documents...")
    
    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    doc_chunks = splitter.split_documents(all_docs)

    my_bar.progress(1.0, text="Creating embeddings...")
    
    # Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    
    my_bar.empty() # Remove progress bar
    return vectorstore

def get_rag_chain(retriever):
    rag_prompt = """You are a helpful assistant. Answer the question using only the context below.
    If the answer is not in the context, say you don't know.
    
    Context: {context}
    
    Question: {question}
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

# --- Main Interface ---

st.markdown('<div class="main-header">DocuChat AI 📚</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Chat with your PDF documents seamlessly.</div>', unsafe_allow_html=True)

# Checks
if not api_key:
    st.info("👈 Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

if not uploaded_files:
    st.info("👈 Please upload a PDF document in the sidebar to start.")
    st.stop()

# Initialize Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm ready to answer questions about your documents."}]

# Process Documents (Only runs if not cached or files changed)
if "vector_db" not in st.session_state or st.session_state["vector_db"] is None:
    with st.spinner("Analyzing documents..."):
        try:
            vector_db = create_vector_db(uploaded_files)
            st.session_state["vector_db"] = vector_db
            st.session_state["retriever"] = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            st.toast("Documents processed successfully!", icon="🎉")
        except Exception as e:
            st.error(f"Error processing documents: {e}")
            st.stop()

# Display Chat History
chat_container = st.container()
with chat_container:
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

# User Input
if user_input := st.chat_input("Ask a question about your PDF..."):
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
