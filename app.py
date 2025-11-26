try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # If pysqlite3 is not installed (e.g., running locally), pass.
    pass
# -----------------------------------------

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
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

# --- UI Setup ---
st.title("RAG QA Chatbot 🤖")
st.markdown("Upload **one or more** PDF documents and ask questions based on their content.")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Configuration")
    
    # 1. API Key Input
    user_api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.divider()
    
    # 2. File Uploader (Multiple Files Allowed)
    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# --- Backend Logic ---

def process_documents(uploaded_files):
    """
    Handles saving, loading, splitting, and vector storage for MULTIPLE files.
    """
    all_docs = []
    
    # 1. Loop through each uploaded file
    for uploaded_file in uploaded_files:
        # Create a temp file for each upload so PyMuPDF can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            # Load the document
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs) # Collect pages from all files
        finally:
            # Clean up the temp file
            os.remove(temp_file_path)

    # 2. Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    doc_chunks = splitter.split_documents(all_docs)

    # 3. Create Embeddings & Vector Store
    # We use an in-memory vector store for the session
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Chroma.from_documents(
        documents=doc_chunks,
        collection_name="rag_collection",
        embedding=embeddings,
    )
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def get_rag_chain(retriever):
    """
    Creates the Retrieval-Augmented Generation chain.
    """
    rag_prompt = """You are an assistant who is an expert in question-answering tasks.
              Answer the following question using only the following pieces of retrieved context.
              If the answer is not in the context, do not make up answers, just say that you don't know.
              Keep the answer detailed and well formatted based on the information from the context.
              Please ensure the language and grammar would be correct.

              Question:
              {question}

              Context:
              {context}

              Answer:
              """
    
    rag_prompt_template = ChatPromptTemplate.from_template(rag_prompt)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": (retriever | format_docs), "question": RunnablePassthrough()}
        | rag_prompt_template
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- Main App Execution ---

# 1. Check API Key
if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key
elif "OPENAI_API_KEY" not in os.environ:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# 2. Handle File Processing
if uploaded_files:
    # Use session state to avoid re-processing on every message
    if "vectors_processed" not in st.session_state:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            try:
                retriever = process_documents(uploaded_files)
                st.session_state["retriever"] = retriever
                st.session_state["vectors_processed"] = True
                st.success("Documents processed successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
                st.stop()
else:
    st.info("Please upload PDF documents to start chatting.")
    st.stop()

# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "Hi! I've read your documents. Ask me anything about them."})

# 4. Display Chat Messages
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. Handle User Input
user_input = st.chat_input("Ask a question...")

if user_input:
    # Display user message
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
