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
st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📚")

# --- UI Setup ---
st.title("Multi-PDF RAG Chatbot 📚")
st.markdown("Upload **one or more** PDF documents and ask questions based on their content.")

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("Configuration")
    user_api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    st.divider()
    
    # MODIFIED: Added accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# --- Helper Functions ---

def process_documents(uploaded_files):
    """
    Handles saving, loading, splitting, and vector storage for MULTIPLE files.
    """
    all_docs = []
    
    # 1. Loop through each uploaded file
    for uploaded_file in uploaded_files:
        # Create a temp file for each upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        try:
            # Load the document
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs) # Add pages to the master list
        finally:
            # Clean up the temp file
            os.remove(temp_file_path)

    # 2. Split all collected documents
    # (Chunk size can be adjusted based on total volume)
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    doc_chunks = splitter.split_documents(all_docs)

    # 3. Create Embeddings & Vector Store
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

# --- Main Application Logic ---

if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key
elif "OPENAI_API_KEY" not in os.environ:
    st.info("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

# MODIFIED: Check if the list `uploaded_files` is not empty
if uploaded_files:
    if "vectors_processed" not in st.session_state:
        with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):
            # Pass the list of files to the processing function
            retriever = process_documents(uploaded_files)
            st.session_state["retriever"] = retriever
            st.session_state["vectors_processed"] = True
            st.success("Documents processed!")
else:
    st.info("Please upload PDF documents to start chatting.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "Hi! I've read your documents. Ask me anything about them."})

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask a question about your PDFs...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

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
