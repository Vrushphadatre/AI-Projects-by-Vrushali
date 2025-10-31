# --------------------------------------------------------------
# Chatbot with RAG (Retrieval-Augmented Generation)
# Built by: Vrushali Phadatre
# --------------------------------------------------------------

import os
import warnings
import logging
import streamlit as st

# LangChain & Groq Libraries
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# --------------------------------------------------------------
# Phase 1: Streamlit UI Setup
# --------------------------------------------------------------

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(page_title="Ask Chatbot!", page_icon="💬")
st.title("💬 Ask Chatbot!")

# Maintain session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# --------------------------------------------------------------
# Phase 3: PDF VectorStore (RAG)
# --------------------------------------------------------------

@st.cache_resource
def get_vectorstore():
    pdf_name = "ems_ai_survey.pdf"
    if not os.path.exists(pdf_name):
        st.error(f"Missing file: {pdf_name}")
        return None

    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)

    return index.vectorstore


# --------------------------------------------------------------
# Phase 2: Chat Model + Prompt Template
# --------------------------------------------------------------

def get_chat_response(user_prompt):
    """Handles Groq LLM + RAG Retrieval Pipeline"""

    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are a highly intelligent AI assistant.
        Always give the most accurate and concise answer possible.
        Question: {user_prompt}
        Start your answer directly, without small talk.
    """)

    # Load model from Groq
    model = "llama3-8b-8192"
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY not found. Please set it in your environment.")
        return "Missing Groq API key."

    groq_chat = ChatGroq(api_key=api_key, model_name=model)

    # Get vectorstore for retrieval
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "Could not load document or embeddings."

    chain = RetrievalQA.from_chain_type(
        llm=groq_chat,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    result = chain({"query": user_prompt})
    return result["result"]


# --------------------------------------------------------------
# Streamlit Chat Interface
# --------------------------------------------------------------

prompt = st.chat_input("Ask me anything...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = get_chat_response(prompt)
    except Exception as e:
        response = f"Error: {str(e)}"

    # Display assistant response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
