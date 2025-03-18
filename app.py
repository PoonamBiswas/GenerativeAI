import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# Check if FAISS index exists
if not os.path.exists("faiss_index/index.faiss"):
    st.error("⚠️ FAISS index not found! Please run `create_index.py` first.")
    st.stop()

# Load FAISS index safely
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Create RAG chain
llm = Ollama(model="llama2")
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.title("🌾 Agriculture RAG Chatbot")
st.write("Ask any agriculture-related question!")

# User Input
query = st.text_input("Type your question:")

if query:
    response = rag_chain.invoke({"query": query})
    st.write("🤖 Chatbot Response:")
    st.write(response["result"])  # Extract only the answer
