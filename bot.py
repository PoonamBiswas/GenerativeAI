from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.document_loaders import PyMuPDFLoader

# Specify the PDF file path
pdf_path = "/Users/poonam/dev/llamaenv/agenticRag/agribot/data/farmerbook.pdf"

# Load PDF using PyMuPDFLoader
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

# Print basic details
print(f"✅ Loaded {len(documents)} pages from {pdf_path}")

# Preview first page content
print("📄 First Page Content:")
print(documents[0].page_content[:500])  # Show first 500 characters

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Split PDF content into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Print the number of chunks
print(f"🔹 Total Chunks Created: {len(docs)}")
print(f"📝 First Chunk Preview: {docs[0].page_content}")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert text into vector embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save FAISS index
vectorstore.save_local("faiss_index")

from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

from langchain.chains import RetrievalQA

# Load FAISS index safely
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True  # Explicitly allow deserialization
)

retriever = vectorstore.as_retriever()

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
