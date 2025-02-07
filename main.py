from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

# Load and split documents
loader = TextLoader("example.txt")  # Ensure this file exists
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(docs, embeddings)

# Create retrieval-based Q&A
qa = RetrievalQA.from_chain_type(llm=OpenAI(), retriever=vector_store.as_retriever())

# Test RAG pipeline
query = "What is this document about?"
print(qa.run(query))


#test

import os

file_path = "example.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File '{file_path}' not found. Please provide the correct path.")

loader = TextLoader(file_path)
documents = loader.load()