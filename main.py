from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import time

max_retries = 5
retry_delay = 10  # seconds

for attempt in range(max_retries):
    try:
        embeddings = OpenAIEmbeddings(api_key="your-api-key-here")
        # Your code to create vector store and other logic here
        break
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)



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