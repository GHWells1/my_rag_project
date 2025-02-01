from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Load and split documents
loader = TextLoader("example.txt")  # Replace with your dataset
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
