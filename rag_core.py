import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    print(f"✓ Loaded {len(pages)} pages")
    return pages

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"✓ {len(chunks)} chunks created")
    return chunks

def store_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print("✓ Stored in ChromaDB")
    return db

def load_existing_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return db

def build_qa_chain(db):
    prompt = PromptTemplate.from_template("""
Use the following context to answer the question.
If you don't know, say "I don't know based on this document."

Context: {context}
Question: {question}
Answer:""")

    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✓ QA chain ready")
    return chain

def ask(chain, question):
    print(f"\nQ: {question}")
    answer = chain.invoke(question)
    print(f"A: {answer}")
    return answer

if __name__ == "__main__":
    PDF_PATH = "data/machine-learning-cheat-sheet.pdf"  # ← your PDF name

    pages  = load_pdf(PDF_PATH)
    chunks = chunk_documents(pages)
    db     = store_embeddings(chunks)
    chain  = build_qa_chain(db)

    ask(chain, "What is clustering?")
    ask(chain, "Explain sparse linear models")