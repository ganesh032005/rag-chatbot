from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil, os
from rag_core import load_pdf, chunk_documents, store_embeddings, build_qa_chain, load_existing_db

app = FastAPI()


chain = None


def load_existing_db():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return db

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global chain
    save_path = f"data/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages  = load_pdf(save_path)
    chunks = chunk_documents(pages)
    db     = store_embeddings(chunks)
    chain  = build_qa_chain(db)

    return {"status": "PDF processed", "filename": file.filename}

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(q: Question):
    global chain
    if chain is None:
        db    = load_existing_db()
        chain = build_qa_chain(db)
    answer = chain.invoke(q.question)
    return {"question": q.question, "answer": answer}

@app.get("/")
def root():
    return {"status": "RAG API is running"}