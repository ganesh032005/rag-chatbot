import streamlit as st
import tempfile, os
from rag_core import load_pdf, chunk_documents, store_embeddings, build_qa_chain, load_existing_db

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Ask Your PDF")
st.caption("Upload a PDF and ask questions about it")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.session_state.chain is None:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            pages  = load_pdf(tmp_path)
            chunks = chunk_documents(pages)
            db     = store_embeddings(chunks)
            st.session_state.chain = build_qa_chain(db)
            os.unlink(tmp_path)
        st.success("✓ PDF processed! Ask your questions.")

    if st.session_state.chain:
        st.info("PDF ready ✓")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chain = None
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})