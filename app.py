import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chatbot", page_icon="📄")
st.title("📄 Ask Your PDF")
st.caption("Upload a PDF and ask questions about it")

# ── Session state for chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False

# ── Sidebar: PDF upload ──
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF..."):
            response = requests.post(
                f"{API_URL}/upload",
                files={"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            )
        if response.status_code == 200:
            st.success("✓ PDF processed! Ask your questions.")
            st.session_state.pdf_uploaded = True
        else:
            st.error("Failed to process PDF")

    if st.session_state.pdf_uploaded:
        st.info("PDF ready ✓")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.pdf_uploaded = False
        st.rerun()

# ── Chat history display ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat input ──
if question := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question}
            )
        if response.status_code == 200:
            answer = response.json()["answer"]
        else:
            answer = "Error getting answer. Make sure API is running."
        st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})