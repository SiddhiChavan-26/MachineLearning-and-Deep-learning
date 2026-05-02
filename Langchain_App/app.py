import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="RAG App", page_icon="📄", layout="centered")
st.title("📄 RAG Document Q&A")
st.caption("Upload a document and ask questions about it using LLaMA 3.3 via Groq.")

# ── API Key ───────────────────────────────────────────────────
# Reads from Streamlit Cloud Secrets (set GROQ_API_KEY there)
api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

client = Groq(api_key=api_key)

# ── Document Loader ───────────────────────────────────────────
def load_document(uploaded_file):
    name = uploaded_file.name

    if name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text

    elif name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join(para.text for para in doc.paragraphs)

    else:
        return None

# ── File Upload ───────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "pdf", "docx"],
    help="Supported: .txt, .pdf, .docx"
)

if uploaded_file:
    with st.spinner("Reading document..."):
        document = load_document(uploaded_file)

    if not document or not document.strip():
        st.error("❌ Could not extract text from this file.")
        st.stop()

    st.success(f"✅ Loaded **{uploaded_file.name}** — {len(document):,} characters")

    # Show a preview
    with st.expander("📖 Document Preview (first 1000 chars)"):
        st.text(document[:1000])

    # ── Chat History ──────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── User Input ────────────────────────────────────────────
    if prompt := st.chat_input("Ask something about the document..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build Groq messages
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer ONLY using the document provided. "
                "If the answer is not in the document, say so clearly."
            )
        }
        doc_context = {
            "role": "user",
            "content": f"Here is the document to reference:\n\n{document}"
        }
        doc_ack = {
            "role": "assistant",
            "content": "Understood. I will answer only based on this document."
        }

        # Include chat history for multi-turn
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[system_msg, doc_context, doc_ack] + history,
                        max_tokens=1024,
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"❌ Error from Groq API: {e}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👆 Upload a .txt, .pdf, or .docx file to get started.")