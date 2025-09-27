import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile

# ---------------- Sidebar ---------------- #
st.sidebar.title("📄 PDF Chat with Gemini")

api_key = st.sidebar.text_input("🔑 Enter your Gemini API Key", type="password")
uploaded_files = st.sidebar.file_uploader("📁 Upload PDF Files", type=["pdf"], accept_multiple_files=True)

top_k = st.sidebar.slider("🔍 Top N Relevant Chunks", min_value=1, max_value=10, value=4)

# ---------------- Gemini Setup ---------------- #
def test_api_key(key):
    try:
        genai.configure(api_key=key)
        _ = genai.embed_content(model="gemini-embedding-001", content="test")  # Try embedding
        return True
    except Exception as e:
        return False

if api_key:
    if test_api_key(api_key):
        st.sidebar.success("✅ Gemini API key is valid and connected.")

        class GeminiEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [genai.embed_content(model="models/embedding-001", content=t)["embedding"] for t in texts]

            def embed_query(self, text):
                return genai.embed_content(model="models/embedding-001", content=text)["embedding"]

        if uploaded_files:
            st.sidebar.success("📄 PDFs uploaded.")
            all_text = ""
            for file in uploaded_files:
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        all_text += text

            # Chunk the text
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(all_text)
            documents = [Document(page_content=chunk) for chunk in chunks]

            embedder = GeminiEmbeddings()
            with st.spinner("🔎 Creating embeddings..."):
                embeddings = embedder.embed_documents([doc.page_content for doc in documents])

            st.sidebar.success("✅ Embeddings created and stored.")

            # ---------------- Main QA Interface ---------------- #
            st.title("🤖 Ask Questions about your PDFs")

            user_query = st.text_input("Ask a question:")

            if user_query:
                query_embedding = embedder.embed_query(user_query)
                sim_scores = cosine_similarity([query_embedding], embeddings)[0]

                # Get top_k chunks based on slider
                top_indices = np.argsort(sim_scores)[::-1][:top_k]
                top_chunks = [documents[i].page_content for i in top_indices if sim_scores[i] > 0.1]

                if top_chunks:
                    context = "\n\n".join(top_chunks)
                    prompt = f"""You are a helpful assistant. Answer the following question using the context below.

Context:
{context}

Question: {user_query}

Answer:"""

                    model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
                    response = model.generate_content(prompt)

                    st.markdown("### 📚 Answer")
                    st.write(response.text)
                else:
                    st.warning("🤔 I am unable to find relevant data from the files.")
        else:
            st.sidebar.info("📥 Please upload PDFs.")
    else:
        st.sidebar.error("❌ Invalid Gemini API key. Please try again.")
else:
    st.sidebar.info("🔐 Please enter your Gemini API key to begin.")

