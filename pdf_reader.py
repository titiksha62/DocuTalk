import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from google import genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- Sidebar ---------------- #

st.sidebar.title("📄 PDF Chat with Gemini")

api_key = st.sidebar.text_input(
    "🔑 Enter your Gemini API Key",
    type="password"
)

uploaded_files = st.sidebar.file_uploader(
    "📁 Upload PDF Files",
    type=["pdf"],
    accept_multiple_files=True
)

top_k = st.sidebar.slider(
    "🔍 Top N Relevant Chunks",
    min_value=1,
    max_value=10,
    value=4
)


# ---------------- Gemini Setup ---------------- #

def test_api_key(key):
    """
    Test whether the Gemini API key is valid
    by making a small embedding request.
    """
    try:
        client = genai.Client(api_key=key)

        client.models.embed_content(
            model="gemini-embedding-001",
            contents="test"
        )

        return True

    except Exception:
        return False


def embed_documents(client, texts):
    """
    Generate embeddings for multiple document chunks.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts
    )

    return [
        embedding.values
        for embedding in result.embeddings
    ]


def embed_query(client, text):
    """
    Generate an embedding for the user's query.
    """
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values


# ---------------- Main Application ---------------- #

if api_key:

    if test_api_key(api_key):

        st.sidebar.success(
            "✅ Gemini API key is valid and connected."
        )

        # Create Gemini client
        client = genai.Client(api_key=api_key)


        # ---------------- PDF Upload ---------------- #

        if uploaded_files:

            st.sidebar.success("📄 PDFs uploaded.")

            all_text = ""

            for file in uploaded_files:

                reader = PdfReader(file)

                for page in reader.pages:

                    text = page.extract_text()

                    if text:
                        all_text += text + "\n"


            # ---------------- Chunk the Text ---------------- #

            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            documents = text_splitter.split_text(all_text)


            # Check whether PDF contained readable text

            if not documents:

                st.warning(
                    "⚠️ No readable text was found in the uploaded PDFs."
                )

            else:

                # ---------------- Create Embeddings ---------------- #

                with st.spinner("🔎 Creating embeddings..."):

                    embeddings = embed_documents(
                        client,
                        documents
                    )

                st.sidebar.success(
                    "✅ Embeddings created and stored."
                )


                # ---------------- Main QA Interface ---------------- #

                st.title("🤖 Ask Questions about your PDFs")

                user_query = st.text_input(
                    "Ask a question:"
                )


                if user_query:

                    with st.spinner("🤔 Finding the answer..."):

                        # Generate embedding for user's question

                        query_embedding = embed_query(
                            client,
                            user_query
                        )


                        # Calculate cosine similarity

                        sim_scores = cosine_similarity(
                            [query_embedding],
                            embeddings
                        )[0]


                        # Get top-k most relevant chunks

                        top_indices = np.argsort(
                            sim_scores
                        )[::-1][:top_k]


                        top_chunks = [
                            documents[i]
                            for i in top_indices
                            if sim_scores[i] > 0.1
                        ]


                        # ---------------- Generate Answer ---------------- #

                        if top_chunks:

                            context = "\n\n".join(
                                top_chunks
                            )


                            prompt = f"""
You are a helpful assistant that answers questions
about uploaded PDF documents.

Use ONLY the information provided in the context below.

If the answer cannot be found in the context,
say:

"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{user_query}

Answer:
"""


                            # Generate response using Gemini

                            response = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=prompt
                            )


                            st.markdown("### 📚 Answer")

                            st.write(response.text)


                        else:

                            st.warning(
                                "🤔 I am unable to find relevant data from the files."
                            )


        else:

            st.sidebar.info(
                "📥 Please upload PDFs."
            )


    else:

        st.sidebar.error(
            "❌ Invalid Gemini API key. Please try again."
        )


else:

    st.sidebar.info(
        "🔐 Please enter your Gemini API key to begin."
    )
