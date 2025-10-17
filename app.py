import streamlit as st
import uuid
from chromadb import PersistentClient
from pytube import YouTube
from openai import OpenAI
from openai.embeddings_utils import get_embedding

# ---------------- Config ----------------
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

# Initialize ChromaDB
client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection("youtube_transcripts")

# Initialize OpenAI client
openai_client = OpenAI()

# ---------------- Utilities ----------------
def get_transcript(video_url):
    """Fetch transcript using pytube with safety checks."""
    try:
        YouTube._js_cache.clear()  # avoid old cache issues
        yt = YouTube(video_url)

        if yt.age_restricted:
            raise RuntimeError("Video is age-restricted.")
        if not yt.captions:
            raise RuntimeError("No captions available for this video.")

        # Prefer English or auto-generated English
        caption = None
        for code in ["en", "a.en"]:
            if code in yt.captions:
                caption = yt.captions[code]
                break
        if not caption:
            caption = list(yt.captions.values())[0]

        srt_text = caption.generate_srt_captions()
        # Remove numbers and timestamps from SRT
        lines = [line for line in srt_text.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line]
        text = " ".join(lines)
        if not text.strip():
            raise RuntimeError("Captions are empty after processing.")
        return text

    except Exception as e:
        raise RuntimeError(f"Unable to fetch transcript: {str(e)}")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks with overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def add_transcript(video_url):
    """Fetch transcript, create embeddings, and store in ChromaDB."""
    text = get_transcript(video_url)
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks created from transcript.")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"video_url": video_url, "chunk_index": i} for i in range(len(chunks))]
    embeddings = [get_embedding(chunk, model=EMBEDDING_MODEL) for chunk in chunks]

    collection.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
    return len(chunks)

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve top-k relevant chunks for a query."""
    query_embedding = get_embedding(query, model=EMBEDDING_MODEL)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return docs, metas

def answer_question(query):
    """Generate a simple answer from retrieved chunks."""
    docs, metas = retrieve_relevant_chunks(query)
    if not docs:
        return "No relevant information found. Add a video first!"
    combined_context = " ".join(docs)
    return f"Based on the videos: {combined_context[:500]}..."

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="YouTube RAG Q&A", layout="wide")
st.title("🎥 YouTube RAG Q&A App")

col1, col2 = st.columns([1, 2])

# ----- Add Video -----
with col1:
    st.header("Add Video")
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Process Video"):
        if video_url.strip():
            try:
                n_chunks = add_transcript(video_url)
                st.success(f"✅ Processed {n_chunks} chunks and stored in ChromaDB!")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")

# ----- Ask Questions -----
with col2:
    st.header("Ask Questions")
    question = st.text_input("Your question", placeholder="What is this video about?")
    if st.button("Get Answer"):
        if question.strip():
            answer = answer_question(question)
            st.success(f"💬 {answer}")
