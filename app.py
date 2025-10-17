import streamlit as st
import re
import uuid
from chromadb import PersistentClient

# Import transcript API globally
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# ---------------- CONFIG ----------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---------------- UTILITIES ----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_youtube_transcript(video_url):
    """Fetch transcript robustly for different library versions."""
    try:
        # Extract video ID from multiple formats
        video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
        if not video_id_match:
            st.error("❌ Invalid YouTube URL.")
            return None

        video_id = video_id_match.group(1)
        st.info("🔄 Fetching transcript...")

        # Handle API version differences safely
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            # ✅ Modern versions use list_transcripts()
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # Prefer English or first available
            for lang_code in ["en", "en-US", "en-GB", "hi"]:
                try:
                    transcript = transcript_list.find_transcript([lang_code])
                    text = " ".join([entry['text'] for entry in transcript.fetch()])
                    st.success(f"✅ Transcript found in '{lang_code}' ({len(text)} chars)")
                    return text.strip()
                except NoTranscriptFound:
                    continue
            # fallback to first available transcript
            transcript = transcript_list.find_manually_created_transcript(transcript_list._manually_created_transcripts.keys()) \
                if transcript_list._manually_created_transcripts else transcript_list._generated_transcripts.popitem()[1]
            text = " ".join([entry['text'] for entry in transcript.fetch()])
            st.success("✅ Fallback transcript fetched successfully.")
            return text.strip()

        elif hasattr(YouTubeTranscriptApi, "get_transcript"):
            # ✅ Older versions use get_transcript()
            for lang in ['en', 'en-US', 'en-GB', 'hi', 'auto']:
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    text = " ".join([entry['text'] for entry in transcript])
                    st.success(f"✅ Transcript found ({len(text)} chars) in '{lang}'")
                    return text.strip()
                except (NoTranscriptFound, TranscriptsDisabled):
                    continue
            st.error("❌ No subtitles found for this video.")
            return None
        else:
            st.error("❌ Unsupported youtube-transcript-api version.")
            return None

    except VideoUnavailable:
        st.error("❌ The video is unavailable or restricted.")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None


def get_chroma_client_and_collection():
    """Create or get a persistent ChromaDB collection."""
    client = PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="youtube_rag")
    return client, collection


def add_transcript_to_collection(youtube_url):
    """Fetch transcript, chunk it, and store in ChromaDB."""
    text = get_youtube_transcript(youtube_url)
    if not text:
        return None

    chunks = chunk_text(text)
    if not chunks:
        st.warning("⚠️ Transcript too short to process.")
        return None

    client, collection = get_chroma_client_and_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"youtube_url": youtube_url, "chunk_index": i} for i in range(len(chunks))]
    collection.add(ids=ids, metadatas=metadatas, documents=chunks)
    return {"n_chunks": len(chunks)}


def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve relevant transcript chunks for a query."""
    client, collection = get_chroma_client_and_collection()
    results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas"])
    docs = results['documents'][0] if results['documents'] else []
    metas = results.get('metadatas', [[]])[0]
    return docs, metas


def answer_question(question, context_chunks):
    """Simple context-based response."""
    if not context_chunks:
        return "No relevant information found in stored transcripts."
    combined_context = " ".join(context_chunks)
    return f"Based on the video: {combined_context[:400]}..."


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="🎥 YouTube RAG Q&A", layout="wide")
st.title("🎬 YouTube RAG Q&A")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📥 Add a YouTube Video")
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("Process Video"):
        if youtube_url:
            with st.spinner("⏳ Processing video..."):
                result = add_transcript_to_collection(youtube_url)
            if result:
                st.success(f"✅ Processed {result['n_chunks']} transcript chunks!")
            else:
                st.error("❌ Could not process this video (no subtitles or error).")
        else:
            st.warning("⚠️ Please enter a valid YouTube link first.")

with col2:
    st.header("💬 Ask a Question")
    question = st.text_input("Your question", placeholder="E.g. What is the main topic of this video?")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("🔍 Searching..."):
                docs, metas = retrieve_relevant_chunks(question)
            if docs:
                answer = answer_question(question, docs)
                st.success(f"💡 {answer}")

                st.subheader("🔎 Retrieved Transcript Segments")
                for i, doc in enumerate(docs):
                    st.markdown(f"**Segment {i+1}:** {doc[:250]}...")
            else:
                st.info("ℹ️ No processed videos yet. Add one first.")
        else:
            st.warning("⚠️ Please enter a question first.")

st.divider()
st.info("💡 Tip: Use captioned videos like TED Talks or tutorials for best results.")
