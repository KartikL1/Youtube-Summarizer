import streamlit as st
import re
import uuid
import textwrap
from chromadb import PersistentClient

# Config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def get_youtube_transcript(video_url):
    """Working transcript fetcher"""
    try:
        from pytube import YouTube
        
        # Extract video ID
        video_id_match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11})", video_url)
        if not video_id_match:
            st.error("❌ Invalid YouTube URL")
            return None
            
        video_id = video_id_match.group(1)
        st.info("🔄 Fetching transcript...")

        # Create YouTube object
        yt = YouTube(video_url)
        
        # Get available caption tracks
        caption_tracks = yt.captions
        
        if not caption_tracks:
            st.error("❌ No captions available for this video")
            return None
        
        # Try English first, then any available language
        languages_to_try = ['en', 'a.en', 'en-US', 'en-GB']
        
        for lang_code in languages_to_try:
            if lang_code in caption_tracks:
                try:
                    caption = caption_tracks[lang_code]
                    transcript = caption.generate_srt_captions()
                    # Convert SRT to plain text
                    lines = transcript.split('\n')
                    text_lines = []
                    for line in lines:
                        # Skip timestamp lines and empty lines
                        if '-->' not in line and line.strip() and not line.strip().isdigit():
                            text_lines.append(line.strip())
                    text = ' '.join(text_lines)
                    if text.strip():
                        st.success(f"✅ Found {lang_code} captions ({len(text)} characters)")
                        return text.strip()
                except Exception as e:
                    continue
        
        # If no specific language found, try any available caption
        for lang_code, caption in caption_tracks.items():
            try:
                transcript = caption.generate_srt_captions()
                lines = transcript.split('\n')
                text_lines = []
                for line in lines:
                    if '-->' not in line and line.strip() and not line.strip().isdigit():
                        text_lines.append(line.strip())
                text = ' '.join(text_lines)
                if text.strip():
                    st.success(f"✅ Found {lang_code} captions ({len(text)} characters)")
                    return text.strip()
            except Exception as e:
                continue
        
        st.error("❌ Could not extract captions from available tracks")
        return None
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple text chunking"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if len(chunk) > 20:
            chunks.append(chunk)
    
    return chunks

def get_chroma_client_and_collection():
    client = PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="youtube_rag")
    return client, collection

def add_transcript_to_collection(youtube_url):
    text = get_youtube_transcript(youtube_url)
    if not text:
        return None

    chunks = chunk_text(text)
    if len(chunks) == 0:
        return None

    client, collection = get_chroma_client_and_collection()
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"youtube_url": youtube_url, "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(ids=ids, metadatas=metadatas, documents=chunks)
    return {"n_chunks": len(chunks)}

def retrieve_relevant_chunks(query, top_k=3):
    client, collection = get_chroma_client_and_collection()
    results = collection.query(query_texts=[query], n_results=top_k, include=["documents", "metadatas"])
    docs = results['documents'][0] if results['documents'] else []
    metas = results.get('metadatas', [[]])[0]
    return docs, metas

def answer_question(question, context_chunks):
    if not context_chunks:
        return "No relevant information found."
    
    # Simple answer generation
    combined_context = " ".join(context_chunks)
    return f"Based on the video: {combined_context[:300]}..."

# Streamlit UI
st.set_page_config(page_title="YouTube RAG", layout="wide")
st.title("🎥 YouTube RAG Q&A")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Add Video")
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Process Video"):
        if youtube_url:
            result = add_transcript_to_collection(youtube_url)
            if result:
                st.success(f"✅ Processed {result['n_chunks']} chunks!")
            else:
                st.error("❌ Failed to process video")

with col2:
    st.header("Ask Questions")
    question = st.text_input("Your question", placeholder="What is this video about?")
    
    if st.button("Get Answer"):
        if question:
            docs, metas = retrieve_relevant_chunks(question)
            if docs:
                answer = answer_question(question, docs)
                st.success(f"💬 {answer}")
                
                st.subheader("Retrieved Content:")
                for i, doc in enumerate(docs):
                    st.write(f"**Passage {i+1}:** {doc[:200]}...")
            else:
                st.info("💡 No videos processed yet. Add a video first!")

st.info("💡 Try videos with captions like TED Talks or educational content")
