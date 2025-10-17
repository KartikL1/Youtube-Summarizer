import streamlit as st
import uuid
from chromadb import PersistentClient
from pytube import YouTube

# Initialize ChromaDB
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("youtube_transcripts")

# ---------------- Extract transcript ----------------
def get_transcript_pytube(video_url):
    try:
        yt = YouTube(video_url)
        captions = yt.captions

        if not captions:
            raise ValueError("No captions available for this video.")

        # Prefer English captions if available
        caption = None
        for code in ["en", "a.en"]:
            if code in captions:
                caption = captions[code]
                break

        if not caption:
            # fallback: pick first available caption
            caption = list(captions.values())[0]

        srt_text = caption.generate_srt_captions()
        # Remove numbers and timestamps from SRT
        lines = [line for line in srt_text.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line]
        text = " ".join(lines)
        if not text.strip():
            raise ValueError("Captions are empty after processing.")
        return text
    except Exception as e:
        raise RuntimeError(f"Unable to fetch transcript: {str(e)}")

# ---------------- Store in Chroma ----------------
def add_transcript(video_url):
    st.info("🔄 Fetching transcript...")
    try:
        text = get_transcript_pytube(video_url)
        st.success(f"✅ Transcript fetched ({len(text)} characters)")

        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id], metadatas=[{"url": video_url}])
        return f"✅ Stored transcript (ID: {doc_id[:8]})"
    except Exception as e:
        st.error(f"❌ {str(e)}")
        return None

# ---------------- Streamlit UI ----------------
st.title("🎥 YouTube Transcript Fetcher (Pytube Only)")
video_url = st.text_input("Enter YouTube URL:")

if st.button("Fetch & Store"):
    if not video_url.strip():
        st.warning("Please enter a YouTube video URL.")
    else:
        with st.spinner("Processing..."):
            result = add_transcript(video_url)
            if result:
                st.success(result)

st.markdown("---")
st.markdown("💡 Tip: Test with captioned videos like [TED Talks](https://www.youtube.com/watch?v=H14bBuluwB8)")
