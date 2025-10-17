import streamlit as st
import re
import uuid
from chromadb import PersistentClient
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# -----------------------------------
# 🎯 Initialize ChromaDB client
# -----------------------------------
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("youtube_transcripts")


# -----------------------------------
# 🔍 Extract YouTube video ID
# -----------------------------------
def extract_video_id(url: str):
    """
    Extracts video ID from a YouTube URL.
    """
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# -----------------------------------
# 🧾 Fetch Transcript Function
# -----------------------------------
def get_youtube_transcript(url: str):
    """
    Fetches transcript for a given YouTube video URL using youtube-transcript-api.
    Handles unavailable videos, disabled captions, and fallback languages.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL format.")

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching English transcript first
        try:
            transcript = transcript_list.find_transcript(['en'])
        except:
            # If English not found, pick first available
            available_langs = [t.language_code for t in transcript_list]
            if available_langs:
                transcript = transcript_list.find_transcript([available_langs[0]])
            else:
                raise NoTranscriptFound("No transcript available in any language.")

        fetched = transcript.fetch()
        text = " ".join([x['text'] for x in fetched if x['text'].strip()])
        return text

    except VideoUnavailable:
        raise RuntimeError("This YouTube video is unavailable.")
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error fetching transcript: {str(e)}")


# -----------------------------------
# 💾 Store Transcript in ChromaDB
# -----------------------------------
def add_transcript_to_collection(url: str):
    """
    Fetches a transcript and stores it in the Chroma collection.
    """
    st.info("🔄 Fetching transcript...")
    try:
        text = get_youtube_transcript(url)
        st.success(f"✅ Transcript fetched successfully ({len(text)} characters)")

        doc_id = str(uuid.uuid4())
        collection.add(
            documents=[text],
            ids=[doc_id],
            metadatas=[{"url": url}],
        )

        return f"✅ Transcript added to collection (ID: {doc_id[:8]})"

    except Exception as e:
        st.error(f"❌ {str(e)}")
        return None


# -----------------------------------
# 🚀 Streamlit App Interface
# -----------------------------------
st.set_page_config(page_title="YouTube Transcript Fetcher", page_icon="🎥", layout="centered")

st.title("🎥 YouTube Transcript Fetcher + ChromaDB Store")
st.caption("Extract video transcripts and save them locally for retrieval or RAG projects.")

youtube_url = st.text_input("🔗 Enter YouTube video URL:")

if st.button("Fetch & Store Transcript"):
    if youtube_url.strip():
        with st.spinner("Processing..."):
            result = add_transcript_to_collection(youtube_url)
            if result:
                st.success(result)
    else:
        st.warning("⚠️ Please enter a valid YouTube video URL.")

st.divider()
st.markdown("✅ **Tip:** Try a video with captions, e.g. [How Great Leaders Inspire Action (TED)](https://www.youtube.com/watch?v=H14bBuluwB8)")
