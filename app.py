import streamlit as st
import re
import uuid
import xml.etree.ElementTree as ET

from chromadb import PersistentClient

# Try to import youtube_transcript_api gracefully
try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
    )
    YT_TRANSCRIPT_API_AVAILABLE = True
except Exception:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = NoTranscriptFound = VideoUnavailable = Exception
    YT_TRANSCRIPT_API_AVAILABLE = False

# We'll use pytube as a robust fallback
try:
    from pytube import YouTube as PytubeYouTube
    PYTUBE_AVAILABLE = True
except Exception:
    PytubeYouTube = None
    PYTUBE_AVAILABLE = False

# requests for direct timedtext fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False

# Initialize ChromaDB
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("youtube_transcripts")

# ---------------- utils ----------------
def extract_video_id(url: str):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def text_from_pytube_captions(video_url, preferred_langs=("en", "en-US", "en-GB")):
    """Try to read captions using pytube (generate_srt_captions)."""
    if not PYTUBE_AVAILABLE:
        raise RuntimeError("pytube not installed")
    yt = PytubeYouTube(video_url)
    captions = yt.captions
    if not captions:
        raise RuntimeError("No captions available via pytube")
    # Try preferred languages first
    for code in preferred_langs:
        tag = None
        for c in captions:
            # pytube caption objects have .code (like 'en' or 'a.en' for auto)
            if getattr(c, "code", "").startswith(code) or getattr(c, "code", "") == code:
                tag = c
                break
        if tag:
            srt = tag.generate_srt_captions()
            # remove any SRT numbering/timestamps -> keep only text
            return "\n".join([line for line in srt.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line])
    # fallback: pick first caption
    first = next(iter(captions), None)
    if first:
        srt = first.generate_srt_captions()
        return "\n".join([line for line in srt.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line])
    raise RuntimeError("No usable captions found via pytube")

def text_from_timedtext_endpoint(video_id, lang="en"):
    """Call YouTube timedtext endpoint and parse XML captions (manual captions often here)."""
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("requests not installed")
    # Common timedtext endpoints
    urls = [
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}",
        f"https://video.google.com/timedtext?lang={lang}&v={video_id}",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=srv3",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=10)
            if r.status_code != 200:
                continue
            if not r.text.strip():
                continue
            # parse XML
            try:
                root = ET.fromstring(r.text)
                texts = []
                for child in root.findall(".//text"):
                    if child.text:
                        texts.append(child.text.replace("\n", " ").strip())
                if texts:
                    return " ".join(texts)
            except ET.ParseError:
                # sometimes service returns strange payloads; try to return raw text
                if r.text.strip():
                    return r.text.strip()
        except Exception:
            continue
    raise RuntimeError("Timedtext endpoint returned no captions")

def get_youtube_transcript_any(url: str):
    """
    Universal transcript getter:
      1) tries youtube_transcript_api (both old and new shapes)
      2) falls back to pytube captions
      3) falls back to direct timedtext HTTP endpoint
    Raises RuntimeError with clear message on failure.
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise RuntimeError("Invalid YouTube URL format.")

    # 1) youtube_transcript_api (many shapes)
    if YT_TRANSCRIPT_API_AVAILABLE:
        try:
            # Try old-style function if available
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                # try common languages
                for lang in ["en", "en-US", "en-GB", "hi", "auto"]:
                    try:
                        out = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                        if out:
                            return " ".join([entry.get("text", "") for entry in out if entry.get("text", "").strip()])
                    except (NoTranscriptFound, TranscriptsDisabled):
                        continue
                    except Exception:
                        # continue trying other means
                        continue
            # Try new-style list_transcripts if present
            if hasattr(YouTubeTranscriptApi, "list_transcripts"):
                try:
                    tl = YouTubeTranscriptApi.list_transcripts(video_id)
                    # prefer english
                    for code in ["en", "en-US", "en-GB", "hi"]:
                        try:
                            tr = tl.find_transcript([code])
                            fetched = tr.fetch()
                            return " ".join([entry.get("text", "") for entry in fetched if entry.get("text", "").strip()])
                        except Exception:
                            continue
                    # fallback: pick first transcript available
                    try:
                        # transcript list is iterable; pick first and fetch
                        first_tr = next(iter(tl))
                        fetched = first_tr.fetch()
                        return " ".join([entry.get("text", "") for entry in fetched if entry.get("text", "").strip()])
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            # fallthrough to other methods
            pass

    # 2) pytube fallback
    if PYTUBE_AVAILABLE:
        try:
            return text_from_pytube_captions(url)
        except Exception:
            pass

    # 3) direct timedtext HTTP fallback
    if REQUESTS_AVAILABLE:
        for lang in ("en", "en-GB", "en-US", "hi"):
            try:
                txt = text_from_timedtext_endpoint(video_id, lang=lang)
                if txt and txt.strip():
                    return txt
            except Exception:
                continue

    # If nothing worked, raise with helpful diagnostics
    diagnostics = []
    diagnostics.append(f"youtube_transcript_api_installed={YT_TRANSCRIPT_API_AVAILABLE}")
    diagnostics.append(f"pytube_installed={PYTUBE_AVAILABLE}")
    diagnostics.append(f"requests_installed={REQUESTS_AVAILABLE}")
    raise RuntimeError("Unable to fetch transcript via any method. Diagnostics: " + "; ".join(diagnostics))

# ---------------- store in chroma ----------------
def add_transcript_to_collection(url: str):
    st.info("🔄 Fetching transcript...")
    try:
        text = get_youtube_transcript_any(url)
        if not text or not text.strip():
            st.error("❌ Transcript fetched but empty.")
            return None
        st.success(f"✅ Transcript fetched ({len(text)} chars)")
        doc_id = str(uuid.uuid4())
        collection.add(documents=[text], ids=[doc_id], metadatas=[{"url": url}])
        return f"✅ Stored transcript (id={doc_id[:8]})"
    except Exception as e:
        st.error(f"❌ {str(e)}")
        return None

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="YouTube Transcript Fetcher (robust)", layout="centered")
st.title("YouTube Transcript Fetcher — robust (multi-fallback)")
st.write("This version tries multiple methods to fetch captions and gives diagnostic info on failure.")

youtube_url = st.text_input("YouTube URL:")
if st.button("Fetch & Store"):
    if not youtube_url.strip():
        st.warning("Enter a YouTube URL")
    else:
        with st.spinner("Processing..."):
            res = add_transcript_to_collection(youtube_url)
            if res:
                st.success(res)
            else:
                st.info("Check the red messages above for details.")
st.markdown("---")
st.markdown("**Note:** If this still fails on Streamlit Cloud, please ensure your `requirements.txt` includes `pytube` and `requests`, and then use the Streamlit app 'Manage app' → 'Clear cache and redeploy'.")
