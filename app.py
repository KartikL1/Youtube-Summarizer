import streamlit as st
import uuid
from chromadb import PersistentClient
from pytube import YouTube
from openai import OpenAI

# ---------------- Config ----------------
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
GPT_MODEL = "gpt-3.5-turbo"

# ---------------- OpenAI Client ----------------
# Replace with your key or use environment variable
openai_client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")

# ---------------- ChromaDB ----------------
client = PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection("youtube_transcripts")

# ---------------- Utilities ----------------
def get_embedding(text, model=EMBEDDING_MODEL):
    response = openai_client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def get_transcript(video_url):
    try:
        YouTube._js_cache.clear()
        yt = YouTube(video_url)
        if yt.age_restricted:
            raise RuntimeError("Video is age-restricted.")
        if not yt.captions:
            raise RuntimeError("No captions available.")

        caption = None
        for code in ["en", "a.en"]:
            if code in yt.captions:
                caption = yt.captions[code]
                break
        if not caption:
            caption = list(yt.captions.values())[0]

        srt_text = caption.generate_srt_captions()
        lines = [
            line for line in srt_text.splitlines()
            if line.strip() and not line.strip().isdigit() and "-->" not in line
        ]
        text = " ".join(lines)
        if not text.strip():
            raise RuntimeError("Captions empty after processing.")
        return text

    except Exception as e:
        raise RuntimeError(f"Unable to fetch transcript: {str(e)}")

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_size]))
        i += chunk_size - overlap
    return chunks

def add_transcript(video_url):
    text = get_transcript(video_url)
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("No chunks created.")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"video_url": video_url, "chunk_index": i} for i in range(len(chunks))]
    embeddings = [get_embedding(chunk) for chunk in chunks]

    collection.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
    return len(chunks)

def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    docs = results.get("documents", [[]])[0]
    return docs

def generate_gpt_answer(query, context_chunks):
    if not context_chunks:
        return "No relevant information found."
    context_text = "\n\n".join(context_chunks)
    prompt = f"Answer the question based on the following transcript excerpts:\n{context_text}\n\nQuestion: {query}"
    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

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
                st.success(f"✅ Processed {n_chunks} chunks!")
            except Exception as e:
                st.error(f"❌ Failed: {str(e)}")

# ----- Ask Questions -----
with col2:
    st.header("Ask Questions")
    question = st.text_input("Your question", placeholder="What is this video about?")
    if st.button("Get Answer"):
        if question.strip():
            chunks = retrieve_relevant_chunks(question)
            answer = generate_gpt_answer(question, chunks)
            st.success(f"💬 {answer}")
            st.subheader("Retrieved Excerpts:")
            for i, c in enumerate(chunks):
                st.write(f"**Chunk {i+1}:** {c[:200]}...")
