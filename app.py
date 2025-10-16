def get_youtube_transcript(video_url):
    """Working transcript fetcher using pytube captions"""
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
