import streamlit as st
import tempfile
import os
import torch
import numpy as np
import time
import warnings
import librosa
import soundfile as sf
import io
import zipfile
from datetime import datetime
from streamlit_mic_recorder import speech_to_text, mic_recorder
from pydub import AudioSegment
import magic
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import json

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Page configuration
st.set_page_config(
    page_title="Dockerized Speaker Diarization + ASR",
    page_icon="🐳",
    layout="wide"
)

# Progress container in session state
if 'progress_container' not in st.session_state:
    st.session_state.progress_container = None

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Header
st.title("🐳 Dockerized Speaker Diarization + ASR")
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .speaker-box {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f8f9fa;
    }
    .play-format {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Function to update progress display
def update_progress(container, current_segment, total_segments, speaker, segment_idx, segments_count):
    """Update the progress display with a single set of progress elements"""
    if container is not None:
        with container:
            # Clear previous content
            container.empty()
            
            # Display current progress
            st.markdown(f"**Transcribing {speaker}:** segment {segment_idx+1}/{segments_count}")
            st.progress(current_segment / total_segments)
            st.caption(f"Total: {current_segment}/{total_segments} segments")

# Force state update function
def force_state_update():
    """Forces Streamlit to update the state"""
    if 'force_update' not in st.session_state:
        st.session_state.force_update = 0
    st.session_state.force_update += 1

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    
    # Model settings
    st.subheader("Model Parameters")
    
    # Whisper settings
    with st.expander("Whisper Settings"):
        beam_size = st.slider("Beam Size", min_value=1, max_value=10, value=5, 
                             help="Higher values increase accuracy but slow down processing")
        best_of = st.slider("Best of", min_value=1, max_value=10, value=5,
                           help="Number of candidates when sampling with non-zero temperature")
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1,
                               help="Temperature for sampling, 0.0 is greedy decoding")
        language = st.selectbox("Language", ["ru", "en", "de", "es", "fr", "it", "auto"], 
                               index=0, help="Language for transcription, 'auto' for automatic detection")
        vad_filter = st.checkbox("VAD Filter", value=True, 
                               help="Enable voice activity detection to filter out non-speech parts")
    
    # Diarization settings
    with st.expander("Diarization Settings"):
        num_speakers = st.number_input("Number of Speakers (0 for auto)", min_value=0, max_value=10, value=0,
                                      help="0 for automatic detection")
        min_speakers = st.number_input("Min Speakers", min_value=1, max_value=10, value=1)
        max_speakers = st.number_input("Max Speakers", min_value=1, max_value=20, value=10)
    
    # Processing settings
    with st.expander("Processing Settings"):
        chunk_duration = st.slider("Chunk Duration (seconds)", min_value=10, max_value=60, value=30,
                                  help="Duration of audio chunks for processing")
        sample_rate = st.selectbox("Sample Rate", [8000, 16000, 22050, 44100, 48000], index=1,
                                  help="Sample rate for audio processing")
    
    # Debug mode
    debug_mode = st.checkbox("🐞 Debug Mode", value=False)
    
    # Export format settings
    st.subheader("Export Options")
    include_timestamps = st.checkbox("Include Timestamps in Play Format", value=True)
    format_time_as_hhmmss = st.checkbox("Format Time as HH:MM:SS", value=True)
    include_speaker_statistics = st.checkbox("Include Speaker Statistics", value=True)

# Get HF token from environment variables
hf_token = os.environ.get("HF_TOKEN", "")
whisper_model_size = os.environ.get("WHISPER_MODEL_SIZE", "medium")
device = os.environ.get("DEVICE", "cpu")
default_language = os.environ.get("DEFAULT_LANGUAGE", "ru")
debug_mode_env = os.environ.get("DEBUG_MODE", "false").lower() == "true"

if debug_mode_env:
    debug_mode = True

# Initialize session state
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'processing_result' not in st.session_state:
    st.session_state.processing_result = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'diarization_pipeline' not in st.session_state:
    st.session_state.diarization_pipeline = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'recording_key' not in st.session_state:
    st.session_state.recording_key = str(time.time())
if 'audio_info' not in st.session_state:
    st.session_state.audio_info = None
if 'language' not in st.session_state:
    st.session_state.language = language if language != "auto" else None

# Load models function
def load_models():
    """Load both diarization and ASR models"""
    with st.spinner("🔄 Loading models... This may take a moment on first run"):
        try:
            # Load diarization pipeline
            if st.session_state.diarization_pipeline is None:
                st.session_state.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                
                # Move to GPU if available and requested
                if device == "cuda" and torch.cuda.is_available():
                    st.session_state.diarization_pipeline.to(torch.device("cuda"))
                    st.sidebar.success("✅ Diarization model loaded on GPU")
                else:
                    st.sidebar.info("ℹ️ Diarization model loaded on CPU")
            
            # Load Whisper model
            if st.session_state.whisper_model is None:
                compute_type = "float16" if device == "cuda" and torch.cuda.is_available() else "float32"
                device_used = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                
                st.session_state.whisper_model = WhisperModel(
                    whisper_model_size,
                    device=device_used,
                    compute_type=compute_type
                )
                st.sidebar.success(f"✅ Whisper {whisper_model_size} model loaded on {device_used.upper()}")
            
            st.session_state.models_loaded = True
            force_state_update()  # Force state update
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.error("Make sure you have accepted the terms for both models on Hugging Face:")
            st.error("- pyannote/speaker-diarization-3.1")
            st.error("- pyannote/segmentation-3.0")
            return False

# Audio processing function
def process_audio(audio_bytes, target_sample_rate=16000):
    """Process and convert audio to the required format"""
    try:
        # Convert audio to WAV format using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        
        # Convert to mono and target sample rate
        audio_segment = audio_segment.set_channels(1).set_frame_rate(target_sample_rate)
        
        # Export to WAV in memory
        wav_buffer = io.BytesIO()
        audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Read with soundfile
        data, sr = sf.read(wav_buffer)
        
        # Convert to torch tensor
        waveform = torch.from_numpy(data).float().unsqueeze(0)
        
        # Save processed audio to temporary file for diarization
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, data, sr)
            audio_path = tmp_file.name
        
        st.session_state.audio_info = {
            'duration': len(data) / sr,
            'sample_rate': sr,
            'path': audio_path,
            'shape': waveform.shape
        }
        
        return waveform, sr, audio_path
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        return None, None, None

# Process function that combines diarization and ASR
def process_full_pipeline(audio_path, waveform, sample_rate):
    """Run the full pipeline: diarization + ASR"""
    try:
        # Create a container for progress updates
        progress_container = st.container()
        st.session_state.progress_container = progress_container
        
        # Step 1: Speaker diarization
        with progress_container:
            with st.spinner("🎤 Performing speaker diarization..."):
                diarization_kwargs = {}
                if num_speakers > 0:
                    diarization_kwargs["num_speakers"] = num_speakers
                else:
                    diarization_kwargs["min_speakers"] = min_speakers
                    diarization_kwargs["max_speakers"] = max_speakers
                
                diarization_result = st.session_state.diarization_pipeline(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    **diarization_kwargs
                )
        
        # Step 2: Speech recognition by speaker segments
        with progress_container:
            st.spinner("💬 Transcribing speech by speakers...")
            
            # Group segments by speaker
            speaker_segments = {}
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))
            
            # Initialize progress display
            total_segments = sum(len(segs) for speaker, segs in speaker_segments.items())
            current_segment = 0
            
            # Create initial progress display
            update_progress(
                progress_container,
                current_segment,
                total_segments,
                "Initial",
                0,
                1
            )
            
            # Transcribe each speaker's segments
            transcription_result = []
            speaker_texts = {}
            
            for speaker, segments in speaker_segments.items():
                speaker_texts[speaker] = []
                
                for idx, (start, end) in enumerate(segments):
                    current_segment += 1
                    progress = current_segment / total_segments
                    
                    # Update progress display
                    update_progress(
                        progress_container,
                        current_segment,
                        total_segments,
                        speaker,
                        idx,
                        len(segments)
                    )
                    
                    try:
                        # Extract audio segment
                        audio = AudioSegment.from_wav(audio_path)
                        segment = audio[int(start * 1000):int(end * 1000)]
                        
                        # Save segment to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_seg:
                            segment.export(tmp_seg.name, format="wav")
                            seg_path = tmp_seg.name
                        
                        # Transcribe segment
                        segments_result, info = st.session_state.whisper_model.transcribe(
                            seg_path,
                            beam_size=beam_size,
                            best_of=best_of,
                            temperature=temperature,
                            language=st.session_state.language if st.session_state.language else None,
                            vad_filter=vad_filter,
                            word_timestamps=True
                        )
                        
                        # Combine transcribed text
                        text = " ".join([segment.text for segment in segments_result]).strip()
                        
                        if text:
                            transcription_result.append({
                                'speaker': speaker,
                                'start': start,
                                'end': end,
                                'text': text,
                                'duration': end - start
                            })
                            speaker_texts[speaker].append({
                                'start': start,
                                'end': end,
                                'text': text,
                                'duration': end - start
                            })
                        
                        # Clean up temporary file
                        os.unlink(seg_path)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error transcribing segment for {speaker} ({start:.1f}s-{end:.1f}s): {str(e)}")
                        continue
        
        # Clear progress container
        if st.session_state.progress_container is not None:
            st.session_state.progress_container.empty()
        
        # Create play format
        play_format = []
        all_segments = sorted(transcription_result, key=lambda x: x['start'])
        
        for i, item in enumerate(all_segments, 1):
            # Format time based on settings
            if format_time_as_hhmmss:
                start_time = time.strftime('%H:%M:%S', time.gmtime(item['start'])) + f".{int(item['start'] % 1 * 100):02d}"
                end_time = time.strftime('%H:%M:%S', time.gmtime(item['end'])) + f".{int(item['end'] % 1 * 100):02d}"
            else:
                start_time = f"{item['start']:.2f}"
                end_time = f"{item['end']:.2f}"
            
            timestamp = f"[{start_time} - {end_time}]" if include_timestamps else ""
            play_format.append({
                'number': i,
                'speaker': item['speaker'],
                'text': item['text'],
                'timestamp': timestamp,
                'start': item['start'],
                'end': item['end']
            })
        
        # Calculate speaker statistics
        speaker_stats = {}
        for speaker, segments in speaker_texts.items():
            total_duration = sum(seg['duration'] for seg in segments)
            total_utterances = len(segments)
            avg_duration = total_duration / total_utterances if total_utterances > 0 else 0
            
            speaker_stats[speaker] = {
                'total_duration': total_duration,
                'total_utterances': total_utterances,
                'average_duration': avg_duration,
                'transcript': " ".join([seg['text'] for seg in segments])
            }
        
        st.session_state.processing_result = {
            'diarization': diarization_result,
            'transcription': transcription_result,
            'speaker_texts': speaker_texts,
            'play_format': play_format,
            'speaker_stats': speaker_stats,
            'audio_info': st.session_state.audio_info
        }
        
        return True
        
    except Exception as e:
        if st.session_state.progress_container is not None:
            st.session_state.progress_container.empty()
        st.error(f"❌ Error in full pipeline: {str(e)}")
        return False

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🎙️ Record Audio")
    
    # Force update state if needed
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    # Record audio
    audio = mic_recorder(
        start_prompt="⏺️ Start Recording", 
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key=f'mic_recorder_{st.session_state.recording_key}'
    )
    
    if audio is not None:
        st.session_state.audio_bytes = audio['bytes']
        st.success("✅ Audio recorded successfully!")
        st.audio(st.session_state.audio_bytes, format="audio/wav")
    
    # Upload audio file option
    st.subheader("📁 Or Upload Audio File")
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg', 'm4a', 'flac', 'webm'])
    
    if uploaded_file is not None:
        st.session_state.audio_bytes = uploaded_file.getvalue()
        st.success("✅ File uploaded successfully!")
        st.audio(st.session_state.audio_bytes, format="audio/wav")
    
    # Clear button
    if st.button("🗑️ Clear Audio", type="secondary", use_container_width=True):
        st.session_state.audio_bytes = None
        st.session_state.processing_done = False
        st.session_state.processing_result = None
        st.session_state.audio_info = None
        st.session_state.recording_key = str(time.time())
        st.rerun()
    
    # Show model status
    if hf_token:
        st.success("✅ Hugging Face token configured")
    else:
        st.error("❌ Hugging Face token not configured. Set HF_TOKEN in your .env file.")
    
    if st.session_state.models_loaded:
        st.success("✅ Models loaded successfully")
    else:
        st.info("⏳ Models will load automatically when you process audio")
    
    # Process button
    if st.session_state.audio_bytes is not None:
        if not st.session_state.models_loaded:
            if st.button("🔄 Load Models", use_container_width=True):
                load_models()
                st.rerun()
        else:
            if st.button("🚀 Process Audio", type="primary", use_container_width=True, disabled=st.session_state.processing_done):
                # Process audio
                waveform, sr, audio_path = process_audio(st.session_state.audio_bytes, sample_rate)
                
                if waveform is not None and audio_path is not None:
                    # Run full pipeline
                    if process_full_pipeline(audio_path, waveform, sr):
                        st.session_state.processing_done = True
                        st.rerun()
                else:
                    st.error("❌ Failed to process audio")

with col2:
    st.header("📊 Results")
    
    if not st.session_state.processing_done:
        st.info("ℹ️ Record or upload audio, then click 'Process Audio' to start diarization and transcription.")
        
        # Show model status
        if hf_token:
            st.success("✅ Hugging Face token configured")
        else:
            st.error("❌ Hugging Face token not configured. Set HF_TOKEN in your .env file.")
        
        if st.session_state.models_loaded:
            st.success("✅ Models loaded successfully")
        else:
            st.info("⏳ Models will load automatically when you process audio")
    else:
        result = st.session_state.processing_result
        
        # Show speaker statistics if enabled
        if include_speaker_statistics:
            st.subheader("📈 Speaker Statistics")
            stats_cols = st.columns(len(result['speaker_stats']))
            
            for idx, (speaker, stats) in enumerate(result['speaker_stats'].items()):
                with stats_cols[idx % len(stats_cols)]:
                    st.markdown(f"""
                    <div class="speaker-box">
                        <h4>{speaker}</h4>
                        <p><strong>Total speech time:</strong> {stats['total_duration']:.1f} seconds</p>
                        <p><strong>Number of utterances:</strong> {stats['total_utterances']}</p>
                        <p><strong>Average utterance length:</strong> {stats['average_duration']:.1f} seconds</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Play format view
        st.subheader("🎭 Dialogue (Play Format)")
        
        for item in result['play_format']:
            timestamp_display = f"{item['timestamp']} " if item['timestamp'] else ""
            st.markdown(f"""
            <div class="play-format">
                <strong>{item['number']:02d}. {timestamp_display}{item['speaker']}:</strong><br>
                {item['text']}
            </div>
            """, unsafe_allow_html=True)
        
        # Individual speaker transcripts
        st.subheader("👤 Individual Speaker Transcripts")
        
        for speaker, segments in result['speaker_texts'].items():
            with st.expander(f"📝 {speaker} Full Transcript"):
                full_text = " ".join([seg['text'] for seg in segments])
                st.write(f"**{speaker}:** {full_text}")
        
        # Download section
        st.subheader("📥 Download Results")
        
        # Create downloadable files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Play format text
        play_text = f"DIALOGUE TRANSCRIPT ({timestamp})\n" + "=" * 60 + "\n\n"
        for item in result['play_format']:
            timestamp_display = f"{item['timestamp']} " if item['timestamp'] else ""
            play_text += f"{item['number']:02d}. {timestamp_display}{item['speaker']}:\n    {item['text']}\n\n"
        
        # Speaker transcripts
        speaker_texts = {}
        for speaker, segments in result['speaker_texts'].items():
            content = f"TRANSCRIPT FOR {speaker} ({timestamp})\n" + "=" * 50 + "\n\n"
            for i, seg in enumerate(segments, 1):
                content += f"{i:02d}. [{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n"
                content += f"    Duration: {seg['duration']:.2f}s\n\n"
            speaker_texts[speaker] = content
        
        # RTTM format
        rttm_content = ""
        for turn, _, speaker in result['diarization'].itertracks(yield_label=True):
            rttm_content += f"SPEAKER audio 1 {turn.start:.3f} {turn.duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
        
        # Create ZIP file with all formats
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zip_file:
            # Add play format
            zip_file.writestr(f"dialogue_play_format_{timestamp}.txt", play_text)
            
            # Add speaker transcripts
            for speaker, content in speaker_texts.items():
                safe_speaker = speaker.replace(" ", "_").replace("/", "_")
                zip_file.writestr(f"{safe_speaker}_transcript_{timestamp}.txt", content)
            
            # Add RTTM file
            zip_file.writestr(f"diarization_results_{timestamp}.rttm", rttm_content)
            
            # Add JSON with all data
            json_data = {
                'metadata': {
                    'processed_at': timestamp,
                    'audio_duration': result['audio_info']['duration'],
                    'sample_rate': result['audio_info']['sample_rate'],
                    'models': {
                        'diarization': 'pyannote/speaker-diarization-3.1',
                        'asr': f'faster-whisper-{whisper_model_size}'
                    },
                    'parameters': {
                        'beam_size': beam_size,
                        'language': language,
                        'num_speakers': num_speakers,
                        'min_speakers': min_speakers,
                        'max_speakers': max_speakers
                    }
                },
                'results': {
                    'transcription': result['transcription'],
                    'speaker_stats': {
                        speaker: {
                            'total_duration': stats['total_duration'],
                            'total_utterances': stats['total_utterances'],
                            'average_duration': stats['average_duration']
                        }
                        for speaker, stats in result['speaker_stats'].items()
                    }
                }
            }
            zip_file.writestr(f"results_metadata_{timestamp}.json", json.dumps(json_data, indent=2))
        
        zip_buffer.seek(0)
        
        # Download buttons
        col_download1, col_download2, col_download3, col_download4 = st.columns(4)
        
        with col_download1:
            st.download_button(
                label="🎭 Play Format TXT",
                data=play_text,
                file_name=f"dialogue_play_format_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_download2:
            st.download_button(
                label="👤 Speaker TXTs",
                data="\n\n".join(speaker_texts.values()),
                file_name=f"all_speaker_transcripts_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_download3:
            st.download_button(
                label="📊 RTTM File",
                data=rttm_content,
                file_name=f"diarization_results_{timestamp}.rttm",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_download4:
            st.download_button(
                label="📦 All Results ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"speaker_diarization_results_{timestamp}.zip",
                mime="application/zip",
                use_container_width=True
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>
        <strong>Dockerized Speaker Diarization + ASR</strong><br>
        Powered by pyannote.audio and faster-whisper<br>
        Container ID: {container_id}
    </small>
</div>
""".format(container_id=os.environ.get("HOSTNAME", "unknown")), unsafe_allow_html=True)