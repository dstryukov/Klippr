import os
import logging
import yt_dlp
import ffmpeg
from faster_whisper import WhisperModel
import torch
from config import settings

logger = logging.getLogger(__name__)

class VideoIngestor:
    def __init__(self, temp_dir: str = "tmp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        model_size = settings.WHISPER_MODEL
        requested_device = settings.DEVICE
        
        # Safe fallback if CUDA requested but not available in PyTorch.
        actual_device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and actual_device != "cuda":
            logger.warning("CUDA requested for Whisper, but torch.cuda.is_available() is false. Falling back to CPU.")
        
        if actual_device == "cuda":
            # GTX 10-series (Pascal) has compute capability 6.1 and is more stable with float32.
            major, _ = torch.cuda.get_device_capability()
            compute_type = "float16" if major >= 7 else "float32"
            device_name = torch.cuda.get_device_name(0)
        else:
            compute_type = "int8"
            device_name = "CPU"
        
        logger.info(
            "Initializing faster-whisper model '%s' on %s (%s, compute_type=%s)...",
            model_size,
            actual_device,
            device_name,
            compute_type,
        )
        self.whisper_model = WhisperModel(model_size, device=actual_device, compute_type=compute_type)
        logger.info("Whisper model initialized.")

    def download_video(self, url: str) -> str:
        logger.info(f"Downloading video from {url}...")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            # Avoid downloading playlists by accident when a URL contains list=...
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            logger.info(f"Video downloaded successfully: {video_path}")
            return video_path

    def extract_audio(self, video_path: str) -> str:
        logger.info(f"Extracting audio from {video_path}...")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f"{base_name}.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

    def transcribe(self, audio_path: str) -> list[dict]:
        logger.info(f"Transcribing audio from {audio_path} with word timestamps...")
        segments, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        transcript = []
        for segment in segments:
            words = []
            if segment.words:
                for word in segment.words:
                    text = (word.word or "").strip()
                    if not text:
                        continue
                    words.append({
                        "start": float(word.start),
                        "end": float(word.end),
                        "text": text,
                    })

            transcript.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "words": words,
            })
            
        logger.info(f"Transcription complete. Total segments: {len(transcript)}")
        return transcript
