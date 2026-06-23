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
        device = settings.DEVICE
        
        # Safe fallback if CUDA requested but not available
        actual_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        compute_type = "float16" if actual_device == "cuda" else "int8"
        
        logger.info(f"Initializing faster-whisper model ('{model_size}') on {actual_device}...")
        self.whisper_model = WhisperModel(model_size, device=actual_device, compute_type=compute_type)
        logger.info("Whisper model initialized.")

    def download_video(self, url: str) -> str:
        logger.info(f"Downloading video from {url}...")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
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
        logger.info(f"Transcribing audio from {audio_path}...")
        segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        transcript = []
        for segment in segments:
            transcript.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            
        logger.info(f"Transcription complete. Total segments: {len(transcript)}")
        return transcript
