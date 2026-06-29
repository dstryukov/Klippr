import os
import logging
import time
import yt_dlp
import ffmpeg
from faster_whisper import WhisperModel
import torch
from config import settings
from typing import Any

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
        t0 = time.monotonic()
        self.whisper_model = WhisperModel(model_size, device=actual_device, compute_type=compute_type)
        logger.info("Whisper model initialized in %.1fs.", time.monotonic() - t0)

    @staticmethod
    def _validate_url(url: str) -> None:
        """Basic sanity check before handing the URL to yt-dlp."""
        if not url or not url.strip():
            raise ValueError("Video URL is empty")
        blocked = ["playlist", "channel", "/c/", "/@"]
        lower = url.lower()
        for pattern in blocked:
            if pattern in lower and "watch" not in lower and "shorts" not in lower:
                raise ValueError(f"URL looks like a playlist/channel link, not a single video: {url}")

    def download_video(self, url: str) -> str:
        self._validate_url(url)
        logger.info(f"Downloading video from {url}...")

        # Reuse an already-downloaded file in the temp directory.
        for fname in os.listdir(self.temp_dir):
            if fname.endswith((".mp4", ".mkv", ".webm")):
                existing = os.path.join(self.temp_dir, fname)
                if os.path.getsize(existing) > 0:
                    logger.info("Video already downloaded, reusing: %s", existing)
                    return existing

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': os.path.join(self.temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            # Avoid downloading playlists by accident when a URL contains list=...
            'noplaylist': True,
        }
        t0 = time.monotonic()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            logger.info("Video downloaded in %.1fs: %s", time.monotonic() - t0, video_path)
            return video_path

    def extract_audio(self, video_path: str) -> str:
        logger.info(f"Extracting audio from {video_path}...")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f"{base_name}.wav")

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info("Audio already extracted, reusing: %s", audio_path)
            return audio_path

        try:
            t0 = time.monotonic()
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info("Audio extracted in %.1fs: %s", time.monotonic() - t0, audio_path)
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")

    def transcribe(self, audio_path: str, job: Any = None) -> list[dict]:
        logger.info(f"Transcribing audio from {audio_path}...")
        t0 = time.monotonic()
        segments, info = self.whisper_model.transcribe(
            audio_path,
            beam_size=2,  # Reduced from 5 to 2 for 2-3x speedup with almost no accuracy loss
            vad_filter=True,
            word_timestamps=True,
        )
        
        logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        
        transcript = []
        segment_count = 0
        last_log_time = time.monotonic()
        for segment in segments:
            if job and getattr(job, "cancel_requested", False):
                raise InterruptedError("Transcription cancelled by user")
            segment_count += 1
            now = time.monotonic()
            if now - last_log_time >= 2:
                elapsed = now - t0
                dur = getattr(info, "duration", 0)
                if dur > 0:
                    perc = int((segment.end / dur) * 100)
                    msg = f"Transcribing... {int(segment.end)}s / {int(dur)}s ({perc}%)"
                else:
                    msg = f"Transcribing... {int(segment.end)}s processed"
                    
                if now - getattr(self, "_last_console_log", 0) >= 15:
                    logger.info(msg + f" [{int(elapsed)}s elapsed]")
                    self._last_console_log = now
                    
                if job:
                    job.stage = msg
                    
                last_log_time = now

            words_data = []
            if hasattr(segment, "words") and segment.words:
                words_data = [
                    {
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end)
                    }
                    for w in segment.words
                ]

            transcript.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
                "words": words_data,
            })
            
        logger.info("Transcription complete in %.1fs. Total segments: %d", time.monotonic() - t0, len(transcript))
        return transcript

    def diarize(self, audio_path: str) -> list[dict]:
        """Run speaker diarization using pyannote.audio.

        Returns a list of segments: [{"start": float, "end": float, "speaker": str}, ...]
        Falls back to an empty list if pyannote.audio is not installed or fails.
        """
        if not getattr(settings, "ENABLE_DIARIZATION", True):
            logger.info("Diarization is disabled in config, skipping.")
            return []

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            logger.warning("pyannote.audio is not installed. Skipping diarization.")
            return []

        logger.info("Running speaker diarization on %s...", audio_path)
        t0 = time.monotonic()
        try:
            requested_device = getattr(settings, "DEVICE", "cpu")
            device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"

            hf_token = None
            if getattr(settings, "HF_TOKEN", None):
                hf_token = settings.HF_TOKEN.get_secret_value()

            # Authenticate via huggingface_hub to avoid use_auth_token deprecation issues
            if hf_token:
                try:
                    import huggingface_hub
                    huggingface_hub.login(token=hf_token)
                except ImportError:
                    pass

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            pipeline.to(device)

            diarization = pipeline(audio_path)
            segments: list[dict] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker,
                })

            logger.info("Diarization complete in %.1fs. Total segments: %d", time.monotonic() - t0, len(segments))
            return segments

        except Exception as e:
            logger.warning("Diarization failed (%s). Continuing without speaker info.", e)
            return []
