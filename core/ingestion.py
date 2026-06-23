import os
import logging
import yt_dlp
import ffmpeg
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class VideoIngestor:
    def __init__(self, temp_dir: str = "tmp"):
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        # Initialize whisper model. Using 'base' model for decent speed/accuracy balance.
        logger.info("Initializing faster-whisper model ('base')...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("Whisper model initialized.")

    def download_video(self, url: str) -> str:
        """Downloads a video from YouTube (or other sources) using yt-dlp."""
        logger.info(f"Downloading video from {url}...")
        ydl_opts = {
            'format': 'best',
            'outtmpl': os.path.join(self.temp_dir, '%(id)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                logger.info(f"Video downloaded successfully: {video_path}")
                return video_path
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise

    def extract_audio(self, video_path: str) -> str:
        """Extracts audio to .wav format (16kHz, mono) using ffmpeg."""
        logger.info(f"Extracting audio from {video_path}...")
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(self.temp_dir, f"{base_name}.wav")
        
        try:
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Audio extracted successfully: {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"Error extracting audio: {error_msg}")
            raise RuntimeError(f"FFmpeg error: {error_msg}")
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise

    def transcribe(self, audio_path: str) -> list[dict]:
        """Transcribes audio using faster-whisper."""
        logger.info(f"Transcribing audio from {audio_path}...")
        try:
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
            logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            
            result = []
            for segment in segments:
                result.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end
                })
            logger.info(f"Transcription completed. Found {len(result)} segments.")
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
