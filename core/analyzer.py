import json
import logging
import openai
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
from config import settings

logger = logging.getLogger(__name__)

class HighlightAnalyzer:
    def __init__(self):
        # OpenRouter Setup
        or_api_key = None
        if settings.OPENROUTER_API_KEY:
            or_api_key = settings.OPENROUTER_API_KEY.get_secret_value()
            
        self.openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=or_api_key or "dummy_or_key",
        )
        self.openrouter_model = settings.OPENROUTER_MODEL

        # Groq Setup (Fallback)
        groq_api_key = None
        if settings.GROQ_API_KEY:
            groq_api_key = settings.GROQ_API_KEY.get_secret_value()
            
        self.groq_client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_api_key or "dummy_groq_key",
        )
        self.groq_model = settings.GROQ_MODEL

    def _seconds_to_timecode(self, seconds: float) -> str:
        """Converts seconds into [HH:MM:SS] format."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    def format_transcript(self, transcript: list[dict]) -> str:
        """Formats the Whisper transcript list into a readable text with timecodes."""
        lines = []
        for segment in transcript:
            start_tc = self._seconds_to_timecode(segment["start"])
            text = segment["text"]
            lines.append(f"{start_tc} {text}")
        return "\n".join(lines)

    def _extract_json_array(self, text: str) -> list[dict]:
        """Robustly extracts JSON array from a string, ignoring conversational fluff."""
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx+1]
            return json.loads(json_str)
        raise ValueError("No JSON array bounds '[' and ']' found in the response")

    @retry(
        wait=wait_fixed(10), 
        stop=stop_after_attempt(3), 
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True
    )
    def _call_llm_with_retry(self, client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
        """Helper to call LLM with automatic retry on 429 Too Many Requests."""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    def _get_highlights_from_llm(self, system_prompt: str, formatted_text: str) -> list[dict]:
        """Handles 1 extra fallback loop on top of tenacity for JSON parse errors and provider fallbacks."""
        for attempt in range(2):
            content = ""
            try:
                try:
                    logger.info(f"Sending to OpenRouter (Model: {self.openrouter_model}, Attempt {attempt + 1})...")
                    content = self._call_llm_with_retry(self.openrouter_client, self.openrouter_model, system_prompt, formatted_text)
                except Exception as e:
                    logger.warning(f"OpenRouter API failed: {e}. Falling back to Groq (Model: {self.groq_model})...")
                    content = self._call_llm_with_retry(self.groq_client, self.groq_model, system_prompt, formatted_text)
                
                parsed_clips = self._extract_json_array(content)
                logger.info(f"Successfully extracted {len(parsed_clips)} highlights.")
                return parsed_clips
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON Parse error on attempt {attempt + 1}: {e}\nContent was: {content}")
                if attempt == 1:
                    logger.error("Failed to parse JSON after 2 attempts.")
                    raise ValueError("LLM returned invalid JSON")
            except ValueError as e:
                logger.warning(f"Validation error on attempt {attempt + 1}: {e}\nContent was: {content}")
                if attempt == 1:
                    raise
            except Exception as e:
                logger.error(f"Error calling LLM APIs (both failed): {e}")
                raise
        return []

    def chunk_transcript(self, transcript: list[dict], chunk_duration_sec: float = 900) -> list[list[dict]]:
        """Splits the transcript into chunks of max `chunk_duration_sec` seconds (e.g., 15 minutes)."""
        chunks = []
        current_chunk = []
        current_start = 0.0
        
        for segment in transcript:
            if not current_chunk:
                current_start = segment["start"]
                
            if segment["start"] - current_start > chunk_duration_sec:
                chunks.append(current_chunk)
                current_chunk = [segment]
                current_start = segment["start"]
            else:
                current_chunk.append(segment)
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def find_highlights(self, transcript: list[dict], num_clips: int = 3) -> list[dict]:
        """
        Sends chunks of transcript to LLM to identify the best moments.
        Returns a parsed JSON array of clip dictionaries.
        """
        chunks = self.chunk_transcript(transcript, chunk_duration_sec=900)
        logger.info(f"Transcript split into {len(chunks)} chunks (15 min max).")
        
        system_prompt = f"""Ты — профессиональный AI-продюсер TikTok и YouTube Shorts. Твоя задача — найти в транскрипте видео самые виральные, эмоциональные и интересные моменты для нарезки.
КРИТЕРИИ ОТБОРА:
Завершенность мысли: Кусок должен иметь четкое начало и конец. Не вырезай фразы на полуслове.
Эмоция или Инсайт: Ищи моменты смеха, спора, шока, или очень сильные полезные советы.
Хук (Крючок): Первые 3 секунды клипа должны цеплять зрителя.
ФОРМАТ ОТВЕТА:
Верни СТРОГО валидный JSON-массив без markdown. Каждый объект должен содержать:
start_time: float (в секундах, из таймкода)
end_time: float (в секундах)
title: string (цепляющий заголовок для клипа, до 50 символов)
reason: string (почему это вирально, 1 предложение)
Найди ровно {num_clips} лучших моментов."""

        all_highlights = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            formatted_text = self.format_transcript(chunk)
            try:
                clips = self._get_highlights_from_llm(system_prompt, formatted_text)
                all_highlights.extend(clips)
            except Exception as e:
                logger.error(f"Skipping chunk {i+1} due to error: {e}")
                
        # For MVP: just return the first `num_clips` we collected across chunks
        return all_highlights[:num_clips]

    def snap_to_silence(self, highlights: list[dict], audio_path: str, transcript: list[dict]) -> list[dict]:
        """
        Snaps highlight boundaries to the nearest silence to avoid cutting words in half.
        Also uses the transcript to expand boundaries if a boundary cuts a transcript segment.
        """
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence
        except ImportError:
            logger.error("pydub is not installed. Please install it to use silence snapping.")
            return highlights

        logger.info(f"Snapping {len(highlights)} highlights to silence using audio: {audio_path}")
        try:
            audio = AudioSegment.from_file(audio_path)
        except Exception as e:
            logger.error(f"Failed to load audio for silence detection: {e}")
            return highlights
            
        adjusted_highlights = []
        
        def find_nearest_silence(target_ms: int, window_ms: int = 2000) -> int:
            search_start = max(0, target_ms - window_ms)
            search_end = min(len(audio), target_ms + window_ms)
            
            if search_start >= search_end:
                return target_ms
                
            chunk = audio[search_start:search_end]
            
            silences = detect_silence(chunk, min_silence_len=200, silence_thresh=chunk.dBFS - 16)
            
            if not silences:
                return target_ms
                
            closest_silence = target_ms
            min_dist = float('inf')
            
            for s in silences:
                silence_center = (s[0] + s[1]) / 2.0 + search_start
                dist = abs(silence_center - target_ms)
                if dist < min_dist:
                    min_dist = dist
                    closest_silence = int(silence_center)
                    
            return closest_silence

        for clip in highlights:
            adjusted_clip = clip.copy()
            
            start_sec = float(adjusted_clip["start_time"])
            end_sec = float(adjusted_clip["end_time"])
            
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            new_start_ms = find_nearest_silence(start_ms)
            new_end_ms = find_nearest_silence(end_ms)
            
            new_start_sec = new_start_ms / 1000.0
            new_end_sec = new_end_ms / 1000.0
            
            for seg in transcript:
                seg_start = seg["start"]
                seg_end = seg["end"]
                
                if seg_start + 0.1 < new_start_sec < seg_end - 0.1:
                    new_start_sec = seg_start
                
                if seg_start + 0.1 < new_end_sec < seg_end - 0.1:
                    new_end_sec = seg_end
            
            adjusted_clip["start_time"] = round(new_start_sec, 2)
            adjusted_clip["end_time"] = round(new_end_sec, 2)
            adjusted_highlights.append(adjusted_clip)
            
        return adjusted_highlights
