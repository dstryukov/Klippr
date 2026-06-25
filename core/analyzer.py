import json
import logging
import openai
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
from config import settings

logger = logging.getLogger(__name__)

class HighlightAnalyzer:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        
        if self.provider == "openrouter":
            or_api_key = settings.OPENROUTER_API_KEY.get_secret_value() if settings.OPENROUTER_API_KEY else "dummy_or_key"
            self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_api_key)
        else:
            groq_api_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else "dummy_groq_key"
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

    def _seconds_to_timecode(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"[{h:02d}:{m:02d}:{s:02d}]"

    def format_transcript(self, transcript: list[dict]) -> str:
        lines = []
        for segment in transcript:
            start_tc = self._seconds_to_timecode(segment["start"])
            text = segment["text"]
            lines.append(f"{start_tc} {text}")
        return "\n".join(lines)

    def _extract_json_array(self, text: str) -> list[dict]:
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
        for attempt in range(2):
            content = ""
            try:
                try:
                    logger.info(f"Sending to {self.provider} (Model: {self.model}, Attempt {attempt + 1})...")
                    content = self._call_llm_with_retry(self.client, self.model, system_prompt, formatted_text)
                except Exception as e:
                    logger.error(f"API Request failed: {e}")
                    raise
                
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
                logger.error(f"Error calling LLM APIs: {e}")
                raise
        return []

    def chunk_transcript(self, transcript: list[dict], chunk_duration_sec: float = 900) -> list[list[dict]]:
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

    def _validate_and_fix_highlights(self, highlights: list[dict], transcript: list[dict]) -> list[dict]:
        if not transcript:
            return highlights
            
        max_duration = transcript[-1]["end"]
        valid_highlights = []
        
        # Read from settings dynamically
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        
        for clip in highlights:
            try:
                start_sec = float(clip.get("start_time", 0))
                end_sec = float(clip.get("end_time", 0))
            except (ValueError, TypeError):
                continue
                
            start_sec = max(0.0, start_sec)
            end_sec = min(max_duration, end_sec)
            
            if start_sec >= end_sec:
                continue
                
            duration = end_sec - start_sec
            
            if duration < min_clip_len:
                start_idx = 0
                end_idx = len(transcript) - 1
                
                min_start_diff = float('inf')
                for i, seg in enumerate(transcript):
                    diff = abs(seg["start"] - start_sec)
                    if diff < min_start_diff:
                        min_start_diff = diff
                        start_idx = i
                        
                min_end_diff = float('inf')
                for i, seg in enumerate(transcript):
                    diff = abs(seg["end"] - end_sec)
                    if diff < min_end_diff:
                        min_end_diff = diff
                        end_idx = i
                        
                while (transcript[end_idx]["end"] - transcript[start_idx]["start"]) < min_clip_len:
                    expanded = False
                    if start_idx > 0:
                        start_idx -= 1
                        expanded = True
                    if end_idx < len(transcript) - 1 and (transcript[end_idx]["end"] - transcript[start_idx]["start"]) < min_clip_len:
                        end_idx += 1
                        expanded = True
                        
                    if not expanded:
                        break
                        
                start_sec = transcript[start_idx]["start"]
                end_sec = transcript[end_idx]["end"]
                duration = end_sec - start_sec

            elif duration > max_clip_len:
                start_idx = 0
                min_start_diff = float('inf')
                for i, seg in enumerate(transcript):
                    diff = abs(seg["start"] - start_sec)
                    if diff < min_start_diff:
                        min_start_diff = diff
                        start_idx = i
                        
                end_idx = start_idx
                while end_idx < len(transcript) - 1:
                    next_end = transcript[end_idx + 1]["end"]
                    if next_end - transcript[start_idx]["start"] > max_clip_len:
                        break
                    end_idx += 1
                    
                start_sec = transcript[start_idx]["start"]
                end_sec = transcript[end_idx]["end"]
                duration = end_sec - start_sec

            if duration >= min_clip_len:
                clip["start_time"] = round(start_sec, 2)
                clip["end_time"] = round(end_sec, 2)
                valid_highlights.append(clip)
            else:
                logger.warning(f"Dropped highlight '{clip.get('title')}' - could not expand to {min_clip_len} seconds.")
                
        return valid_highlights

    def find_highlights(self, transcript: list[dict], num_clips: int = None) -> list[dict]:
        if num_clips is None:
            num_clips = settings.NUM_CLIPS
            
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
            
        chunks = self.chunk_transcript(transcript, chunk_duration_sec=900)
        logger.info(f"Transcript split into {len(chunks)} chunks (15 min max).")
        
        system_prompt = f"""Ты — профессиональный AI-продюсер TikTok и YouTube Shorts. Твоя задача — найти в транскрипте видео самые виральные моменты для нарезки на клипы.

КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
1. ДЛИНА КЛИПА: Каждый клип должен быть ОТ {min_clip_len} ДО {max_clip_len} СЕКУНД. Не больше, не меньше!
2. ЗАВЕРШЕННОСТЬ: Клип должен иметь четкое начало и конец. Не вырезай фразы на полуслове.
3. ХУК: Первые 3-5 секунд клипа должны цеплять зрителя (вопрос, шок, инсайт).
4. КОНТЕКСТ: Если момент короткий, расширь его, добавив контекст до и после.

КРИТЕРИИ ОТБОРА (в порядке важности):
- Сильная эмоция (смех, шок, спор, удивление)
- Полезный инсайт или совет
- Интересная история или пример
- Неожиданный поворот мысли

ФОРМАТ ОТВЕТА:
Верни СТРОГО валидный JSON-массив без markdown. Каждый объект:
{{
  "start_time": float (в секундах, минимум {min_clip_len} сек до end_time),
  "end_time": float (в секундах, максимум {max_clip_len} сек от start_time),
  "title": string (цепляющий заголовок, до 50 символов),
  "reason": string (почему это вирально, 1 предложение)
}}

Найди ровно {num_clips} лучших моментов. Если видео слишком короткое для {num_clips} клипов по {min_clip_len} секунд, верни столько, сколько возможно."""

        all_highlights = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
            formatted_text = self.format_transcript(chunk)
            try:
                clips = self._get_highlights_from_llm(system_prompt, formatted_text)
                clips = self._validate_and_fix_highlights(clips, transcript)
                all_highlights.extend(clips)
            except Exception as e:
                logger.error(f"Skipping chunk {i+1} due to error: {e}")
                
        return all_highlights[:num_clips]

    def snap_to_silence(self, highlights: list[dict], audio_path: str, transcript: list[dict]) -> list[dict]:
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
            
            if new_end_sec - new_start_sec >= (settings.MIN_CLIP_DURATION - 0.5):
                adjusted_clip["start_time"] = round(new_start_sec, 2)
                adjusted_clip["end_time"] = round(new_end_sec, 2)
                adjusted_highlights.append(adjusted_clip)
            else:
                adjusted_highlights.append(clip)
            
        return adjusted_highlights
