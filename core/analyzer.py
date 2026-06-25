import json
import logging
import math
import re
from typing import Any

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
            end_tc = self._seconds_to_timecode(segment["end"])
            text = segment.get("text", "").strip()
            if text:
                lines.append(f"{start_tc}-{end_tc} {text}")
        return "\n".join(lines)

    def _extract_json_array(self, text: str) -> list[dict]:
        # Strip common markdown fences, then take the broadest JSON array.
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        start_idx = cleaned.find('[')
        end_idx = cleaned.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx:end_idx + 1]
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
        raise ValueError("No valid JSON array found in the response")

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(openai.RateLimitError),
        reraise=True,
    )
    def _call_llm_with_retry(self, client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.25,
        )
        return response.choices[0].message.content.strip()

    def _get_highlights_from_llm(self, system_prompt: str, formatted_text: str) -> list[dict]:
        for attempt in range(2):
            content = ""
            try:
                logger.info(f"Sending to {self.provider} (Model: {self.model}, Attempt {attempt + 1})...")
                content = self._call_llm_with_retry(self.client, self.model, system_prompt, formatted_text)
                parsed_clips = self._extract_json_array(content)
                logger.info(f"Successfully extracted {len(parsed_clips)} highlight candidates.")
                return parsed_clips
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}\nContent was: {content}")
                if attempt == 1:
                    raise ValueError("LLM returned invalid JSON")
            except Exception as e:
                logger.error(f"Error calling/parsing LLM response: {e}")
                if attempt == 1:
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
            return []
            
        max_duration = transcript[-1]["end"]
        valid_highlights = []
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

            start_idx = min(range(len(transcript)), key=lambda i: abs(transcript[i]["start"] - start_sec))
            end_idx = min(range(len(transcript)), key=lambda i: abs(transcript[i]["end"] - end_sec))
            if end_idx < start_idx:
                end_idx = start_idx

            duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]
            while duration < min_clip_len:
                expanded = False
                if start_idx > 0:
                    start_idx -= 1
                    expanded = True
                duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]
                if duration < min_clip_len and end_idx < len(transcript) - 1:
                    end_idx += 1
                    expanded = True
                duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]
                if not expanded:
                    break

            while transcript[end_idx]["end"] - transcript[start_idx]["start"] > max_clip_len and end_idx > start_idx:
                end_idx -= 1

            start_sec = transcript[start_idx]["start"]
            end_sec = transcript[end_idx]["end"]
            duration = end_sec - start_sec

            if min_clip_len <= duration <= max_clip_len:
                normalized = dict(clip)
                normalized["start_time"] = round(start_sec, 2)
                normalized["end_time"] = round(end_sec, 2)
                normalized["title"] = str(normalized.get("title") or "Крутой момент")[:80]
                normalized["reason"] = str(normalized.get("reason") or "Потенциально сильный фрагмент")[:240]
                try:
                    normalized["score"] = float(normalized.get("score", 50))
                except (ValueError, TypeError):
                    normalized["score"] = 50.0
                valid_highlights.append(normalized)
            else:
                logger.warning("Dropped highlight '%s' because duration %.1fs is outside %s-%ss.", clip.get("title"), duration, min_clip_len, max_clip_len)
        return valid_highlights

    def _remove_overlaps(self, highlights: list[dict]) -> list[dict]:
        selected: list[dict] = []
        for clip in sorted(highlights, key=lambda x: float(x.get("score", 0)), reverse=True):
            start = float(clip["start_time"])
            end = float(clip["end_time"])
            overlaps = False
            for chosen in selected:
                c_start = float(chosen["start_time"])
                c_end = float(chosen["end_time"])
                overlap = max(0.0, min(end, c_end) - max(start, c_start))
                shorter = max(1.0, min(end - start, c_end - c_start))
                if overlap / shorter > 0.35:
                    overlaps = True
                    break
            if not overlaps:
                selected.append(clip)
        return sorted(selected, key=lambda x: x["start_time"])

    def _fallback_highlights(self, transcript: list[dict], num_clips: int) -> list[dict]:
        """Simple local fallback when the LLM fails or returns too few candidates."""
        min_len = settings.MIN_CLIP_DURATION
        max_len = settings.MAX_CLIP_DURATION
        if not transcript:
            return []

        scored = []
        keywords = (
            "важно", "секрет", "ошибка", "проблем", "почему", "как", "деньги", "результат",
            "никогда", "всегда", "главное", "инсайт", "смотри", "представь", "шок", "лучше",
        )
        for i, seg in enumerate(transcript):
            text = seg.get("text", "").strip()
            if not text:
                continue
            words = text.split()
            density = len(words) / max(1.0, seg["end"] - seg["start"])
            keyword_hits = sum(1 for k in keywords if k.lower() in text.lower())
            punctuation = text.count("?") * 2 + text.count("!")
            score = density + keyword_hits * 3 + punctuation
            scored.append((score, i))

        candidates = []
        for _, idx in sorted(scored, reverse=True)[: max(num_clips * 3, 6)]:
            start_idx = idx
            end_idx = idx
            while transcript[end_idx]["end"] - transcript[start_idx]["start"] < min_len:
                left_room = start_idx > 0
                right_room = end_idx < len(transcript) - 1
                if not left_room and not right_room:
                    break
                if right_room:
                    end_idx += 1
                if transcript[end_idx]["end"] - transcript[start_idx]["start"] >= min_len:
                    break
                if left_room:
                    start_idx -= 1
            while transcript[end_idx]["end"] - transcript[start_idx]["start"] > max_len and end_idx > start_idx:
                end_idx -= 1
            text = " ".join(s.get("text", "").strip() for s in transcript[start_idx:end_idx + 1])
            candidates.append({
                "start_time": transcript[start_idx]["start"],
                "end_time": transcript[end_idx]["end"],
                "title": (text[:55] + "…") if len(text) > 55 else text or "Клип",
                "reason": "Локальный fallback выбрал плотный фрагмент с потенциальным хуком.",
                "score": 35,
            })
        return self._validate_and_fix_highlights(candidates, transcript)

    def find_highlights(self, transcript: list[dict], num_clips: int = None) -> list[dict]:
        if num_clips is None:
            num_clips = settings.NUM_CLIPS
            
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        chunks = self.chunk_transcript(transcript, chunk_duration_sec=900)
        logger.info(f"Transcript split into {len(chunks)} chunks (15 min max).")

        candidates_per_chunk = max(num_clips * 2, 6)
        system_prompt = f"""Ты — профессиональный продюсер коротких видео уровня TikTok/Reels/YouTube Shorts.
Твоя задача — выбрать НЕ просто информативные куски, а моменты, которые зритель досмотрит и сохранит.

Критически важно:
1. Каждый клип должен быть от {min_clip_len} до {max_clip_len} секунд.
2. У клипа должен быть сильный хук в первые 3-5 секунд: вопрос, конфликт, неожиданный тезис, эмоция или конкретная польза.
3. Не выбирай вступления, воду, приветствия, технические паузы и фрагменты без законченной мысли.
4. Выбирай фрагменты с самодостаточным смыслом: зритель должен понять контекст без полного видео.
5. Таймкоды должны быть в секундах от начала исходного видео.

Верни СТРОГО валидный JSON-массив без markdown. Каждый объект:
{{
  "start_time": float,
  "end_time": float,
  "title": "короткий цепляющий заголовок до 50 символов",
  "reason": "почему этот фрагмент удержит внимание",
  "score": число от 1 до 100
}}

Верни до {candidates_per_chunk} лучших кандидатов из этого куска транскрипта, отсортированных по score."""

        all_candidates: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}...")
            formatted_text = self.format_transcript(chunk)
            try:
                clips = self._get_highlights_from_llm(system_prompt, formatted_text)
                clips = self._validate_and_fix_highlights(clips, transcript)
                all_candidates.extend(clips)
            except Exception as e:
                logger.error(f"Skipping LLM highlights for chunk {i + 1} due to error: {e}")

        if len(all_candidates) < num_clips:
            logger.warning("LLM returned too few candidates (%s/%s). Adding local fallback candidates.", len(all_candidates), num_clips)
            all_candidates.extend(self._fallback_highlights(transcript, num_clips))

        ranked = self._remove_overlaps(all_candidates)
        ranked = sorted(ranked, key=lambda x: float(x.get("score", 0)), reverse=True)[:num_clips]
        ranked = sorted(ranked, key=lambda x: x["start_time"])
        logger.info("Selected %s final highlights.", len(ranked))
        return ranked

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
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        
        def find_nearest_silence(target_ms: int, window_ms: int = 2000) -> int:
            search_start = max(0, target_ms - window_ms)
            search_end = min(len(audio), target_ms + window_ms)
            if search_start >= search_end:
                return target_ms
            chunk = audio[search_start:search_end]
            silence_thresh = min(chunk.dBFS - 12, -35) if math.isfinite(chunk.dBFS) else -40
            silences = detect_silence(chunk, min_silence_len=180, silence_thresh=silence_thresh)
            if not silences:
                return target_ms
            return min((int((s[0] + s[1]) / 2.0 + search_start) for s in silences), key=lambda x: abs(x - target_ms))

        for clip in highlights:
            adjusted_clip = clip.copy()
            start_sec = float(adjusted_clip["start_time"])
            end_sec = float(adjusted_clip["end_time"])
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            new_start_sec = find_nearest_silence(start_ms) / 1000.0
            new_end_sec = find_nearest_silence(end_ms) / 1000.0
            
            for seg in transcript:
                seg_start = seg["start"]
                seg_end = seg["end"]
                if seg_start + 0.1 < new_start_sec < seg_end - 0.1:
                    new_start_sec = seg_start
                if seg_start + 0.1 < new_end_sec < seg_end - 0.1:
                    new_end_sec = seg_end
            
            duration = new_end_sec - new_start_sec
            if min_clip_len - 0.5 <= duration <= max_clip_len + 1.5:
                adjusted_clip["start_time"] = round(new_start_sec, 2)
                adjusted_clip["end_time"] = round(new_end_sec, 2)
            adjusted_highlights.append(adjusted_clip)
        return adjusted_highlights
