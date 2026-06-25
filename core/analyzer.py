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
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
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
            temperature=0.2,
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

    def _clip_text(self, transcript: list[dict], start_sec: float, end_sec: float, max_chars: int = 700) -> str:
        parts = []
        for seg in transcript:
            if seg["start"] < end_sec and seg["end"] > start_sec:
                text = seg.get("text", "").strip()
                if text:
                    parts.append(text)
        text = " ".join(parts)
        return (text[:max_chars] + "…") if len(text) > max_chars else text

    def _score_value(self, clip: dict) -> float:
        for key in ("total_score", "score", "retention_score", "hook_score"):
            try:
                return float(clip.get(key))
            except (TypeError, ValueError):
                pass
        return 50.0

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
                logger.warning("Dropped highlight with invalid timestamps: %s", clip)
                continue

            start_sec = max(0.0, start_sec)
            end_sec = min(max_duration, end_sec)
            if start_sec >= end_sec:
                logger.warning("Dropped highlight '%s' because start >= end after clamping.", clip.get("title"))
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
                normalized["duration"] = round(duration, 1)
                normalized["title"] = str(normalized.get("title") or "Крутой момент")[:80]
                normalized["reason"] = str(normalized.get("reason") or "Потенциально сильный фрагмент")[:320]
                normalized["text"] = normalized.get("text") or self._clip_text(transcript, start_sec, end_sec)
                normalized["score"] = self._score_value(normalized)
                if "total_score" not in normalized:
                    normalized["total_score"] = normalized["score"]
                valid_highlights.append(normalized)
            else:
                logger.warning(
                    "Dropped highlight '%s' because duration %.1fs is outside %s-%ss.",
                    clip.get("title"),
                    duration,
                    min_clip_len,
                    max_clip_len,
                )
        return valid_highlights

    def _remove_overlaps(self, highlights: list[dict], max_overlap_ratio: float = 0.45) -> list[dict]:
        selected: list[dict] = []
        for clip in sorted(highlights, key=self._score_value, reverse=True):
            start = float(clip["start_time"])
            end = float(clip["end_time"])
            overlaps = False
            for chosen in selected:
                c_start = float(chosen["start_time"])
                c_end = float(chosen["end_time"])
                overlap = max(0.0, min(end, c_end) - max(start, c_start))
                shorter = max(1.0, min(end - start, c_end - c_start))
                if overlap / shorter > max_overlap_ratio:
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
            "мама", "папа", "история", "смешно", "страшно", "конфликт", "вопрос", "ответ",
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
        for _, idx in sorted(scored, reverse=True)[: max(num_clips * 4, 10)]:
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
            text = self._clip_text(transcript, transcript[start_idx]["start"], transcript[end_idx]["end"])
            candidates.append({
                "start_time": transcript[start_idx]["start"],
                "end_time": transcript[end_idx]["end"],
                "title": (text[:55] + "…") if len(text) > 55 else text or "Клип",
                "hook": text.split(".")[0][:120] if text else "",
                "reason": "Локальный fallback выбрал плотный фрагмент с потенциальным хуком.",
                "hook_score": 35,
                "standalone_score": 35,
                "payoff_score": 30,
                "retention_score": 35,
                "clarity_score": 40,
                "total_score": 35,
                "score": 35,
                "text": text,
            })
        return self._validate_and_fix_highlights(candidates, transcript)

    def _candidate_prompt(self, candidates_per_chunk: int, min_clip_len: int | float, max_clip_len: int | float) -> str:
        return f"""Ты — профессиональный редактор коротких видео для TikTok, Reels и YouTube Shorts.

Твоя задача — найти в транскрипте фрагменты, которые могут работать как самостоятельные короткие видео.

НЕ выбирай просто информативные места. Выбирай только фрагменты, где есть:
1. Хук в первые 3–5 секунд.
2. Самостоятельный смысл без просмотра полного видео.
3. Эмоция, конфликт, удивление, польза, история или сильное мнение.
4. Понятный payoff в конце: вывод, панчлайн, инсайт, развязка.
5. Минимум воды, приветствий, вступлений и технических пояснений.

Плохой клип:
- начинается с контекста, который непонятен зрителю;
- содержит только середину мысли;
- требует знать, что было до этого;
- не имеет сильной первой фразы;
- является просто пересказом без эмоции или пользы.

Хороший клип:
- можно понять отдельно;
- первая фраза цепляет;
- в нём есть напряжение, вопрос, ошибка, секрет, конфликт, история или конкретная польза;
- зритель может досмотреть его до конца.

Оцени каждый кандидат по шкале 1–100:
- hook_score: насколько сильны первые 3–5 секунд;
- standalone_score: понятен ли клип без контекста;
- payoff_score: есть ли сильное завершение;
- retention_score: будет ли зритель досматривать;
- clarity_score: нет ли воды и обрывов мысли.

Верни JSON-массив без markdown. Каждый объект:
{{
  "start_time": float,
  "end_time": float,
  "title": "короткий заголовок",
  "hook": "первая цепляющая фраза",
  "reason": "почему это может сработать",
  "hook_score": 1-100,
  "standalone_score": 1-100,
  "payoff_score": 1-100,
  "retention_score": 1-100,
  "clarity_score": 1-100,
  "total_score": 1-100
}}

Правила:
- Длина клипа: {min_clip_len}–{max_clip_len} секунд.
- Лучше вернуть меньше клипов, чем плохие.
- Не выбирай несколько фрагментов подряд про одно и то же.
- Верни до {candidates_per_chunk} кандидатов, отсортированных по total_score.
- Таймкоды должны быть в секундах от начала исходного видео."""

    def find_highlight_candidates(self, transcript: list[dict], num_candidates: int = 12) -> list[dict]:
        num_candidates = max(1, int(num_candidates))
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        chunks = self.chunk_transcript(transcript, chunk_duration_sec=900)
        logger.info(
            "Transcript split into %s chunks. Looking for up to %s candidates with duration %s-%ss.",
            len(chunks),
            num_candidates,
            min_clip_len,
            max_clip_len,
        )

        candidates_per_chunk = max(num_candidates, 8)
        system_prompt = self._candidate_prompt(candidates_per_chunk, min_clip_len, max_clip_len)

        all_candidates: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing highlight candidate chunk %s/%s...", i + 1, len(chunks))
            formatted_text = self.format_transcript(chunk)
            try:
                clips = self._get_highlights_from_llm(system_prompt, formatted_text)
                clips = self._validate_and_fix_highlights(clips, transcript)
                logger.info("Chunk %s produced %s valid candidates.", i + 1, len(clips))
                all_candidates.extend(clips)
            except Exception as e:
                logger.error("Skipping LLM candidates for chunk %s due to error: %s", i + 1, e)

        if len(all_candidates) < max(3, min(num_candidates, 6)):
            logger.warning(
                "LLM returned too few valid candidates (%s). Adding local fallback candidates.",
                len(all_candidates),
            )
            all_candidates.extend(self._fallback_highlights(transcript, num_candidates))

        ranked = self._remove_overlaps(all_candidates, max_overlap_ratio=0.55)
        if len(ranked) < num_candidates:
            ranked = self._remove_overlaps(all_candidates, max_overlap_ratio=0.8)

        if len(ranked) < num_candidates:
            ranked = sorted(all_candidates, key=self._score_value, reverse=True)

        deduped = []
        seen = set()
        for clip in sorted(ranked, key=self._score_value, reverse=True):
            key = (round(float(clip["start_time"])), round(float(clip["end_time"])))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clip)

        final = deduped[:num_candidates]
        logger.info("Prepared %s highlight candidates for preview.", len(final))
        return final

    def find_highlights(self, transcript: list[dict], num_clips: int = None) -> list[dict]:
        if num_clips is None:
            num_clips = settings.NUM_CLIPS
        num_clips = max(1, int(num_clips))
        candidates = self.find_highlight_candidates(transcript, num_candidates=max(num_clips * 4, 10))
        selected = sorted(candidates, key=self._score_value, reverse=True)[:num_clips]
        selected = sorted(selected, key=lambda x: x["start_time"])
        logger.info("Selected %s final highlights for %s requested clips.", len(selected), num_clips)
        return selected

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
                adjusted_clip["duration"] = round(duration, 1)
                adjusted_clip["text"] = self._clip_text(transcript, new_start_sec, new_end_sec)
            adjusted_highlights.append(adjusted_clip)
        return adjusted_highlights
