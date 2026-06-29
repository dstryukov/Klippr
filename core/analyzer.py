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
        elif self.provider == "gemini":
            gemini_api_key = settings.GEMINI_API_KEY.get_secret_value() if settings.GEMINI_API_KEY else "dummy_gemini_key"
            self.client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_api_key)
        elif self.provider == "fireworks":
            fireworks_api_key = settings.FIREWORKS_API_KEY.get_secret_value() if settings.FIREWORKS_API_KEY else "dummy_fireworks_key"
            self.client = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=fireworks_api_key)
        else:
            groq_api_key = settings.GROQ_API_KEY.get_secret_value() if settings.GROQ_API_KEY else "dummy_groq_key"
            self.client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)

    def format_whisper_transcript(self, whisper_data: list[dict], chunk_duration: int = 10) -> str:
        if not whisper_data:
            return ""
        
        blocks = []
        current_text = []
        block_start = None
        
        for segment in whisper_data:
            text = segment.get("text", "").strip()
            if not text:
                continue
                
            if block_start is None:
                block_start = segment["start"]
                
            current_text.append(text)
            
            if segment["end"] - block_start >= chunk_duration:
                blocks.append(f"[{block_start:.2f} - {segment['end']:.2f}] {' '.join(current_text)}")
                current_text = []
                block_start = None
                
        if current_text and block_start is not None:
            last_end = whisper_data[-1]["end"]
            blocks.append(f"[{block_start:.2f} - {last_end:.2f}] {' '.join(current_text)}")
            
        return "\n".join(blocks)


    def _extract_json_array(self, text: str) -> list[dict]:
        import re
        import json
        
        # Strip <think>...</think> tags if present
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # Try to find a JSON block in markdown (take the last one, as reasoning might have code blocks)
        matches = re.findall(r"```(?:json)?(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if matches:
            text_to_parse = matches[-1].strip()
        else:
            text_to_parse = text.strip()
            
        # Search for {"highlights":
        match = re.search(r'\{\s*"highlights"\s*:', text_to_parse)
        if match:
            start_idx = match.start()
            end_idx = text_to_parse.rfind("}")
            if end_idx > start_idx:
                try:
                    json_str = text_to_parse[start_idx:end_idx + 1]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "highlights" in parsed:
                        return parsed["highlights"]
                except json.JSONDecodeError:
                    pass
                
        # Fallback to finding array directly
        start_idx_arr = text_to_parse.rfind("[")
        end_idx_arr = text_to_parse.rfind("]")
        if start_idx_arr != -1 and end_idx_arr != -1 and end_idx_arr > start_idx_arr:
            try:
                parsed_arr = json.loads(text_to_parse[start_idx_arr:end_idx_arr + 1])
                if isinstance(parsed_arr, list):
                    return parsed_arr
            except json.JSONDecodeError:
                pass
        # If we reach here, we failed
        import time
        debug_file = f"debug_llm_response_{int(time.time())}.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(text)
        logger.error(f"Failed to parse JSON. Full response saved to {debug_file}")
        raise ValueError("No valid JSON array or object found in the response")

    def _call_llm_with_retry(self, client: OpenAI, model: str, system_prompt: str, user_prompt: str, job: Any = None) -> str:
        import time
        max_attempts = 5
        for attempt in range(max_attempts):
            if job and getattr(job, "cancel_requested", False):
                raise InterruptedError("Analysis cancelled by user")
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    max_tokens=8192,
                )
                return response.choices[0].message.content.strip()
            except (openai.RateLimitError, openai.APIConnectionError) as e:
                if attempt == max_attempts - 1:
                    raise
                logger.info(f"LLM API rate limit or connection error: {e}. Retrying in 15s...")
                for _ in range(15):
                    if job and getattr(job, "cancel_requested", False):
                        raise InterruptedError("Analysis cancelled by user")
                    time.sleep(1)
        return ""

    def _get_highlights_from_llm(self, system_prompt: str, formatted_text: str, job: Any = None) -> list[dict]:
        for attempt in range(2):
            content = ""
            try:
                logger.info(f"Sending to {self.provider} (Model: {self.model}, Attempt {attempt + 1})...")
                content = self._call_llm_with_retry(self.client, self.model, system_prompt, formatted_text, job)
                logger.debug("LLM response (first 500 chars): %s", content[:500])
                parsed_clips = self._extract_json_array(content)
                
                for clip in parsed_clips:
                    clip["title"] = clip.get("topic_title", clip.get("title", "Highlight"))
                    clip["hook"] = clip.get("hook_text", "")
                    clip["exact_text"] = clip.get("exact_text", "").strip()
                    
                    try:
                        hook_str = int(clip.get("hook_strength", 0))
                    except (ValueError, TypeError):
                        hook_str = 0
                    try:
                        clarity = int(clip.get("standalone_clarity", 0))
                    except (ValueError, TypeError):
                        clarity = 0
                    try:
                        impact = int(clip.get("emotional_impact", 0))
                    except (ValueError, TypeError):
                        impact = 0
                    clip["total_score"] = hook_str + clarity + impact
                        
                parsed_clips.sort(key=lambda x: x.get("total_score", 0), reverse=True)
                
                logger.info(f"Successfully extracted {len(parsed_clips)} highlight candidates.")
                return parsed_clips
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}\nContent was (first 1000 chars): {content[:1000]}")
                if attempt == 1:
                    raise ValueError(f"LLM returned invalid JSON: {e}")
                
                # Sleep before retrying to avoid hammering rate limits on parsing failures
                for _ in range(10):
                    if job and getattr(job, "cancel_requested", False):
                        raise InterruptedError("Analysis cancelled by user")
                    import time
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error calling/parsing LLM response: {e}\nContent was: {content[:1000]}")
                if attempt == 1:
                    raise
                for _ in range(10):
                    if job and getattr(job, "cancel_requested", False):
                        raise InterruptedError("Analysis cancelled by user")
                    import time
                    time.sleep(1)
        return []

    def chunk_transcript(self, transcript: list[dict], chunk_duration_sec: float = 900, overlap_sec: float = 60) -> list[list[dict]]:
        chunks = []
        current_chunk = []
        current_start = 0.0

        for segment in transcript:
            if not current_chunk:
                current_start = segment["start"]

            current_chunk.append(segment)

            # Если превысили лимит чанка
            if segment["end"] - current_start >= chunk_duration_sec:
                chunks.append(current_chunk)

                # Создаем новый чанк, забирая последние overlap_sec секунд из текущего
                overlap_start_time = segment["end"] - overlap_sec
                current_chunk = [seg for seg in current_chunk if seg["end"] > overlap_start_time]

                # Если после фильтрации чанк пуст (например, один гигантский сегмент), берем хотя бы последний
                if not current_chunk:
                    current_chunk = [segment]
                current_start = current_chunk[0]["start"]

        if current_chunk and (not chunks or chunks[-1] != current_chunk):
            chunks.append(current_chunk)
        return chunks

    def _clip_text(self, transcript: list[dict], start_sec: float, end_sec: float, max_chars: int = 2500) -> str:
        parts = []
        for seg in transcript:
            if seg["start"] < end_sec and seg["end"] > start_sec:
                text = seg.get("text", "").strip()
                if text:
                    parts.append(text)
        text = " ".join(parts)
        return (text[:max_chars] + "…") if len(text) > max_chars else text

    def _validate_exact_text(self, clip: dict, transcript: list[dict]) -> dict:
        """Validate LLM's exact_text against actual transcript.

        Adjusts clip boundaries if the quoted text is found at a different position,
        or applies a score penalty if the quote doesn't match at all.
        """
        exact_text = clip.get("exact_text", "").strip()
        if not exact_text:
            return clip

        start_sec = float(clip["start_time"])
        end_sec = float(clip["end_time"])

        actual_text = self._clip_text(transcript, start_sec, end_sec, max_chars=2000)
        if not actual_text:
            return clip

        actual_words = actual_text.lower().split()
        quote_words = exact_text.lower().split()

        if len(quote_words) < 3:
            return clip

        def _find_best_match(needle_words, haystack_words, search_start=0, search_range=None):
            if search_range is None:
                search_range = len(haystack_words)
            search_end = min(search_start + search_range, len(haystack_words))
            best_pos = -1
            best_ratio = 0.0
            n = len(needle_words)
            for i in range(search_start, max(search_start + 1, search_end - n + 1)):
                window = haystack_words[i:i + n]
                if not window:
                    continue
                matches = sum(1 for a, b in zip(needle_words, window) if a == b)
                ratio = matches / n
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i
            return best_pos, best_ratio

        head_words = quote_words[:min(15, len(quote_words) // 2)]
        tail_words = quote_words[-min(15, len(quote_words) // 2):]

        head_pos, head_ratio = _find_best_match(head_words, actual_words, search_start=0, search_range=min(80, len(actual_words)))
        tail_pos, tail_ratio = _find_best_match(tail_words, actual_words, search_start=max(0, len(actual_words) - 80))

        avg_ratio = (head_ratio + tail_ratio) / 2.0
        adjusted = dict(clip)

        # Adjust start boundary if head found at different position
        if head_ratio > 0.6 and head_pos >= 0:
            head_text = " ".join(actual_words[max(0, head_pos):head_pos + len(head_words)])
            for seg in transcript:
                if head_text[:20] in seg.get("text", "").lower() and seg["start"] >= start_sec - 10:
                    if abs(seg["start"] - start_sec) > 1.0:
                        logger.info(
                            "exact_text validation: adjusting start %.1f → %.1f for '%s'",
                            start_sec, seg["start"], clip.get("title"),
                        )
                        adjusted["start_time"] = round(seg["start"], 2)
                    break

        # Adjust end boundary if tail found at different position
        if tail_ratio > 0.6 and tail_pos >= 0:
            tail_text = " ".join(actual_words[tail_pos:tail_pos + len(tail_words)])
            for seg in reversed(transcript):
                if tail_text[-20:] in seg.get("text", "").lower() and seg["end"] <= end_sec + 10:
                    if abs(seg["end"] - end_sec) > 1.0:
                        logger.info(
                            "exact_text validation: adjusting end %.1f → %.1f for '%s'",
                            end_sec, seg["end"], clip.get("title"),
                        )
                        adjusted["end_time"] = round(seg["end"], 2)
                        adjusted["duration"] = round(float(adjusted["end_time"]) - float(adjusted["start_time"]), 1)
                    break

        # Score penalty for complete mismatch
        if avg_ratio < 0.4:
            penalty = 0.85
            old_score = adjusted.get("total_score", 50)
            adjusted["total_score"] = round(old_score * penalty, 1)
            adjusted["score"] = round(adjusted.get("score", old_score) * penalty, 1)
            adjusted["exact_text_match"] = round(avg_ratio, 2)
            logger.warning(
                "exact_text mismatch for '%s' (match=%.0f%%). Score penalized: %.1f → %.1f",
                clip.get("title"), avg_ratio * 100, old_score, adjusted["total_score"],
            )
        else:
            adjusted["exact_text_match"] = round(avg_ratio, 2)
            if avg_ratio < 0.7:
                logger.info(
                    "exact_text partial match for '%s' (match=%.0f%%). Boundaries may have been adjusted.",
                    clip.get("title"), avg_ratio * 100,
                )

        return adjusted

    def _score_value(self, clip: dict) -> float:
        for key in ("total_score", "score", "retention_score", "hook_score"):
            try:
                return float(clip.get(key))
            except (TypeError, ValueError):
                pass
        return 50.0

    def _get_speaker_at(self, diarization: list[dict], time_sec: float) -> str | None:
        """Return speaker label at a given time, or None if no diarization."""
        for seg in diarization:
            if seg["start"] <= time_sec <= seg["end"]:
                return seg["speaker"]
        return None

    def _count_speakers_in_range(self, diarization: list[dict], start_sec: float, end_sec: float) -> int:
        """Count unique speakers in a time range."""
        speakers = set()
        for seg in diarization:
            if seg["end"] > start_sec and seg["start"] < end_sec:
                speakers.add(seg["speaker"])
        return len(speakers)

    def _find_speaker_turn_boundary(self, diarization: list[dict], time_sec: float, direction: int, max_shift: float = 3.0) -> float | None:
        """Find nearest speaker turn boundary near a given time.

        direction: -1 to search backwards, +1 to search forwards.
        Returns the boundary time or None if no boundary found within max_shift.
        """
        if not diarization:
            return None
        best = None
        best_dist = max_shift
        for seg in diarization:
            boundary = seg["start"] if direction > 0 else seg["end"]
            dist = abs(boundary - time_sec)
            if dist < best_dist and dist > 0.3:  # at least 0.3s away to be meaningful
                best_dist = dist
                best = boundary
        return best

    def _find_thought_end(self, transcript: list[dict], seg_idx: int, direction: int, max_segments: int = 8) -> int:
        """Find the nearest segment boundary where a thought/sentence ends.

        Looks for sentence-ending punctuation (. ! ? ...) or a long pause (>0.5s gap).
        direction: +1 to search forward, -1 to search backward.
        Returns the adjusted segment index.
        """
        sentence_endings = {".", "!", "?", "…", "...", "…"}
        checked = 0
        idx = seg_idx
        while 0 <= idx < len(transcript) and checked < max_segments:
            text = transcript[idx].get("text", "").strip()
            # Check if text ends (or starts) with sentence-ending punctuation
            if direction > 0:
                if text and any(text.rstrip().endswith(p) for p in sentence_endings):
                    return idx
                # Check for long pause before next segment
                if idx + 1 < len(transcript):
                    gap = transcript[idx + 1]["start"] - transcript[idx]["end"]
                    if gap > 0.5:
                        return idx
                idx += 1
            else:
                if text and any(text.rstrip().endswith(p) for p in sentence_endings):
                    return idx
                if idx - 1 >= 0:
                    gap = transcript[idx]["start"] - transcript[idx - 1]["end"]
                    if gap > 0.5:
                        return idx
                idx -= 1
            checked += 1
        return seg_idx  # fallback: no clean thought boundary found

    def _validate_and_fix_highlights(self, highlights: list[dict], transcript: list[dict], diarization: list[dict] | None = None) -> list[dict]:
        if not transcript:
            return []

        max_duration = transcript[-1]["end"]
        valid_highlights = []
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        # Soft max: allow up to 30% over max to preserve complete thoughts
        soft_max_clip_len = int(max_clip_len * 1.3)

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
                logger.warning(
                    "Dropped highlight '%s' because start >= end after clamping. "
                    "Original: start=%.2f end=%.2f, max_duration=%.2f",
                    clip.get("title"), float(clip.get("start_time", 0)), float(clip.get("end_time", 0)), max_duration,
                )
                continue

            start_idx = min(range(len(transcript)), key=lambda i: abs(transcript[i]["start"] - start_sec))
            end_idx = min(range(len(transcript)), key=lambda i: abs(transcript[i]["end"] - end_sec))
            if end_idx < start_idx:
                end_idx = start_idx

            duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]

            # Expand if too short (up to min_clip_len * 1.5)
            max_expand_min = min_clip_len * 1.5
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
                if duration > max_expand_min:
                    break

            # Smart trim if too long: find thought boundary instead of hard cut
            overlong = False
            if duration > max_clip_len:
                # Try to find a natural thought end within soft_max window
                thought_end_idx = self._find_thought_end(transcript, end_idx, direction=-1, max_segments=6)
                # Only accept if it brings us closer to max_clip_len
                candidate_duration = transcript[thought_end_idx]["end"] - transcript[start_idx]["start"]
                if candidate_duration <= soft_max_clip_len:
                    end_idx = thought_end_idx
                    duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]
                else:
                    # Try forward from a slightly earlier point
                    earlier_idx = max(start_idx, end_idx - 3)
                    thought_end_idx = self._find_thought_end(transcript, earlier_idx, direction=-1, max_segments=4)
                    candidate_duration = transcript[thought_end_idx]["end"] - transcript[start_idx]["start"]
                    if min_clip_len <= candidate_duration <= soft_max_clip_len:
                        end_idx = thought_end_idx
                        duration = transcript[end_idx]["end"] - transcript[start_idx]["start"]
                    else:
                        # Accept as overlong — complete thought is more important than duration
                        overlong = True
                        logger.info(
                            "Highlight '%s' accepted as overlong (%.1fs > %ss) to preserve complete thought.",
                            clip.get("title"), duration, max_clip_len,
                        )

            start_sec = transcript[start_idx]["start"]
            end_sec = transcript[end_idx]["end"]
            duration = end_sec - start_sec

            # Accept clip if it's at least min_clip_len (or close) and not absurdly long
            if duration >= min_clip_len * 0.7 and duration <= soft_max_clip_len * 1.2:
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
                if overlong:
                    normalized["overlong"] = True

                # Validate LLM's exact_text against real transcript
                normalized = self._validate_exact_text(normalized, transcript)

                # Diarization-aware scoring: bonus for mono-speaker clips
                if diarization:
                    n_speakers = self._count_speakers_in_range(diarization, start_sec, end_sec)
                    normalized["speaker_count"] = n_speakers
                    if n_speakers == 1:
                        normalized["score"] = normalized["score"] * 1.08
                        normalized["total_score"] = normalized["total_score"] * 1.08
                    elif n_speakers >= 3:
                        normalized["score"] = normalized["score"] * 0.95
                        normalized["total_score"] = normalized["total_score"] * 0.95

                valid_highlights.append(normalized)
            else:
                logger.warning(
                    "Dropped highlight '%s' because duration %.1fs is outside acceptable range (%.0f-%.0fs).",
                    clip.get("title"),
                    duration,
                    min_clip_len * 0.7,
                    soft_max_clip_len * 1.2,
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

    def _candidate_prompt(self, candidates_per_chunk: int, min_clip_len: int | float, max_clip_len: int | float, soft_max_clip_len: int | float | None = None) -> str:
        if soft_max_clip_len is None:
            soft_max_clip_len = int(max_clip_len * 1.3)
        return f"""Ты — профессиональный редактор коротких видео для TikTok, Reels и YouTube Shorts.

Транскрипт разбит на блоки с таймкодами: [начало - конец] текст...
Твоя задача — выбрать диапазоны предложений, которые станут самостоятельными Shorts/Reels.

Выбирай только фрагменты, где есть один из типов контента:
- История (рассказ, пример из жизни, эксперимент)
- Факт (научный факт, статистика, исследование)
- Мнение (сильное, спорное, эмоциональное)
- Юмор (смешной момент, ирония, неожиданность)

Критерии качества:
1. Хук в первые 3–5 секунд (первое предложение должно цеплять).
2. Самостоятельный смысл: зрителю не нужен контекст всего видео.
3. Понятный финал: вывод, ответ на вопрос, панчлайн, инсайт, развязка.
4. Минимум воды.

🔴 ПРАВИЛО ЗАВЕРШЁННОЙ МЫСЛИ (КРИТИЧНО):
- Последнее предложение клипа обязано быть выводом, ответом на вопрос, панчлайном или развязкой.
- ЗАПРЕЩЕНО заканчивать на: переходе к новой теме, незаконченном рассуждении, «и вот...», «далее...», вопросе без ответа.
- ИГНОРИРУЙ ЛИМИТ ВРЕМЕНИ, если мысль не закончена. Лучше сделать клип на 120 секунд, чем оборвать мысль на полуслове!
- Проверь: если убрать последнее предложение — мысль остаётся незаконченной? Если да — добавь ещё предложения до логического конца.

🔴 ОЧИСТКА ОТ ВОДЫ:
- ПРОПУСКАЙ предложения-связки, дисклеймеры, призывы к действию: «поделитесь в комментариях», «ссылка в описании», «заходите на канал».
- ПРОПУСКАЙ повторения одного и того же разными словами.
- ПРОПУСКАЙ междометия и заполнители: «ну вот», «как бы», «то есть», «простой пример».
- ПРОПУСКАЙ отступления от главной темы клипа.
- НЕ включай предложения, которые не добавляют смысла к основной мысли клипа.

🔴 ЗАПРЕЩЕНО начинать клип со связок: «Но», «И», «А», «Поэтому», «Также», «Кстати», «Ну»
🔴 ЗАПРЕЩЕНО начинать с висячих местоимений: «Он», «Она», «Это», если непонятно к чему
- Первое предложение должно быть сильным: вопрос, утверждение, начало истории — не вводное слово.

🔴 РАЗНООБРАЗИЕ:
- Не выбирай несколько фрагментов об одной и той же подтеме.
- Выбирай разные АСПЕКТЫ темы (разные примеры, разные выводы, разные истории).

Верни СТРОГО И ТОЛЬКО JSON-объект с ключом "highlights", содержащим массив. 
КРИТИЧЕСКИ ВАЖНО: Запрещено выводить любые размышления, пояснения или теги <think>. Твой ответ должен начинаться с символа {{ и заканчиваться символом }}.
Каждый объект в массиве:
{{
  "highlights": [
    {{
      "topic_title": "короткий заголовок",
      "hook_text": "точная цитата первого предложения (хук)",
      "payoff_text": "точная цитата финального предложения (развязка)",
      "hook_strength": 1-10,
      "standalone_clarity": 1-10,
      "emotional_impact": 1-10,
      "start_time": <число с плавающей точкой, из квадратных скобок начала>,
      "end_time": <число с плавающей точкой, из квадратных скобок конца>
    }}
  ]
}}

Правила:
- Целевая длина: {min_clip_len}–{max_clip_len} секунд.
- Если мысль не помещается — можно до {soft_max_clip_len} секунд. Главное — мысль завершена.
- Верни до {candidates_per_chunk} лучших фрагментов.
- Если хороших нет — верни пустой массив [] в "highlights".
- Возвращай ТОЛЬКО JSON-объект, без markdown и без пояснений."""

    def find_highlight_candidates(self, transcript: list[dict], num_candidates: int = 12, job: Any = None) -> list[dict]:
        num_candidates = max(1, int(num_candidates))
        min_clip_len = settings.MIN_CLIP_DURATION
        max_clip_len = settings.MAX_CLIP_DURATION
        chunks = self.chunk_transcript(transcript, chunk_duration_sec=120, overlap_sec=30)
        logger.info(
            "Transcript split into %s chunks. Looking for up to %s candidates with duration %s-%ss.",
            len(chunks),
            num_candidates,
            min_clip_len,
            max_clip_len,
        )

        candidates_per_chunk = max(num_candidates, 8)
        soft_max_clip_len = int(max_clip_len * 1.3)
        system_prompt = self._candidate_prompt(candidates_per_chunk, min_clip_len, max_clip_len, soft_max_clip_len)

        all_candidates: list[dict[str, Any]] = []
        chunk_boundaries: list[tuple[float, float]] = []
        for i, chunk in enumerate(chunks):
            if job and getattr(job, "cancel_requested", False):
                raise InterruptedError("Analysis cancelled by user")
            logger.info("Processing highlight candidate chunk %s/%s...", i + 1, len(chunks))
            formatted_text = self.format_whisper_transcript(chunk)
            try:
                clips = self._get_highlights_from_llm(system_prompt, formatted_text, job)
                # Map is direct since we asked for start_time and end_time
                for clip in clips:
                    try:
                        clip["start_time"] = float(clip.get("start_time", 0))
                        clip["end_time"] = float(clip.get("end_time", 0))
                    except (ValueError, TypeError):
                        clip["start_time"] = 0.0
                        clip["end_time"] = 0.0
                    logger.info("  Clip '%s': %.1f-%.1fs",
                                clip.get("topic_title", "?"),
                                clip["start_time"], clip["end_time"])
                clips = self._validate_and_fix_highlights(clips, transcript)
                chunk_start = chunk[0]["start"] if chunk else 0
                chunk_end = chunk[-1]["end"] if chunk else 0
                chunk_boundaries.append((chunk_start, chunk_end))
                for c in clips:
                    c["_chunk_index"] = i
                logger.info("Chunk %s produced %s valid candidates.", i + 1, len(clips))
                all_candidates.extend(clips)
            except Exception as e:
                logger.error("Skipping LLM candidates for chunk %s due to error: %s", i + 1, e)

        if not all_candidates:
            logger.error("LLM returned 0 valid candidates. Aborting — no fallback.")
            raise RuntimeError(
                "LLM не вернул ни одного валидного кандидата. "
                "Проверьте: 1) доступность API (GROQ_API_KEY), 2) имя модели в config.yaml, "
                "3) логи ошибок JSON parse выше. Fallback отключён."
            )
        if len(all_candidates) < 3:
            logger.warning(
                "LLM returned only %s valid candidates (expected %s). Proceeding without fallback.",
                len(all_candidates), num_candidates,
            )

        # Remove candidates duplicated in overlap zones
        all_candidates = self._deduplicate_overlap_candidates(all_candidates, chunk_boundaries)

        # Remove candidates with highly similar text content
        all_candidates = self._deduplicate_by_content(all_candidates)

        ranked = self._remove_overlaps(all_candidates, max_overlap_ratio=0.45)
        if len(ranked) < num_candidates:
            ranked = self._remove_overlaps(all_candidates, max_overlap_ratio=0.65)

        if len(ranked) < num_candidates:
            ranked = sorted(all_candidates, key=self._score_value, reverse=True)

        # Penalize temporal clustering to promote diversity
        ranked = self._enforce_temporal_diversity(ranked, min_clip_len)

        deduped = []
        seen = set()
        for clip in sorted(ranked, key=self._score_value, reverse=True):
            # Quantize to 5s buckets to catch near-duplicates
            key = (round(float(clip["start_time"]) / 5) * 5, round(float(clip["end_time"]) / 5) * 5)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clip)

        final = deduped[:num_candidates]
        logger.info("Prepared %s highlight candidates for preview.", len(final))
        return final

    def _deduplicate_overlap_candidates(
        self, candidates: list[dict], chunk_boundaries: list[tuple[float, float]]
    ) -> list[dict]:
        """Remove candidates from overlap zones that duplicate candidates from the primary chunk."""
        if len(chunk_boundaries) < 2:
            return candidates

        overlap_zones: list[tuple[float, float]] = []
        for i in range(1, len(chunk_boundaries)):
            prev_end = chunk_boundaries[i - 1][1]
            curr_start = chunk_boundaries[i][0]
            if curr_start < prev_end:
                overlap_zones.append((curr_start, prev_end))

        if not overlap_zones:
            return candidates

        filtered = []
        for clip in candidates:
            start = float(clip.get("start_time", 0))
            end = float(clip.get("end_time", 0))
            in_overlap = any(oz_s <= start and end <= oz_e for oz_s, oz_e in overlap_zones)
            chunk_idx = clip.get("_chunk_index", 0)

            if in_overlap and chunk_idx > 0:
                is_dup = False
                for other in candidates:
                    if other is clip:
                        continue
                    o_start = float(other.get("start_time", 0))
                    o_end = float(other.get("end_time", 0))
                    overlap = max(0, min(end, o_end) - max(start, o_start))
                    shorter = max(1.0, min(end - start, o_end - o_start))
                    if overlap / shorter > 0.5 and other.get("_chunk_index", 0) < chunk_idx:
                        is_dup = True
                        break
                if is_dup:
                    logger.info(
                        "Dropping overlap duplicate '%s' (%.1f-%.1f) from chunk %s",
                        clip.get("title"), start, end, chunk_idx,
                    )
                    continue
            filtered.append(clip)

        for c in filtered:
            c.pop("_chunk_index", None)
        return filtered

    def _deduplicate_by_content(self, candidates: list[dict]) -> list[dict]:
        """Remove candidates with highly similar text content (word overlap > 60%)."""
        if len(candidates) < 2:
            return candidates

        def _word_set(text: str) -> set:
            return set(text.lower().split())

        kept = []
        for clip in sorted(candidates, key=self._score_value, reverse=True):
            clip_words = _word_set(clip.get("text", ""))
            is_dup = False
            for kept_clip in kept:
                kept_words = _word_set(kept_clip.get("text", ""))
                if not clip_words or not kept_words:
                    continue
                overlap = len(clip_words & kept_words) / max(len(clip_words), len(kept_words))
                if overlap > 0.6:
                    logger.info(
                        "Dropping content duplicate '%s' (overlap=%.0f%% with '%s')",
                        clip.get("title"), overlap * 100, kept_clip.get("title"),
                    )
                    is_dup = True
                    break
            if not is_dup:
                kept.append(clip)
        return kept

    def _enforce_temporal_diversity(self, candidates: list[dict], min_clip_len: int | float) -> list[dict]:
        """Penalize candidates that cluster in the same time region to promote diversity."""
        if len(candidates) < 3:
            return candidates

        diversity_window = min_clip_len * 3
        scored = []
        for i, clip in enumerate(candidates):
            start = float(clip.get("start_time", 0))
            neighbors = sum(
                1 for j, other in enumerate(candidates)
                if i != j and abs(start - float(other.get("start_time", 0))) < diversity_window
            )
            adjusted = dict(clip)
            if neighbors >= 3:
                adjusted["score"] = round(adjusted.get("score", 50) * 0.85, 1)
                adjusted["total_score"] = round(adjusted.get("total_score", 50) * 0.85, 1)
                adjusted["diversity_penalty"] = 0.85
            elif neighbors >= 2:
                adjusted["score"] = round(adjusted.get("score", 50) * 0.92, 1)
                adjusted["total_score"] = round(adjusted.get("total_score", 50) * 0.92, 1)
                adjusted["diversity_penalty"] = 0.92
            scored.append(adjusted)

        return sorted(scored, key=self._score_value, reverse=True)

    def find_highlights(self, transcript: list[dict], num_clips: int = None) -> list[dict]:
        if num_clips is None:
            num_clips = settings.NUM_CLIPS
        num_clips = max(1, int(num_clips))
        candidates = self.find_highlight_candidates(transcript, num_candidates=max(num_clips * 4, 10))
        selected = sorted(candidates, key=self._score_value, reverse=True)[:num_clips]
        selected = sorted(selected, key=lambda x: x["start_time"])
        logger.info("Selected %s final highlights for %s requested clips.", len(selected), num_clips)
        return selected

    def snap_to_silence(self, highlights: list[dict], audio_path: str, transcript: list[dict], diarization: list[dict] | None = None) -> list[dict]:
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

            # Widen silence search window if diarization is available
            window_ms = 3000 if diarization else 2000
            new_start_sec = find_nearest_silence(start_ms, window_ms=window_ms) / 1000.0
            new_end_sec = find_nearest_silence(end_ms, window_ms=window_ms) / 1000.0

            # Snap to transcript segment boundaries
            for seg in transcript:
                seg_start = seg["start"]
                seg_end = seg["end"]
                if seg_start + 0.1 < new_start_sec < seg_end - 0.1:
                    new_start_sec = seg_start
                if seg_start + 0.1 < new_end_sec < seg_end - 0.1:
                    new_end_sec = seg_end

            # Diarization-aware boundary adjustment:
            # Avoid cutting in the middle of a speaker's utterance
            if diarization:
                start_speaker = self._get_speaker_at(diarization, new_start_sec)
                end_speaker = self._get_speaker_at(diarization, new_end_sec)

                # If start falls mid-utterance, snap to the beginning of that utterance
                if start_speaker:
                    for dseg in diarization:
                        if dseg["speaker"] == start_speaker and dseg["start"] < new_start_sec < dseg["end"]:
                            if new_start_sec - dseg["start"] < 2.0:  # only if close to boundary
                                new_start_sec = dseg["start"]
                            break

                # If end falls mid-utterance, snap to the end of that utterance
                if end_speaker:
                    for dseg in diarization:
                        if dseg["speaker"] == end_speaker and dseg["start"] < new_end_sec < dseg["end"]:
                            if dseg["end"] - new_end_sec < 2.0:
                                new_end_sec = dseg["end"]
                            break

            duration = new_end_sec - new_start_sec
            if min_clip_len - 0.5 <= duration <= max_clip_len + 1.5:
                adjusted_clip["start_time"] = round(new_start_sec, 2)
                adjusted_clip["end_time"] = round(new_end_sec, 2)
                adjusted_clip["duration"] = round(duration, 1)
                adjusted_clip["text"] = self._clip_text(transcript, new_start_sec, new_end_sec)
            elif duration < min_clip_len * 0.7:
                logger.warning(
                    "Dropping clip '%s' after silence snap: duration %.1fs < %.1fs minimum.",
                    adjusted_clip.get("title"), duration, min_clip_len * 0.7,
                )
                continue
            adjusted_highlights.append(adjusted_clip)
        return adjusted_highlights

    def generate_hooks(self, candidates: list[dict], transcript: list[dict]) -> list[dict]:
        """Generate catchy hook text for each candidate using LLM.

        Sets candidate["hook_text"] for each candidate. Falls back to existing
        hook field or first 50 chars of text if LLM fails.
        """
        if not candidates:
            return candidates

        # Build batch prompt with all candidates
        items = []
        for i, c in enumerate(candidates):
            text = c.get("text") or self._clip_text(transcript, float(c.get("start_time", 0)), float(c.get("end_time", 0)))
            items.append(f"[{i + 1}] {text[:300]}")

        system_prompt = (
            "Ты — гениальный маркетолог и сценарист вирусных коротких видео (Reels/Shorts/TikTok).\n"
            "Твоя задача — создать для каждого фрагмента ОДИН невероятно цепляющий hook (до 60 символов).\n"
            "Хук — это первая фраза на экране, которая заставляет мозг зрителя остановиться и смотреть дальше.\n\n"
            "Техники хука (выбирай самую жёсткую интригу):\n"
            "- Провокация/Разрыв шаблона: «Почему богатые скучают?», «Школа делает из нас рабов»\n"
            "- Шок-факт: «Мыши умирали ради удовольствия»\n"
            "- Контринтуитивное: «Алкоголь работает и без алкоголя»\n"
            "- Жесткий секрет: «Этот легальный наркотик нас убивает»\n"
            "- Болевая точка: «Причина, почему у вас нет энергии»\n\n"
            "СТРОГИЕ ПРАВИЛА (ЗАПРЕТЫ):\n"
            "- ЗАПРЕЩЕНЫ энциклопедические заголовки (например: «Л-допа: источник дофамина», «Эксперимент с мышами»).\n"
            "- ЗАПРЕЩЕНЫ скучные обобщения: «Дофамин и мотивация», «Влияние на жизнь».\n"
            "- Максимум 60 символов. Очень коротко и резко.\n"
            "- НЕ цитируй дословно, если в тексте нет шокирующей фразы. ПРИДУМАЙ её по смыслу.\n"
            "- Пиши на том же языке, что и текст.\n"
            "- Без кавычек вокруг хука.\n\n"
            "Зритель должен испытать непреодолимое желание узнать ответ или развязку!\n\n"
            "Верни JSON-массив без markdown. Каждый элемент — строка с hook для соответствующего фрагмента."
        )
        user_prompt = "Фрагменты:\n" + "\n".join(items)

        try:
            content = self._call_llm_with_retry(self.client, self.model, system_prompt, user_prompt)
            # Try to find a JSON block in markdown
            matches = re.findall(r"```(?:json)?(.*?)```", content, re.DOTALL | re.IGNORECASE)
            if matches:
                text_to_parse = matches[-1].strip()
            else:
                text_to_parse = content.strip()
                
            start_idx = text_to_parse.find("[")
            end_idx = text_to_parse.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    hooks = json.loads(text_to_parse[start_idx:end_idx + 1])
                    if isinstance(hooks, list) and len(hooks) >= len(candidates):
                        for i, c in enumerate(candidates):
                            hook = str(hooks[i]).strip()[:60]
                            if hook:
                                c["hook_text"] = hook
                        logger.info("Generated %d hooks via LLM.", len(candidates))
                        return candidates
                except json.JSONDecodeError:
                    pass
                    
            # If we got a plain text response with one hook per line
            lines = [l.strip() for l in content.strip().split("\n") if l.strip() and len(l.strip()) > 5]
            if len(lines) >= len(candidates):
                for i, c in enumerate(candidates):
                    # Remove markdown list markers if present
                    clean_line = re.sub(r"^[-*0-9.]+\s*", "", lines[i])
                    c["hook_text"] = clean_line[:60]
                logger.info("Generated %d hooks via LLM (line format).", len(candidates))
                return candidates
            else:
                logger.warning(f"Hook generation failed to parse. Content: {content[:500]}")
        except Exception as e:
            logger.warning("Hook generation failed (%s). Using fallback hooks.", e)

        # Fallback: use existing hook field or first 50 chars of text
        for c in candidates:
            fallback = c.get("hook") or c.get("text", "")[:50]
            c["hook_text"] = str(fallback)[:60] if fallback else ""
        return candidates
