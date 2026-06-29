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


CONTENT_TYPE_PROMPT = """Analyze this video transcript sample and classify the content type.
Choose one: podcast, interview, tutorial, lecture, commentary, debate, vlog, other.
Also estimate content density: low (mostly filler/chit-chat), medium, or high (dense info/stories).
Respond with JSON only: {"content_type": "...", "density": "..."}"""

VIRALITY_CRITERIA = """
Virality signals to prioritize (ranked by impact):
1. HOOK MOMENTS — statements that create immediate curiosity ("The secret is...", "Nobody talks about...", "I was completely wrong about...")
2. EMOTIONAL PEAKS — genuine surprise, laughter, anger, vulnerability, excitement; raw unscripted reactions
3. OPINION BOMBS — strong, polarizing or counter-intuitive statements that trigger agree/disagree
4. REVELATION MOMENTS — surprising facts, stats, or confessions that reframe how the viewer thinks
5. CONFLICT/TENSION — disagreement, pushback, or a problem being confronted head-on
6. QUOTABLE ONE-LINERS — a sentence that works as a standalone quote card
7. STORY PEAKS — the climax or twist of an anecdote; the payoff moment
8. PRACTICAL VALUE — a concrete tip, hack, or insight the viewer can immediately apply
"""

HIGHLIGHT_SYSTEM_PROMPT = """You are an elite short-form video editor who has studied thousands of viral clips on TikTok, Instagram Reels, and YouTube Shorts. You know exactly what makes viewers stop scrolling, watch to the end, and share.

{virality_criteria}

Content type: {content_type} | Density: {density}

Your task: identify the most viral-worthy highlights from the transcript.

Rules:
- Every highlight must open with a strong HOOK — a line that grabs attention within the first 3 seconds
- Duration sweet spot: {min_clip_len}-{max_clip_len} seconds. Ensure the clip has enough context to be valuable. Go closer to {max_clip_len}s if a story arc or explanation needs full context to land.
- Never cut mid-sentence or mid-thought — each clip must feel complete and self-contained
- Clips must not overlap significantly with each other
- Score 0-100 on viral potential (not general quality)
- Extract as many viral highlights as you organically find in the transcript (do not force clips, only extract truly viral moments)
- For each highlight, generate a "hook_text" (Text Overlay) for the first 2 seconds of the clip.
  - "AUDIO-VISUAL INTERLOCK" MECHANIC: The hook_text MUST create a psychological "open loop" (posing a sharp question, hitting a pain point, or highlighting a stark contrast). The VERY FIRST spoken sentence in the audio clip MUST serve as the immediate answer, punchline, or logical continuation of this on-screen text.
  - RULES for hook_text: 1) Brevity: Under 10 words. 2) Triggers: Inject high conflict, paradox, pain, FOMO, or stark contrast. 3) No cheap clickbait: keep it intellectually honest.
- Explain in one sentence why this clip is viral ("virality_reason")
- IMPORTANT: Write the `title`, `hook_text`, and `virality_reason` in the SAME LANGUAGE as the transcript (e.g. if the transcript is in Russian, write them in Russian).

Respond ONLY with valid JSON (no markdown, no explanation):
{{"highlights":[{{"title":"string","start_time":float,"end_time":float,"score":int,"hook_text":"string","virality_reason":"string"}}]}}"""

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

    def _call_llm_with_retry(self, client: OpenAI, model: str, system_prompt: str, user_prompt: str, job: Any = None, temperature: float = 0.2) -> str:
        import time
        max_attempts = 3
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
                    temperature=temperature,
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

    def _clip_text(self, transcript: list[dict], start_sec: float, end_sec: float, max_chars: int = 2500) -> str:
        parts = []
        for seg in transcript:
            if seg["start"] < end_sec and seg["end"] > start_sec:
                text = seg.get("text", "").strip()
                if text:
                    parts.append(text)
        text = " ".join(parts)
        return (text[:max_chars] + "…") if len(text) > max_chars else text

    def _get_speaker_at(self, diarization: list[dict], time_sec: float) -> str | None:
        """Return speaker label at a given time, or None if no diarization."""
        for seg in diarization:
            if seg["start"] <= time_sec <= seg["end"]:
                return seg["speaker"]
        return None

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
        """Skip hook generation since SamuraiGPT prompt already returns hook_sentence."""
        for c in candidates:
            # If for some reason hook_text is missing, fallback to first 50 chars
            fallback = c.get("hook") or c.get("text", "")[:50]
            if not c.get("hook_text"):
                c["hook_text"] = str(fallback)[:60] if fallback else ""
        return candidates




    def detect_content_type(self, transcript: list[dict], job: Any = None) -> dict[str, str]:
        sample = " ".join(s.get("text", "") for s in transcript[:25])[:3000]
        prompt = f"{CONTENT_TYPE_PROMPT}\n\nTranscript sample:\n{sample}"
        try:
            logger.info("Detecting content type...")
            raw = self._call_llm_with_retry(self.client, self.model, "You are a helpful assistant.", prompt, job, temperature=0.0)
            parsed = self._parse_json_loose(raw)
            return parsed
        except Exception as e:
            logger.warning(f"Failed to detect content type: {e}")
            return {"content_type": "other", "density": "medium"}

    def _parse_json_loose(self, raw: str) -> dict:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            raise


    def _group_into_sentences(self, whisper_data: list[dict]) -> list[dict]:
        sentences = []
        if not whisper_data:
            return sentences

        all_words = []
        for seg in whisper_data:
            if "words" in seg and seg["words"]:
                all_words.extend(seg["words"])
            else:
                all_words.append({
                    "word": seg.get("text", "").strip(),
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0)
                })

        current_words = []
        for w in all_words:
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            current_words.append(w)
            is_end = word_text[-1] in ".!?"
            if is_end:
                sentences.append({
                    "text": " ".join([cw.get("word", "") for cw in current_words]),
                    "start_time": current_words[0].get("start", 0.0),
                    "end_time": current_words[-1].get("end", 0.0),
                    "start": current_words[0].get("start", 0.0),
                    "end": current_words[-1].get("end", 0.0)
                })
                current_words = []
        if current_words:
            sentences.append({
                "text": " ".join([cw.get("word", "") for cw in current_words]),
                "start_time": current_words[0].get("start", 0.0),
                "end_time": current_words[-1].get("end", 0.0),
                "start": current_words[0].get("start", 0.0),
                "end": current_words[-1].get("end", 0.0)
            })
        return sentences

    def build_transcript_text(self, transcript: list[dict]) -> str:
        return "\n".join(f"[{s.get('start', s.get('start_time', 0.0)):.1f}s] {s.get('text', '').strip()}" for s in transcript)

    def chunk_transcript(self, transcript: list[dict], chunk_duration_sec: float = 1200, overlap_sec: float = 60) -> list[dict]:
        duration = sentences[-1].get("end", sentences[-1].get("end_time", 0.0)) if sentences else 0.0
        chunks = []
        start = 0.0
        while start < duration:
            end = min(start + chunk_duration_sec, duration)
            chunk_segs = [
                s for s in transcript
                if s.get("start", s.get("start_time", 0.0)) >= start and s.get("end", s.get("end_time", 0.0)) <= end + overlap_sec
            ]
            if chunk_segs:
                chunks.append({
                    "segments": chunk_segs,
                    "duration": end - start,
                    "_offset": start
                })
            start += chunk_duration_sec - overlap_sec
        return chunks

    def dedupe_highlights(self, highlights: list[dict]) -> list[dict]:
        highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
        kept = []
        for h in highlights:
            h_start = float(h["start_time"])
            h_end = float(h["end_time"])
            h_dur = h_end - h_start
            overlapping = False
            for k in kept:
                latest_start = max(h_start, float(k["start_time"]))
                earliest_end = min(h_end, float(k["end_time"]))
                overlap = earliest_end - latest_start
                if overlap > 0 and overlap > 0.5 * h_dur:
                    overlapping = True
                    break
            if not overlapping:
                kept.append(h)
        return kept

    def _sanitize_highlights(self, raw_highlights: list[dict], duration: float) -> list[dict]:
        if not isinstance(raw_highlights, list):
            return []
        max_end = duration if duration > 0 else float("inf")
        cleaned = []
        for item in raw_highlights:
            if not isinstance(item, dict):
                continue
            try:
                start = float(item.get("start_time", -1.0))
                end = float(item.get("end_time", -1.0))
            except (ValueError, TypeError):
                continue
            if start < 0 or end <= start:
                continue
            if max_end != float("inf"):
                start = min(start, max_end)
                end = min(end, max_end)
                if end <= start:
                    continue
            
            # Use fallback hooks if missing
            hook_text = str(item.get("hook_text") or "").strip()
            
            cleaned.append({
                "title": str(item.get("title") or "Untitled Highlight").strip()[:80],
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, int(float(item.get("score", 0))))),
                "hook_text": hook_text,
                "reason": str(item.get("virality_reason") or "").strip()[:320]
            })
        return cleaned

    def call_highlight_api(self, transcript_text: str, content_info: dict, duration: float, is_chunk: bool = False, job: Any = None) -> list[dict]:
        system_prompt = HIGHLIGHT_SYSTEM_PROMPT.format(
            virality_criteria=VIRALITY_CRITERIA,
            content_type=content_info.get("content_type", "other"),
            density=content_info.get("density", "medium"),
            min_clip_len=getattr(settings, "MIN_CLIP_DURATION", 30),
            max_clip_len=getattr(settings, "MAX_CLIP_DURATION", 180)
        )
        user_prompt = f"Transcript:\n{transcript_text}"
        
        logger.info(f"Requesting highlights from {self.provider} ({self.model})...")
        for attempt in range(2):
            if job and getattr(job, "cancel_requested", False):
                raise InterruptedError("Analysis cancelled by user")
            try:
                raw = self._call_llm_with_retry(self.client, self.model, system_prompt, user_prompt, job)
                parsed = self._extract_json_array(raw) # using Klippr's robust regex extraction
                highlights = self._sanitize_highlights(parsed, duration=duration)
                if highlights:
                    return highlights
            except Exception as e:
                logger.warning(f"Failed to extract highlights on attempt {attempt+1}: {e}")
        return []

    def find_highlight_candidates(self, transcript: list[dict], job: Any = None) -> list[dict]:
        sentences = self._group_into_sentences(transcript)
        duration = sentences[-1].get("end", sentences[-1].get("end_time", 0.0)) if sentences else 0.0
        content_info = self.detect_content_type(sentences, job)
        logger.info(f"Content info: {content_info.get('content_type')} | Density: {content_info.get('density')} | Duration: {duration:.0f}s")

        all_highlights = []
        if duration >= 1800: # LONG_VIDEO_THRESHOLD (30 min)
            chunks = self.chunk_transcript(sentences)
            logger.info(f"Long video — splitting into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                if job and getattr(job, "cancel_requested", False):
                    raise InterruptedError("Analysis cancelled by user")
                offset = chunk.get("_offset", 0)
                text = self.build_transcript_text(chunk.get("segments", []))
                logger.info(f"Processing chunk {i + 1}/{len(chunks)} (offset {offset:.0f}s)")
                highlights = self.call_highlight_api(text, content_info, chunk["duration"], is_chunk=True, job=job)
                for h in highlights:
                    h["start_time"] = float(h["start_time"]) + offset
                    h["end_time"] = float(h["end_time"]) + offset
                    all_highlights.append(h)
        else:
            text = self.build_transcript_text(transcript)
            all_highlights = self.call_highlight_api(text, content_info, duration, is_chunk=False, job=job)

        deduped = self.dedupe_highlights(all_highlights)
        selected = sorted(deduped, key=lambda x: x.get("score", 0), reverse=True)
        
        # Backwards compatibility fields for UI
        for clip in selected:
            clip["total_score"] = clip["score"]
            clip["text"] = self._clip_text(sentences, clip["start_time"], clip["end_time"])
        
        selected = sorted(selected, key=lambda x: x["start_time"])
        logger.info(f"Selected {len(selected)} final highlights.")
        
        # Snap to silence
        # The analyzer caller expects find_highlights to ONLY return candidates, and caller calls snap_to_silence.
        # Wait, the caller is main.py `run_analyzer` or `core/projects.py`. 
        # Actually, let's look at how find_highlights was used. In my script I kept `snap_to_silence`. But I'll let the orchestrator handle it.
        return selected
