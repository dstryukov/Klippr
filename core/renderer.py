import os
import subprocess
import logging
import re
import time
from config import settings
import cv2

import imageio_ffmpeg

logger = logging.getLogger(__name__)

FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()


class VerticalRenderer:
    def __init__(self, output_dir: str = "output", resolution: tuple = (1080, 1920)):
        self.output_dir = output_dir
        self.resolution = resolution
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.crop_mode = settings.CROP_MODE
        self.subtitle_style = settings.SUBTITLE_STYLE
        
        if self.crop_mode == "face_tracking":
            try:
                from ultralytics import YOLO
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                logger.info(f"Loading YOLOv8n-pose on {device}...")
                self.yolo_model = YOLO('yolov8n-pose.pt')
                self.yolo_model.to(device)
            except ImportError:
                logger.error("ultralytics not installed. Falling back to smart_center.")
                self.crop_mode = "smart_center"

    def _ass_color(self, value: str, default: str = "&H00FFFFFF") -> str:
        named = {
            "white": "#FFFFFF",
            "black": "#000000",
            "yellow": "#FFFF00",
            "red": "#FF0000",
            "green": "#00FF00",
            "blue": "#0000FF",
        }
        raw = str(value or "").strip().lower()
        raw = named.get(raw, raw)
        if re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
            r = raw[1:3]
            g = raw[3:5]
            b = raw[5:7]
            return f"&H00{b}{g}{r}".upper()
        return default

    def _ass_inline_color(self, value: str, default: str = "&HFFFFFF&") -> str:
        ass = self._ass_color(value, default="&H00FFFFFF")
        if ass.startswith("&H00") and len(ass) == 10:
            return f"&H{ass[4:]}&"
        return default

    def _escape_ass_text(self, text: str) -> str:
        return (
            str(text or "")
            .replace("\\", r"\\")
            .replace("{", "")
            .replace("}", "")
            .replace("\n", " ")
            .strip()
        )

    def _format_time(self, seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int(round((seconds % 1) * 100))
        if cs >= 100:
            s += 1
            cs = 0
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _subtitle_offset_seconds(self) -> float:
        return float(getattr(settings, "SUBTITLE_TIMING_OFFSET_MS", -80)) / 1000.0

    def _word_events_from_transcript(self, transcript_segments: list[dict], clip_start: float, clip_end: float) -> list[dict]:
        events = []
        for seg in transcript_segments or []:
            words = seg.get("words") or []
            if words:
                for word in words:
                    text = self._escape_ass_text(word.get("word") or word.get("text", ""))
                    if not text:
                        continue
                    start = float(word.get("start", seg["start"]))
                    end = float(word.get("end", start + 0.28))
                    start = max(start, clip_start)
                    end = min(end, clip_end)
                    if end <= start:
                        end = min(start + 0.28, clip_end)
                    events.append({"start_abs": start, "end_abs": end, "text": text})
            else:
                split_words = [self._escape_ass_text(w) for w in seg.get("text", "").split() if w.strip()]
                if not split_words:
                    continue
                seg_start = max(float(seg["start"]), clip_start)
                seg_end = min(float(seg["end"]), clip_end)
                if seg_end <= seg_start:
                    continue
                step = max(0.22, (seg_end - seg_start) / max(1, len(split_words)))
                for idx, text in enumerate(split_words):
                    start = seg_start + idx * step
                    end = min(start + step, clip_end)
                    events.append({"start_abs": start, "end_abs": end, "text": text})

        events = sorted(events, key=lambda e: (e["start_abs"], e["end_abs"]))
        cleaned = []
        for event in events:
            if not cleaned:
                cleaned.append(event)
                continue
            prev = cleaned[-1]
            if event["start_abs"] < prev["start_abs"]:
                continue
            if event["start_abs"] < prev["end_abs"] and event["start_abs"] - prev["start_abs"] > 0.02:
                prev["end_abs"] = max(prev["start_abs"] + 0.08, event["start_abs"])
            cleaned.append(event)
        return cleaned

    def _caption_window(self, words: list[dict], active_idx: int, words_per_caption: int) -> tuple[int, int]:
        max_words = max(1, min(int(words_per_caption), 5))
        start = active_idx
        end = active_idx + 1
        while start > 0 and end - start < max_words:
            prev_word = words[start - 1]
            current_word = words[start]
            gap = current_word["start_abs"] - prev_word["end_abs"]
            if gap > 0.65 or prev_word["text"].endswith((".", "!", "?", "…")):
                break
            start -= 1
            break
        while end < len(words) and end - start < max_words:
            prev_word = words[end - 1]
            next_word = words[end]
            gap = next_word["start_abs"] - prev_word["end_abs"]
            if gap > 0.65 or prev_word["text"].endswith((".", "!", "?", "…")):
                break
            end += 1
        while end - start < max_words and start > 0:
            prev_word = words[start - 1]
            current_word = words[start]
            gap = current_word["start_abs"] - prev_word["end_abs"]
            if gap > 0.65 or prev_word["text"].endswith((".", "!", "?", "…")):
                break
            start -= 1
        return start, end

    def _caption_text(self, words: list[dict], start_idx: int, end_idx: int, active_idx: int) -> str:
        normal_color = self._ass_inline_color(getattr(settings, "SUBTITLE_COLOR", "#FFFFFF"), default="&HFFFFFF&")
        active_color = self._ass_inline_color(getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00"), default="&H00FFFF&")
        pieces = []
        for idx in range(start_idx, end_idx):
            word = words[idx]["text"]
            if idx == active_idx:
                pieces.append(
                    rf"{{\c{active_color}\fscx112\fscy112\bord7}}{word}"
                    rf"{{\c{normal_color}\fscx100\fscy100\bord5}}"
                )
            else:
                pieces.append(word)
        return " ".join(pieces)

    def _generate_ass_file(
        self,
        transcript_segments: list,
        style: str,
        output_path: str,
        title: str = "",
        clip_start: float = 0.0,
        clip_end: float = 0.0,
        hook_text: str = "",
    ):
        font_size = int(getattr(settings, "SUBTITLE_FONT_SIZE", 70))
        primary = self._ass_color(getattr(settings, "SUBTITLE_COLOR", "#FFFFFF"))
        active = self._ass_color(getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00"), default="&H0000FFFF")
        outline = self._ass_color(getattr(settings, "SUBTITLE_STROKE_COLOR", "#000000"), default="&H00000000")
        title_size = max(font_size, 76)
        word_margin_v = int(self.resolution[1] * 0.16)
        title_margin_v = int(self.resolution[1] * 0.12)

        ass_content = f"""[Script Info]
Title: Klippr Subtitles
ScriptType: v4.00+
PlayResX: {self.resolution[0]}
PlayResY: {self.resolution[1]}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial Black,{font_size},{primary},{active},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,40,40,{word_margin_v},1
Style: Title,Arial Black,{title_size},{primary},{active},{outline},&H80000000,-1,0,0,0,100,100,0,0,1,5,0,5,40,40,{title_margin_v},1
"""
        # Hook overlay style (rendered at top of video for first few seconds)
        hook_enabled = bool(getattr(settings, "HOOK_OVERLAY_ENABLED", True))
        if hook_enabled and hook_text:
            hook_font_size = int(getattr(settings, "HOOK_OVERLAY_FONT_SIZE", 80))
            hook_color = self._ass_color(getattr(settings, "HOOK_OVERLAY_COLOR", "#FFFFFF"))
            hook_bg = self._ass_color(getattr(settings, "HOOK_OVERLAY_BG_COLOR", "#000000"))
            # Semi-transparent background via BackColour with alpha
            hook_bg_semi = hook_bg.replace("&H00", "&H80", 1) if hook_bg.startswith("&H00") else hook_bg
            hook_outline = self._ass_color("#000000", default="&H00000000")
            hook_margin_v = int(self.resolution[1] * 0.04)
            ass_content += (
                f"Style: HookOverlay,Arial Black,{hook_font_size},{hook_color},"
                f"{hook_color},{hook_outline},{hook_bg_semi},-1,0,0,0,100,100,0,0,1,6,2,8,"
                f"60,60,{hook_margin_v},1\n"
            )

        ass_content += """
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        if style == "word_by_word":
            word_events = self._word_events_from_transcript(transcript_segments, clip_start, clip_end)
            if word_events:
                offset = self._subtitle_offset_seconds()
                words_per_caption = int(getattr(settings, "SUBTITLE_WORDS_PER_CAPTION", 3))
                clip_duration = max(0.1, clip_end - clip_start)
                logger.info(
                    "Generating word-level subtitles: %s words, offset=%sms, words_per_caption=%s",
                    len(word_events),
                    int(offset * 1000),
                    words_per_caption,
                )
                for idx, word in enumerate(word_events):
                    start_idx, end_idx = self._caption_window(word_events, idx, words_per_caption)
                    event_start = max(0.0, word["start_abs"] - clip_start + offset)
                    if idx + 1 < len(word_events):
                        next_start = word_events[idx + 1]["start_abs"] - clip_start + offset
                        natural_end = word["end_abs"] - clip_start + offset + 0.04
                        if next_start - event_start <= 0.75:
                            event_end = next_start
                        else:
                            event_end = natural_end
                    else:
                        event_end = word["end_abs"] - clip_start + offset + 0.12
                    event_end = min(clip_duration, max(event_start + 0.10, event_end))
                    if event_start >= clip_duration:
                        continue
                    text = self._caption_text(word_events, start_idx, end_idx, idx)
                    ass_content += (
                        f"Dialogue: 0,{self._format_time(event_start)},{self._format_time(event_end)},"
                        f"Caption,,0,0,0,,{text}\n"
                    )
            else:
                logger.warning("No word timestamps available for word_by_word subtitles; falling back to title_only.")
                style = "title_only"

        if style != "word_by_word":
            text = self._escape_ass_text(title if title else "Крутой момент!")
            event_start = "0:00:00.00"
            event_end = self._format_time(max(clip_end - clip_start, 1.0))
            ass_content += f"Dialogue: 0,{event_start},{event_end},Title,,0,0,0,,{text}\n"

        # Hook overlay: show hook text at top for first N seconds
        if hook_enabled and hook_text:
            hook_duration = int(getattr(settings, "HOOK_OVERLAY_DURATION", 4))
            clip_duration = max(0.1, clip_end - clip_start)
            hook_end = min(hook_duration, clip_duration)
            escaped_hook = self._escape_ass_text(hook_text)
            # Add fade-in effect (200ms) and fade-out (300ms)
            ass_content += (
                f"Dialogue: 1,0:00:00.00,{self._format_time(hook_end)},"
                f"HookOverlay,,0,0,0,,"
                f"{{\\fad(200,300)}}{escaped_hook}\n"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

    def _get_video_dimensions(self, video_path: str) -> tuple[int, int]:
        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if orig_w <= 0 or orig_h <= 0:
            raise RuntimeError(f"Could not read video dimensions for {video_path}")
        return orig_w, orig_h

    def _smart_crop_params(self, orig_w: int, orig_h: int, center_x: float | None = None) -> tuple[int, int, int, int]:
        target_ar = self.resolution[0] / self.resolution[1]
        source_ar = orig_w / orig_h
        if source_ar >= target_ar:
            crop_h = orig_h
            crop_w = min(orig_w, int(round(orig_h * target_ar)))
            if crop_w % 2:
                crop_w -= 1
            if center_x is None:
                center_x = orig_w / 2.0
            x1 = int(round(center_x - crop_w / 2))
            x1 = max(0, min(x1, orig_w - crop_w))
            y1 = 0
        else:
            crop_w = orig_w if orig_w % 2 == 0 else orig_w - 1
            crop_h = min(orig_h, int(round(orig_w / target_ar)))
            if crop_h % 2:
                crop_h -= 1
            x1 = 0
            y1 = max(0, int(round((orig_h - crop_h) / 2)))
        return crop_w, crop_h, x1, y1

    def _generate_face_tracking_metadata(self, video_path: str, start: float, end: float, output_path: str):
        """Two-pass face tracking with adaptive EMA, velocity clamp, and lookahead smoothing.

        Pass 1: detect faces every N frames, build raw center_x trajectory with adaptive EMA.
        Pass 2: apply sliding-window (lookahead) smoothing for jitter-free output.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        crop_w, crop_h, x1, y1 = self._smart_crop_params(orig_w, orig_h)
        last_center_x = x1 + crop_w / 2.0

        # Configurable parameters
        skip_frames = int(getattr(settings, "FACE_TRACKING_SKIP_FRAMES", 5))
        max_shift_px = int(getattr(settings, "FACE_TRACKING_MAX_SHIFT", 40))
        lookahead_window = int(getattr(settings, "FACE_TRACKING_LOOKAHEAD", 5))

        # --- Pass 1: build raw trajectory with adaptive EMA + velocity clamp ---
        raw_centers: list[tuple[float, float]] = []  # (timestamp_rel, center_x)
        frame_idx = 0
        prev_crop_x: float | None = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end:
                break
            timestamp_rel = max(0.0, current_time - start)

            if frame_idx % skip_frames == 0:
                results = self.yolo_model(frame, classes=[0], verbose=False)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    biggest_area = 0
                    best_center_x = last_center_x
                    for idx, box in enumerate(boxes):
                        x_a, y_a, x_b, y_b = box.xyxy[0].cpu().numpy()
                        area = (x_b - x_a) * (y_b - y_a)
                        if area > biggest_area:
                            biggest_area = area
                            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                                kpts = results[0].keypoints.xy[idx].cpu().numpy()
                                best_center_x = float(kpts[0][0]) if len(kpts) > 0 and kpts[0][0] > 0 else (x_a + x_b) / 2.0
                            else:
                                best_center_x = (x_a + x_b) / 2.0

                    # Adaptive EMA: alpha scales with velocity
                    velocity = abs(best_center_x - last_center_x)
                    alpha = max(0.05, min(0.35, 0.05 + 0.003 * velocity))
                    last_center_x = (1.0 - alpha) * last_center_x + alpha * best_center_x
                # else: no face detected, keep last_center_x unchanged

            # Velocity clamp on crop_x
            crop_w_f, crop_h_f, new_x1, new_y1 = self._smart_crop_params(orig_w, orig_h, center_x=last_center_x)
            if prev_crop_x is not None:
                delta = new_x1 - prev_crop_x
                if abs(delta) > max_shift_px:
                    new_x1 = int(prev_crop_x + max_shift_px * (1 if delta > 0 else -1))
            prev_crop_x = float(new_x1)

            raw_centers.append((timestamp_rel, last_center_x))
            frame_idx += 1

        cap.release()

        # --- Pass 2: lookahead smoothing (sliding window average) ---
        if len(raw_centers) > lookahead_window * 2:
            smoothed: list[float] = []
            half_w = max(1, lookahead_window // 2)
            center_values = [c[1] for c in raw_centers]
            for i in range(len(center_values)):
                lo = max(0, i - half_w)
                hi = min(len(center_values), i + half_w + 1)
                window = center_values[lo:hi]
                smoothed.append(sum(window) / len(window))
        else:
            smoothed = [c[1] for c in raw_centers]

        # --- Write sendcmd file ---
        out_f = open(output_path, "w", encoding="utf-8")
        for i, (ts_rel, _) in enumerate(raw_centers):
            cx = smoothed[i]
            crop_w_f, crop_h_f, x1_f, y1_f = self._smart_crop_params(orig_w, orig_h, center_x=cx)
            next_ts = raw_centers[i + 1][0] if i + 1 < len(raw_centers) else ts_rel + 0.5
            out_f.write(f"{ts_rel:.3f}-{next_ts:.3f} crop x {x1_f};\n")
        out_f.close()

        return crop_w, crop_h, y1

    def _ass_filter_path(self, path: str) -> str:
        return path.replace("\\", "/").replace(":", r"\:")

    def _should_use_nvenc(self) -> bool:
        requested = bool(getattr(settings, 'USE_NVENC', False)) or getattr(settings, 'DEVICE', 'cpu') == 'cuda'
        if not requested:
            return False
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("NVENC/GPU render requested, but CUDA is not available to torch. Falling back to libx264.")
                return False
        except Exception as e:
            logger.warning("Could not check CUDA availability for NVENC (%s). Falling back to libx264.", e)
            return False
        return True

    def _build_render_command(self, video_path: str, output_path: str, start: float, end: float, vf_chain: str, vcodec: str, preset: str, crf: str, use_nvenc: bool) -> list[str]:
        duration = max(0.1, end - start)
        af_chain = f"atrim=start={start}:duration={duration},asetpts=PTS-STARTPTS"
        command = [FFMPEG_PATH, "-y", "-i", video_path, "-vf", vf_chain, "-af", af_chain, "-c:v", vcodec]
        if use_nvenc:
            command += ["-preset", "p4", "-cq", crf]
        else:
            command += ["-preset", preset, "-crf", crf]
        command += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", output_path]
        return command

    def render_clip(self, video_path: str, highlight: dict, output_path: str, transcript: list[dict] = None):
        logger.info(f"Starting render for clip: '{highlight.get('title')}'")
        t0 = time.monotonic()
        start = float(highlight["start_time"])
        end = float(highlight["end_time"])
        duration = max(0.1, end - start)
        ass_path = os.path.join(self.output_dir, f"subtitles_{os.path.basename(output_path)}.ass")
        valid_segments = []
        if transcript:
            for seg in transcript:
                if seg['start'] < end and seg['end'] > start:
                    valid_segments.append(seg)
        self._generate_ass_file(valid_segments, self.subtitle_style, ass_path, highlight.get('title', ''), clip_start=start, clip_end=end, hook_text=highlight.get('hook_text', ''))
        ass_path_escaped = self._ass_filter_path(ass_path)
        orig_w, orig_h = self._get_video_dimensions(video_path)
        if self.crop_mode == "face_tracking":
            cmd_path = os.path.join(self.output_dir, f"crop_{os.path.basename(output_path)}.cmd")
            logger.info("Generating face tracking metadata...")
            t_track = time.monotonic()
            crop_w, crop_h, crop_y = self._generate_face_tracking_metadata(video_path, start, end, cmd_path)
            logger.info("Face tracking metadata generated in %.1fs", time.monotonic() - t_track)
            cmd_path_escaped = cmd_path.replace("\\", "/")
            crop_filter = f"sendcmd=f='{cmd_path_escaped}',crop={crop_w}:{crop_h}:0:{crop_y}"
        else:
            crop_w, crop_h, crop_x, crop_y = self._smart_crop_params(orig_w, orig_h)
            crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        vf_chain = (
            f"trim=start={start}:duration={duration},setpts=PTS-STARTPTS,"
            f"{crop_filter},scale={self.resolution[0]}:{self.resolution[1]},setsar=1,"
            f"ass='{ass_path_escaped}'"
        )
        preset = getattr(settings, 'FFMPEG_PRESET', 'fast')
        crf = str(getattr(settings, 'FFMPEG_CRF', 23))
        use_nvenc = self._should_use_nvenc()
        vcodec = "h264_nvenc" if use_nvenc else "libx264"
        logger.info("Rendering vertical clip at %sx%s using %s with frame-accurate trim", self.resolution[0], self.resolution[1], vcodec)
        command = self._build_render_command(video_path, output_path, start, end, vf_chain, vcodec, preset, crf, use_nvenc)
        logger.info(f"Running FFmpeg: {' '.join(command)}")
        t_ffmpeg = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 and use_nvenc:
            logger.warning("NVENC render failed after %.1fs, retrying with CPU libx264. Error was:\n%s", time.monotonic() - t_ffmpeg, result.stderr)
            command = self._build_render_command(video_path, output_path, start, end, vf_chain, "libx264", preset, crf, False)
            logger.info(f"Running FFmpeg fallback: {' '.join(command)}")
            t_ffmpeg = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg failed with error:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")
        logger.info("FFmpeg encode took %.1fs", time.monotonic() - t_ffmpeg)
        logger.info("Clip '%s' rendered in %.1fs total", highlight.get('title'), time.monotonic() - t0)
        try:
            if os.path.exists(ass_path):
                os.remove(ass_path)
            if self.crop_mode == "face_tracking" and 'cmd_path' in locals() and os.path.exists(cmd_path):
                os.remove(cmd_path)
        except Exception:
            pass
