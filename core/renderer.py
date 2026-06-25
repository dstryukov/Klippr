import os
import subprocess
import logging
import re
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

    def _escape_ass_text(self, text: str) -> str:
        return str(text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ").strip()

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

    def _word_events_from_transcript(self, transcript_segments: list[dict], clip_start: float, clip_end: float) -> list[dict]:
        events = []
        for seg in transcript_segments or []:
            words = seg.get("words") or []
            if words:
                for word in words:
                    text = self._escape_ass_text(word.get("text", ""))
                    if not text:
                        continue
                    start = max(float(word.get("start", seg["start"])), clip_start)
                    end = min(float(word.get("end", start + 0.35)), clip_end)
                    if end <= start:
                        end = min(start + 0.35, clip_end)
                    events.append({"start": start - clip_start, "end": end - clip_start, "text": text})
            else:
                # Fallback for old transcripts without word timestamps: distribute words across the segment.
                split_words = [self._escape_ass_text(w) for w in seg.get("text", "").split() if w.strip()]
                if not split_words:
                    continue
                seg_start = max(float(seg["start"]), clip_start)
                seg_end = min(float(seg["end"]), clip_end)
                step = max(0.25, (seg_end - seg_start) / max(1, len(split_words)))
                for idx, text in enumerate(split_words):
                    start = seg_start + idx * step
                    end = min(start + step, clip_end)
                    events.append({"start": start - clip_start, "end": end - clip_start, "text": text})
        return [e for e in events if e["end"] > e["start"]]

    def _generate_ass_file(
        self,
        transcript_segments: list,
        style: str,
        output_path: str,
        title: str = "",
        clip_start: float = 0.0,
        clip_end: float = 0.0,
    ):
        font_size = int(getattr(settings, "SUBTITLE_FONT_SIZE", 70))
        primary = self._ass_color(getattr(settings, "SUBTITLE_COLOR", "#FFFFFF"))
        outline = self._ass_color(getattr(settings, "SUBTITLE_STROKE_COLOR", "#000000"), default="&H00000000")
        title_size = max(font_size, 76)
        word_margin_v = int(self.resolution[1] * 0.18)
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
Style: PopWord,Arial Black,{font_size},{primary},&H0000FFFF,{outline},&H80000000,-1,0,0,0,100,100,0,0,1,5,0,2,40,40,{word_margin_v},1
Style: Title,Arial Black,{title_size},{primary},&H0000FFFF,{outline},&H80000000,-1,0,0,0,100,100,0,0,1,5,0,5,40,40,{title_margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        if style == "word_by_word":
            word_events = self._word_events_from_transcript(transcript_segments, clip_start, clip_end)
            if word_events:
                for event in word_events:
                    # CapCut-like pop: the word appears exactly when spoken, grows slightly, then settles.
                    text = r"{\fad(35,70)\fscx115\fscy115\t(0,90,\fscx100\fscy100)}" + event["text"]
                    ass_content += (
                        f"Dialogue: 0,{self._format_time(event['start'])},{self._format_time(event['end'])},"
                        f"PopWord,,0,0,0,,{text}\n"
                    )
            else:
                logger.warning("No word timestamps available for word_by_word subtitles; falling back to title_only.")
                style = "title_only"

        if style != "word_by_word":
            text = self._escape_ass_text(title if title else "Крутой момент!")
            event_start = "0:00:00.00"
            event_end = self._format_time(max(clip_end - clip_start, 1.0))
            ass_content += f"Dialogue: 0,{event_start},{event_end},Title,,0,0,0,,{text}\n"

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
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        crop_w, crop_h, x1, y1 = self._smart_crop_params(orig_w, orig_h)
        last_center_x = x1 + crop_w / 2.0
        frame_idx = 0
        skip_frames = 3
        out_f = open(output_path, "w", encoding="utf-8")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end:
                break
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
                    last_center_x = 0.82 * last_center_x + 0.18 * best_center_x
            crop_w, crop_h, x1, y1 = self._smart_crop_params(orig_w, orig_h, center_x=last_center_x)
            timestamp_rel = max(0.0, current_time - start)
            out_f.write(f"{timestamp_rel:.3f}-{timestamp_rel + 0.5:.3f} crop x {x1};\n")
            frame_idx += 1
        out_f.close()
        cap.release()
        return crop_w, crop_h, y1

    def _ass_filter_path(self, path: str) -> str:
        # FFmpeg's ass filter is picky on Windows. Forward slashes work better than backslashes.
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

    def render_clip(self, video_path: str, highlight: dict, output_path: str, transcript: list[dict] = None):
        logger.info(f"Starting render for clip: '{highlight.get('title')}'")
        start = float(highlight["start_time"])
        end = float(highlight["end_time"])
        
        ass_path = os.path.join(self.output_dir, f"subtitles_{os.path.basename(output_path)}.ass")
        valid_segments = []
        if transcript:
            for seg in transcript:
                if seg['start'] < end and seg['end'] > start:
                    valid_segments.append(seg)
        self._generate_ass_file(valid_segments, self.subtitle_style, ass_path, highlight.get('title', ''), clip_start=start, clip_end=end)
        ass_path_escaped = self._ass_filter_path(ass_path)
        
        orig_w, orig_h = self._get_video_dimensions(video_path)
        if self.crop_mode == "face_tracking":
            cmd_path = os.path.join(self.output_dir, f"crop_{os.path.basename(output_path)}.cmd")
            logger.info("Generating face tracking metadata...")
            crop_w, crop_h, crop_y = self._generate_face_tracking_metadata(video_path, start, end, cmd_path)
            cmd_path_escaped = cmd_path.replace("\\", "/")
            crop_filter = f"sendcmd=f='{cmd_path_escaped}',crop={crop_w}:{crop_h}:0:{crop_y}"
        else:
            crop_w, crop_h, crop_x, crop_y = self._smart_crop_params(orig_w, orig_h)
            crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}"
        
        vf_chain = f"{crop_filter},scale={self.resolution[0]}:{self.resolution[1]},setsar=1,ass='{ass_path_escaped}'"
        preset = getattr(settings, 'FFMPEG_PRESET', 'fast')
        crf = str(getattr(settings, 'FFMPEG_CRF', 23))
        use_nvenc = self._should_use_nvenc()
        vcodec = "h264_nvenc" if use_nvenc else "libx264"
        logger.info("Rendering vertical clip at %sx%s using %s", self.resolution[0], self.resolution[1], vcodec)
        
        command = [
            FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
            "-vf", vf_chain,
            "-c:v", vcodec,
        ]
        if use_nvenc:
            command += ["-preset", "p4", "-cq", crf]
        else:
            command += ["-preset", preset, "-crf", crf]
        command += [
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path,
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0 and use_nvenc:
            logger.warning("NVENC render failed, retrying with CPU libx264. Error was:\n%s", result.stderr)
            command = [
                FFMPEG_PATH, "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", video_path,
                "-vf", vf_chain,
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", crf,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                output_path,
            ]
            logger.info(f"Running FFmpeg fallback: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg failed with error:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")
        logger.info(f"Clip successfully rendered to {output_path}")
        
        try:
            if os.path.exists(ass_path):
                os.remove(ass_path)
            if self.crop_mode == "face_tracking" and 'cmd_path' in locals() and os.path.exists(cmd_path):
                os.remove(cmd_path)
        except Exception:
            pass
