import os
import subprocess
import logging
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

    def _generate_ass_file(self, transcript_segments: list, style: str, output_path: str, title: str = ""):
        ass_content = f"""[Script Info]
Title: Klippr Subtitles
ScriptType: v4.00+
PlayResX: {self.resolution[0]}
PlayResY: {self.resolution[1]}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: WordByWord,Arial Black,80,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,0,2,30,30,200,1
Style: Title,Arial Black,90,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,0,5,30,30,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        
        def format_time(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            cs = int(round((seconds % 1) * 100))
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        if style == "word_by_word" and transcript_segments:
            # We will create one dialogue event that plays through all segments
            # Calculate the total start and end time of this clip text
            clip_start = transcript_segments[0]['start']
            clip_end = transcript_segments[-1]['end']
            
            ass_text = ""
            for seg in transcript_segments:
                # duration in centiseconds
                duration_cs = int(round((seg['end'] - seg['start']) * 100))
                # Fallback to 1 cs to prevent weird rendering issues
                duration_cs = max(1, duration_cs)
                word = seg['text'].strip()
                ass_text += f"{{\\kf{duration_cs}}}{word} "
            
            # Start/End are relative to the extracted clip in FFmpeg, so we should map them from 0
            # Because we use -ss and -to, FFmpeg restarts video timestamps to 0.
            event_start = "0:00:00.00"
            event_end = format_time(clip_end - clip_start + 1.0) # Add a buffer to keep it on screen
            
            ass_content += f"Dialogue: 0,{event_start},{event_end},WordByWord,,0,0,0,,{ass_text.strip()}\n"
            
        else:
            # Title only
            text = title if title else "Крутой момент!"
            event_start = "0:00:00.00"
            event_end = "0:59:59.00" # Stay on screen forever
            ass_content += f"Dialogue: 0,{event_start},{event_end},Title,,0,0,0,,{text}\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

    def _generate_face_tracking_metadata(self, video_path: str, start: float, end: float, output_path: str):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        crop_w = int(orig_h * (9 / 16))
        
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
        
        out_f = open(output_path, "w")
        
        last_center_x = orig_w / 2.0
        frame_idx = 0
        skip_frames = 3
        
        duration = end - start
        total_frames = int(duration * fps)
        
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
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2 - x1) * (y2 - y1)
                        if area > biggest_area:
                            biggest_area = area
                            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                                kpts = results[0].keypoints.xy[idx].cpu().numpy()
                                if len(kpts) > 0 and kpts[0][0] > 0:
                                    best_center_x = float(kpts[0][0])
                                else:
                                    best_center_x = (x1 + x2) / 2.0
                            else:
                                best_center_x = (x1 + x2) / 2.0
                    
                    last_center_x = 0.8 * last_center_x + 0.2 * best_center_x
            
            x_center = last_center_x
            x1 = int(x_center - crop_w / 2)
            if x1 < 0:
                x1 = 0
            elif x1 + crop_w > orig_w:
                x1 = orig_w - crop_w
                
            timestamp_rel = current_time - start
            out_f.write(f"{timestamp_rel:.3f}-{timestamp_rel+0.5:.3f} crop x {x1}, crop y 0, crop w {crop_w}, crop h {orig_h};\n")
            
            frame_idx += 1
            
        out_f.close()
        cap.release()
        return crop_w, orig_h

    def render_clip(self, video_path: str, highlight: dict, output_path: str, transcript: list[dict] = None):
        logger.info(f"Starting render for clip: '{highlight.get('title')}'")
        
        start = highlight["start_time"]
        end = highlight["end_time"]
        
        # 1. Сгенерировать ASS
        ass_path = os.path.join(self.output_dir, f"subtitles_{os.path.basename(output_path)}.ass")
        
        valid_segments = []
        if transcript:
            for seg in transcript:
                if seg['start'] < end and seg['end'] > start:
                    valid_segments.append(seg)
                    
        self._generate_ass_file(valid_segments, self.subtitle_style, ass_path, highlight.get('title', ''))
        
        # 2. Настроить фильтр
        # For ASS filter in Windows, we must escape backslashes and colons in absolute paths
        ass_path_escaped = ass_path.replace("\\", "\\\\").replace(":", "\\\\:")
        
        if self.crop_mode == "face_tracking":
            cmd_path = os.path.join(self.output_dir, f"crop_{os.path.basename(output_path)}.cmd")
            logger.info("Generating face tracking metadata...")
            crop_w, crop_h = self._generate_face_tracking_metadata(video_path, start, end, cmd_path)
            
            cmd_path_escaped = cmd_path.replace("\\", "/")
            # Use asendcmd/sendcmd syntax
            crop_filter = f"sendcmd=f='{cmd_path_escaped}',crop"
        else:
            # Smart center
            cap = cv2.VideoCapture(video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            crop_w = int(orig_h * (9 / 16))
            crop_filter = f"crop={crop_w}:{orig_h}:(iw-{crop_w})/2:0"
            
        vf_chain = f"{crop_filter},scale={self.resolution[0]}:{self.resolution[1]},ass='{ass_path_escaped}'"
        
        # 3. Собрать команду FFmpeg
        # Используем параметры из настроек
        preset = getattr(settings, 'FFMPEG_PRESET', 'fast')
        crf = str(getattr(settings, 'FFMPEG_CRF', 23))
        use_nvenc = getattr(settings, 'USE_NVENC', False)
        
        vcodec = "h264_nvenc" if use_nvenc else "libx264"
        
        command = [
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-ss", str(start),
            "-to", str(end),
            "-vf", vf_chain,
            "-c:v", vcodec,
            "-preset", "p4" if use_nvenc else preset,
            "-crf" if not use_nvenc else "-cq", crf,
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            output_path
        ]
        
        logger.info(f"Running FFmpeg: {' '.join(command)}")
        
        # 4. Выполнить
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed with error:\n{result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")
            
        logger.info(f"Clip successfully rendered to {output_path}")
        
        # Clean up temp files
        try:
            if os.path.exists(ass_path): os.remove(ass_path)
            if self.crop_mode == "face_tracking" and os.path.exists(cmd_path): os.remove(cmd_path)
        except Exception:
            pass
