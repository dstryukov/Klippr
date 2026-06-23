import os
import logging
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np

# moviepy imports
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, VideoClip
import moviepy.video.fx.all as vfx

from config import settings

logger = logging.getLogger(__name__)

class FaceTracker:
    """Stateful frame processor for tracking faces and smoothing movement using YOLOv8."""
    def __init__(self, model, orig_size, crop_size):
        self.model = model
        self.orig_w, self.orig_h = orig_size
        self.crop_w, self.crop_h = crop_size
        self.last_center_x = self.orig_w / 2.0
        
    def __call__(self, get_frame, t):
        frame = get_frame(t)
        results = self.model(frame, classes=[0], verbose=False)
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            biggest_area = 0
            best_center_x = self.last_center_x
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area > biggest_area:
                    biggest_area = area
                    best_center_x = (x1 + x2) / 2.0
            
            self.last_center_x = 0.8 * self.last_center_x + 0.2 * best_center_x
            
        x_center = self.last_center_x
        
        x1 = int(x_center - self.crop_w / 2)
        x2 = int(x_center + self.crop_w / 2)
        
        if x1 < 0:
            x1 = 0
            x2 = self.crop_w
        elif x2 > self.orig_w:
            x2 = self.orig_w
            x1 = self.orig_w - self.crop_w
            
        cropped = frame[:, x1:x2, :]
        return cropped


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
                logger.info(f"Loading YOLOv8n on {device}...")
                self.yolo_model = YOLO('yolov8n.pt')
                self.yolo_model.to(device)
            except ImportError:
                logger.error("ultralytics not installed. Falling back to smart_center.")
                self.crop_mode = "smart_center"

    def _create_title_only_clip(self, text: str, size: tuple, duration: float) -> ImageClip:
        """Style 1: Centered static title with stroke."""
        width, height = size
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        font_size = settings.SUBTITLE_FONT_SIZE
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # Center Y at 40% of height (Safe zone: 15-80%)
        y_center = int(height * 0.40)
        
        max_chars = max(10, int((width * 0.8) / (font_size * 0.55)))
        wrapped_text = textwrap.fill(text, width=max_chars)
        
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        x = (width - text_w) // 2
        y = y_center - (text_h // 2)
        
        stroke = 3
        stroke_color = settings.SUBTITLE_STROKE_COLOR
        text_color = settings.SUBTITLE_COLOR
        
        # Draw 8-way stroke/shadow
        for offset_x in [-stroke, 0, stroke]:
            for offset_y in [-stroke, 0, stroke]:
                if offset_x == 0 and offset_y == 0: continue
                draw.multiline_text((x + offset_x, y + offset_y), wrapped_text, font=font, fill=stroke_color, align="center")
                
        draw.multiline_text((x, y), wrapped_text, font=font, fill=text_color, align="center")
        
        return ImageClip(np.array(img)).set_duration(duration)

    def _create_word_by_word_clip(self, text: str, size: tuple, duration: float) -> VideoClip:
        """Style 2: Opus Clip style word-by-word highlight."""
        width, height = size
        font_size = 50 # Slightly smaller for paragraph
        
        words = text.split()
        if not words:
            return ImageClip(np.zeros((height, width, 4), dtype=np.uint8)).set_duration(duration)
            
        word_duration = duration / len(words)
        y_center = int(height * 0.70) # Lower third
        
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        def make_frame(t):
            img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            current_word_idx = min(int(t / word_duration), len(words) - 1)
            
            # Wrap text manually to calculate exact word coordinates
            lines = []
            current_line = []
            max_line_width = width * 0.8
            
            for w in words:
                current_line.append(w)
                line_text = " ".join(current_line)
                bbox = draw.textbbox((0,0), line_text, font=font)
                w_w = bbox[2] - bbox[0]
                if w_w > max_line_width and len(current_line) > 1:
                    current_line.pop()
                    lines.append(current_line)
                    current_line = [w]
            if current_line:
                lines.append(current_line)
                
            total_height = 0
            line_heights = []
            for line in lines:
                bbox = draw.textbbox((0,0), " ".join(line), font=font)
                lh = bbox[3] - bbox[1]
                line_heights.append(lh)
                total_height += lh + 10 # line spacing
                
            start_y = y_center - (total_height // 2)
            word_idx = 0
            current_y = start_y
            
            stroke = 3
            stroke_color = settings.SUBTITLE_STROKE_COLOR
            
            for i, line in enumerate(lines):
                line_text = " ".join(line)
                bbox = draw.textbbox((0,0), line_text, font=font)
                lw = bbox[2] - bbox[0]
                current_x = (width - lw) // 2
                
                for w in line:
                    is_active = (word_idx == current_word_idx)
                    color = "yellow" if is_active else settings.SUBTITLE_COLOR
                    
                    # Add stroke
                    for offset_x in [-stroke, 0, stroke]:
                        for offset_y in [-stroke, 0, stroke]:
                            if offset_x == 0 and offset_y == 0: continue
                            draw.text((current_x + offset_x, current_y + offset_y), w, font=font, fill=stroke_color)
                            
                    draw.text((current_x, current_y), w, font=font, fill=color)
                    
                    # Move X for next word
                    w_bbox = draw.textbbox((0,0), w + " ", font=font)
                    current_x += w_bbox[2] - w_bbox[0]
                    word_idx += 1
                    
                current_y += line_heights[i] + 10
                
            return np.array(img)
            
        return VideoClip(make_frame, duration=duration)

    def _apply_smart_center_crop(self, clip: VideoFileClip) -> VideoFileClip:
        orig_w, orig_h = clip.size
        target_w, target_h = self.resolution
        target_ratio = target_w / target_h
        current_ratio = orig_w / orig_h
        
        if current_ratio > target_ratio:
            new_w = int(orig_h * target_ratio)
            x_center = orig_w / 2
            y_center = orig_h / 2
            cropped_clip = vfx.crop(clip, x_center=x_center, y_center=y_center, width=new_w, height=orig_h)
        else:
            new_h = int(orig_w / target_ratio)
            x_center = orig_w / 2
            y_center = orig_h / 2
            cropped_clip = vfx.crop(clip, x_center=x_center, y_center=y_center, width=orig_w, height=new_h)
            
        return cropped_clip

    def _apply_face_tracking_crop(self, clip: VideoFileClip) -> VideoFileClip:
        orig_w, orig_h = clip.size
        target_w, target_h = self.resolution
        target_ratio = target_w / target_h
        current_ratio = orig_w / orig_h
        
        if current_ratio > target_ratio:
            crop_w = int(orig_h * target_ratio)
            crop_h = orig_h
        else:
            crop_w = orig_w
            crop_h = int(orig_w / target_ratio)

        tracker = FaceTracker(self.yolo_model, clip.size, (crop_w, crop_h))
        tracked_clip = clip.fl(tracker)
        return tracked_clip

    def render_clip(self, video_path: str, highlight: dict, output_path: str):
        logger.info(f"Starting render for clip: '{highlight.get('title')}'")
        try:
            clip = VideoFileClip(video_path).subclip(highlight["start_time"], highlight["end_time"])
            
            orig_w, orig_h = clip.size
            logger.info(f"Cropping video from {orig_w}x{orig_h} to {self.resolution[0]}x{self.resolution[1]}, mode: {self.crop_mode}")
            
            if self.crop_mode == "face_tracking":
                cropped_clip = self._apply_face_tracking_crop(clip)
            else:
                cropped_clip = self._apply_smart_center_crop(clip)
                
            final_visual = cropped_clip.resize(newsize=self.resolution)
            
            title_text = highlight.get("title", "Крутой момент!")
            logger.info(f"Adding subtitles (Style: {self.subtitle_style})")
            
            if self.subtitle_style == "word_by_word":
                txt_clip = self._create_word_by_word_clip(title_text, self.resolution, duration=final_visual.duration)
            else:
                txt_clip = self._create_title_only_clip(title_text, self.resolution, duration=final_visual.duration)
            
            final_clip = CompositeVideoClip([final_visual, txt_clip])
            
            logger.info(f"Writing final video to {output_path} (this might take a minute)...")
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile="temp-audio.m4a", 
                remove_temp=True, 
                fps=30,
                threads=4,
                logger=None
            )
            
            clip.close()
            txt_clip.close()
            final_clip.close()
            logger.info("Render completed successfully.")
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
            raise
