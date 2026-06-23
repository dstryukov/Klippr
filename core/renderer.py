import os
import logging
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np

# moviepy imports
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import moviepy.video.fx.all as vfx

logger = logging.getLogger(__name__)

class VerticalRenderer:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_text_overlay(self, text: str, size: tuple, font_size: int = 60) -> np.ndarray:
        """
        Creates a transparent image with text centered. 
        Uses Pillow to avoid moviepy's ImageMagick dependency issues on Windows.
        """
        width, height = size
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load Arial Bold (standard on Windows)
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except IOError:
            # Fallback to default
            font = ImageFont.load_default()
            logger.warning("Arial Bold font not found, falling back to default Pillow font.")

        # Wrap text so it fits the screen width (80% of width)
        max_chars = max(10, int((width * 0.8) / (font_size * 0.55)))
        wrapped_text = textwrap.fill(text, width=max_chars)
        
        # Calculate text size and position
        bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Center horizontally, place slightly below center vertically
        x = (width - text_w) // 2
        y = (height - text_h) // 2
        
        # Draw background box (black with 70% opacity)
        pad = 20
        draw.rectangle(
            [x - pad, y - pad, x + text_w + pad, y + text_h + pad],
            fill=(0, 0, 0, 180)
        )
        
        # Draw text with simple fake stroke/shadow for readability
        stroke_color = (0, 0, 0, 255)
        text_color = (255, 255, 255, 255)
        for offset_x in [-2, 2]:
            for offset_y in [-2, 2]:
                draw.multiline_text((x + offset_x, y + offset_y), wrapped_text, font=font, fill=stroke_color, align="center")
                
        draw.multiline_text((x, y), wrapped_text, font=font, fill=text_color, align="center")
        
        return np.array(img)

    def render_clip(self, video_path: str, highlight: dict, output_path: str):
        """
        Cuts the video, center-crops to 9:16, adds text overlay, and exports.
        """
        logger.info(f"Starting render for clip: '{highlight.get('title')}'")
        try:
            # 1. Load video and cut
            logger.info(f"Cutting video from {highlight['start_time']} to {highlight['end_time']}...")
            clip = VideoFileClip(video_path).subclip(highlight["start_time"], highlight["end_time"])
            
            # 2. Target 9:16 aspect ratio (Vertical format)
            w, h = clip.size
            target_ratio = 9.0 / 16.0
            current_ratio = w / h
            
            logger.info(f"Original size: {w}x{h}. Target ratio: 9:16. Cropping...")
            if current_ratio > target_ratio:
                # Video is wider than 9:16 (e.g., 16:9) -> Crop width (Center crop)
                new_w = int(h * target_ratio)
                x_center = w / 2
                y_center = h / 2
                clip = vfx.crop(clip, x_center=x_center, y_center=y_center, width=new_w, height=h)
            else:
                # Video is taller than 9:16 -> Crop height
                new_h = int(w / target_ratio)
                x_center = w / 2
                y_center = h / 2
                clip = vfx.crop(clip, x_center=x_center, y_center=y_center, width=w, height=new_h)
                
            # 3. Create subtitles overlay
            title_text = highlight.get("title", "Крутой момент!")
            txt_img_array = self._create_text_overlay(title_text, clip.size, font_size=70)
            
            # Convert NumPy array back to ImageClip
            txt_clip = ImageClip(txt_img_array).set_duration(clip.duration)
            
            # 4. Composite
            final_clip = CompositeVideoClip([clip, txt_clip])
            
            # 5. Write to file
            logger.info(f"Writing final video to {output_path} (this might take a minute)...")
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile="temp-audio.m4a", 
                remove_temp=True, 
                fps=30,
                threads=4,
                logger=None # Disable moviepy progress bar to keep logs clean
            )
            
            # Close clips to free memory
            clip.close()
            txt_clip.close()
            final_clip.close()
            logger.info("Render completed successfully.")
            
        except Exception as e:
            logger.error(f"Render failed: {e}")
            raise
