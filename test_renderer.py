import sys
import os
import time

# Create dummy video if not exists
video_path = "test_video.mp4"
if not os.path.exists(video_path):
    print("Generating a 10s test video...")
    os.system(f"ffmpeg -f lavfi -i testsrc=duration=10:size=1920x1080:rate=30 -f lavfi -i sine=frequency=1000:duration=10 -c:v libx264 -c:a aac {video_path}")

from core.renderer import VerticalRenderer
from config import settings

settings.FFMPEG_PRESET = "ultrafast"
settings.USE_NVENC = False
settings.CROP_MODE = "face_tracking"

renderer = VerticalRenderer(output_dir="output", resolution=(1080, 1920))

highlight = {
    "start_time": 2.0,
    "end_time": 7.0,
    "title": "FFmpeg Test!"
}

transcript = [
    {"start": 2.5, "end": 3.0, "text": "This"},
    {"start": 3.0, "end": 3.5, "text": "is"},
    {"start": 3.5, "end": 4.5, "text": "FFmpeg!"}
]

print("Starting render...")
start_time = time.time()
try:
    renderer.render_clip(video_path, highlight, "output/test_output.mp4", transcript)
    print(f"Success! Render took {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error: {e}")
