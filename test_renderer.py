import os
import logging
from moviepy.editor import VideoFileClip
from core.renderer import VerticalRenderer
from core.ingestion import VideoIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    ingestor = VideoIngestor(temp_dir="tmp")
    
    try:
        video_path = ingestor.download_video(test_url)
        
        dummy_highlight = {
            "start_time": 0.0,
            "end_time": 5.0,
            "title": "Тест кропа 1080x1920",
            "reason": "Verify cropping pipeline and resolution"
        }
        
        # Test the renderer with smart_center (or face_tracking if changed in config)
        renderer = VerticalRenderer(output_dir="output", resolution=(1080, 1920))
        output_path = os.path.join("output", "test_clip_1080p.mp4")
        
        renderer.render_clip(video_path, dummy_highlight, output_path)
        
        # Verify resolution
        final = VideoFileClip(output_path)
        print(f"\nИтоговое разрешение: {final.size[0]}x{final.size[1]}")
        
        assert final.size[0] == 1080, f"Width is {final.size[0]}, expected 1080!"
        assert final.size[1] == 1920, f"Height is {final.size[1]}, expected 1920!"
        
        final.close()
        
        print(f"Успешно! Видео сохранено в: {output_path}")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
