import os
import logging
from core.renderer import VerticalRenderer
from core.ingestion import VideoIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # 1. Download short video first (18s "Me at the zoo")
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    ingestor = VideoIngestor(temp_dir="tmp")
    
    try:
        video_path = ingestor.download_video(test_url)
        
        # 2. Define a fake highlight (we don't call LLM here to save time/tokens)
        dummy_highlight = {
            "start_time": 2.0,
            "end_time": 7.0,
            "title": "Смотрите, какие у них классные длинные хоботы!",
            "reason": "Funny and iconic first youtube video moment."
        }
        
        # 3. Render
        renderer = VerticalRenderer(output_dir="output")
        output_path = os.path.join("output", "test_clip.mp4")
        
        renderer.render_clip(video_path, dummy_highlight, output_path)
        
        print(f"\nУспешно! Видео сохранено в: {output_path}")
        print("Посмотрите его, чтобы оценить кроп и субтитры.")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
