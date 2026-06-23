import logging
from core.ingestion import VideoIngestor

# Set up logging for the test
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # 18-second video: "Me at the zoo" (the first YouTube video ever)
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
    
    ingestor = VideoIngestor(temp_dir="tmp")
    
    try:
        video_path = ingestor.download_video(test_url)
        audio_path = ingestor.extract_audio(video_path)
        transcript = ingestor.transcribe(audio_path)
        
        print("\n--- Transcription Results ---")
        for segment in transcript:
            print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()
