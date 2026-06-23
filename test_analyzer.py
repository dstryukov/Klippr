import logging
from core.analyzer import HighlightAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    # 10 minute transcript (600 seconds)
    # We will generate synthetic segments of ~5 seconds each
    dummy_transcript = []
    current_time = 0.0
    for i in range(120): # 120 * 5 = 600s
        dummy_transcript.append({
            "start": current_time,
            "end": current_time + 4.8,
            "text": f"Фраза номер {i}. Очень важный смысл и инсайт."
        })
        current_time += 5.0

    analyzer = HighlightAnalyzer()
    
    try:
        highlights = analyzer.find_highlights(dummy_transcript, num_clips=5)
        
        print("\n" + "="*40)
        print("--- Найденные хайлайты ---")
        print("="*40)
        
        valid_clips = 0
        for i, clip in enumerate(highlights, 1):
            duration = clip['end_time'] - clip['start_time']
            print(f"\nКлип {i}: {clip.get('title', 'Без названия')}")
            print(f"Таймкод: {clip.get('start_time')}s - {clip.get('end_time')}s (Длина: {duration:.2f} сек)")
            print(f"Причина: {clip.get('reason')}")
            
            # Assert to show correctness
            if duration >= 30.0 and duration <= 90.0:
                valid_clips += 1
                
        print(f"\nТест завершен. Успешно валидных клипов (30-90 сек): {valid_clips} из {len(highlights)}.")
            
    except Exception as e:
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
