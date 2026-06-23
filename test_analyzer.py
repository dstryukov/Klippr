import logging
from core.analyzer import HighlightAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    dummy_transcript = [
        {"start": 0.0, "end": 2.5, "text": "Всем привет! Сегодня мы поговорим о том,"},
        {"start": 2.5, "end": 5.0, "text": "как нейросети навсегда изменят нашу работу."},
        {"start": 5.0, "end": 8.5, "text": "Многие думают, что AI просто заберет у них работу."},
        {"start": 8.5, "end": 12.0, "text": "Но на самом деле, это ваш самый мощный инструмент."},
        {"start": 12.0, "end": 15.5, "text": "Представьте, что вы можете делать за час то, на что раньше уходил день!"},
        {"start": 15.5, "end": 20.0, "text": "Именно поэтому те, кто освоит AI сегодня, станут лидерами завтра."},
        {"start": 20.0, "end": 25.0, "text": "Так что не бойтесь перемен, используйте их. Подписывайтесь на канал!"}
    ]

    analyzer = HighlightAnalyzer()
    
    try:
        # We request 2 clips for testing
        highlights = analyzer.find_highlights(dummy_transcript, num_clips=2)
        
        print("\n" + "="*40)
        print("--- Найденные хайлайты ---")
        print("="*40)
        for i, clip in enumerate(highlights, 1):
            print(f"\nКлип {i}: {clip.get('title', 'Без названия')}")
            print(f"Таймкод: {clip.get('start_time')}s - {clip.get('end_time')}s")
            print(f"Причина: {clip.get('reason')}")
    except Exception as e:
        logging.error(f"Test failed: {e}")

if __name__ == "__main__":
    main()
