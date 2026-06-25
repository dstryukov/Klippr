# Klippr 🎬
AI Service for cutting long videos into vertical clips (Reels/Shorts).

## Что умеет прототип
- Скачивает видео по URL через `yt-dlp`.
- Извлекает аудио через FFmpeg.
- Транскрибирует через `faster-whisper` с word-level timestamps.
- Ищет кандидаты на хайлайты через OpenRouter или Groq.
- Рендерит вертикальные 9:16 клипы с корректным crop/scale.
- Делает субтитры в стиле CapCut: слово появляется в момент произношения, а не вся строка сразу.
- Использует GPU для Whisper/YOLO при `device: cuda` и пробует NVENC для FFmpeg-рендера; если NVENC недоступен, автоматически откатывается на `libx264`.

## Требования
- Python 3.11+
- Встроенный портативный FFmpeg (скачивается автоматически через `imageio-ffmpeg`)
- Опционально: GPU Nvidia для быстрого рендеринга и работы нейросетей локально.

## Установка
```bash
python -m venv .venv

# Активация для Windows:
.\.venv\Scripts\activate

# Активация для Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Конфигурация API
Для работы LLM системе нужны API ключи от OpenRouter или Groq.
Создайте файл `.env` в корне проекта (вы можете скопировать `.env.example`):
```env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Все остальные настройки — выбор модели, параметры кропа, длительность клипов, стиль субтитров и рендер — сохраняются в `config.yaml` автоматически через админ-панель.

## Запуск

### Способ 1: Интерфейс администратора (рекомендуемый)
Полноценный дашборд для визуальной настройки моделей, субтитров, отслеживания прогресса и скачивания клипов.
```bash
streamlit run admin.py --server.port 8501
```

Для CapCut-подобных субтитров выберите в админке `Subtitle style = word_by_word`.

### Способ 2: Запуск API (бэкенд)
REST API на базе FastAPI. Подходит для интеграции с ботами, например Telegram.
```bash
uvicorn main:app --reload --port 8000
```
Swagger UI будет доступен по адресу: http://127.0.0.1:8000/docs
