# Klippr 🎬
AI service for cutting long videos into vertical clips (Reels/Shorts).

## Что умеет прототип
- Скачивает видео по URL через `yt-dlp`.
- Извлекает аудио через FFmpeg.
- Транскрибирует через `faster-whisper` с word-level timestamps.
- Ищет кандидаты на хайлайты через OpenRouter или Groq.
- Показывает кандидатов в web-интерфейсе: score, hook, reason, текст и таймкоды.
- Рендерит выбранные вертикальные 9:16 клипы с корректным crop/scale.
- Делает субтитры в стиле CapCut: активное слово подсвечивается по таймингу.
- Использует GPU для Whisper/YOLO при `device: cuda` и пробует NVENC для FFmpeg-рендера; если NVENC недоступен, автоматически откатывается на `libx264`.

## Почему больше не Streamlit
Долгие операции вроде транскрибации и FFmpeg-рендера теперь запускаются как background jobs в FastAPI. Браузер может обновиться или временно потерять соединение — анализ/рендер продолжит выполняться на сервере, а UI просто переподключится и продолжит polling job status.

## Требования
- Python 3.11+
- Встроенный портативный FFmpeg (скачивается автоматически через `imageio-ffmpeg`)
- Опционально: GPU Nvidia для быстрого рендеринга и работы нейросетей локально.

## Установка
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Конфигурация API
Для работы LLM системе нужны API ключи от OpenRouter или Groq.
Создайте файл `.env` в корне проекта:
```env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Остальные настройки лежат в `config.yaml`.

## Запуск Studio UI + API
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Откройте:
```text
http://127.0.0.1:8000
```

Swagger UI:
```text
http://127.0.0.1:8000/docs
```

## Новый workflow
1. Создайте проект в сайдбаре.
2. Укажите YouTube URL.
3. Нажмите `Analyze / regenerate candidates`.
4. Дождитесь завершения background job.
5. Во вкладке Review выберите кандидаты.
6. Нажмите `Render selected`.
7. Во вкладке Export скачайте готовые клипы.

## API endpoints
- `GET /api/projects` — список проектов.
- `POST /api/projects` — создать проект.
- `GET /api/projects/{project_id}` — проект, candidates и transcript count.
- `PATCH /api/projects/{project_id}` — обновить name/source_url/notes.
- `POST /api/projects/{project_id}/analyze` — запустить анализ в фоне.
- `POST /api/projects/{project_id}/render` — запустить рендер выбранных candidates в фоне.
- `GET /api/jobs/{job_id}` — статус background job.
- `GET /api/projects/{project_id}/clips/{filename}` — скачать mp4.

## Локальные данные
Проекты и результаты сохраняются в:
```text
data/projects/<project_id>/
```

Эта папка игнорируется git.
