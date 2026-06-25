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

## Установка на Windows с NVIDIA CUDA
PyTorch CUDA ставится отдельной командой из официального PyTorch wheel index. Если поставить обычный `requirements.txt` первым, pip может подтянуть CPU-сборку PyTorch через зависимости вроде `ultralytics`.

```bash
python -m venv .venv
.\.venv\Scripts\activate

python -m pip install --upgrade pip

# 1) Сначала CUDA PyTorch
pip install -r requirements-gpu-cu126.txt

# 2) Потом остальные зависимости проекта
pip install -r requirements.txt
```

Проверка CUDA:
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Официальный PyTorch installer рекомендует выбирать Windows + Pip + нужную CUDA-версию и проверять `torch.cuda.is_available()` после установки.

## Установка без GPU / CPU-only
```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Конфигурация API
Для работы LLM системе нужны API ключи от OpenRouter или Groq.
Создайте файл `.env` в корне проекта:
```env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Остальные настройки лежат в `config.yaml` и редактируются в левой панели интерфейса.

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
2. В левой панели проверьте CUDA/System и настройки пайплайна.
3. Укажите YouTube URL.
4. Нажмите `Analyze / regenerate candidates`.
5. Дождитесь завершения background job.
6. Во вкладке Review выберите кандидаты.
7. Нажмите `Render selected`.
8. Во вкладке Export скачайте готовые клипы.

## API endpoints
- `GET /api/system` — PyTorch/CUDA/GPU status.
- `GET /api/settings` — текущие настройки пайплайна.
- `PATCH /api/settings` — сохранить настройки пайплайна.
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
