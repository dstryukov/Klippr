# Klippr 🎬
AI Service for cutting long videos into vertical clips (Reels/Shorts).

## Требования
- Python 3.11+
- Встроенный портативный FFmpeg (скачивается автоматически через `imageio-ffmpeg`)
- Опционально: GPU (Nvidia) для быстрого рендеринга и работы нейросетей локально.

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
Для работы LLM, системе нужны API ключи (от OpenRouter или Groq). 
Создайте файл `.env` в корне проекта (вы можете скопировать `.env.example`):
```env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

*Все остальные настройки (выбор модели, параметры кропа и длительности) сохраняются в файл `config.yaml` автоматически через админ-панель.*

## Запуск

### Способ 1: Интерфейс администратора (Рекомендуемый)
Полноценный дашборд для визуальной настройки моделей, субтитров, отслеживания прогресса и скачивания клипов.
```bash
streamlit run admin.py --server.port 8501
```

### Способ 2: Запуск API (Бэкенд)
Полноценный REST API на базе FastAPI. Идеально подходит для интеграции с ботами (например, Telegram).
```bash
uvicorn main:app --reload --port 8000
```
Swagger UI будет доступен по адресу: http://127.0.0.1:8000/docs
