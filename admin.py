import streamlit as st
import os
import torch
import yaml
import logging
import io
import uuid
from config import settings, CONFIG_FILE
from core.ingestion import VideoIngestor
from core.analyzer import HighlightAnalyzer
from core.renderer import VerticalRenderer

st.set_page_config(page_title="Klippr Admin", page_icon="🎬", layout="wide")

# Setup logging capture
if "log_stream" not in st.session_state:
    st.session_state.log_stream = io.StringIO()
    handler = logging.StreamHandler(st.session_state.log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    # Add to root logger to capture all module logs
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and h.stream == st.session_state.log_stream for h in root_logger.handlers):
        root_logger.addHandler(handler)

logger = logging.getLogger(__name__)

# --- Sidebar ---
st.sidebar.title("Klippr 🎬")
gpu_avail = torch.cuda.is_available()
st.sidebar.info(f"GPU Available: {'✅ Yes' if gpu_avail else '❌ No (CPU)'}")

st.sidebar.warning("⚠️ API ключи хранятся в файле `.env` и не отображаются здесь в целях безопасности. Отредактируйте файл `.env` вручную для изменения ключей.")

if st.sidebar.button("🗑 Сбросить к настройкам по умолчанию", use_container_width=True):
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
    settings._load_yaml() # Reload defaults
    st.rerun()

st.sidebar.divider()

# Import/Export configuration
st.sidebar.subheader("Резервное копирование")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config_yaml_data = f.read()
    
st.sidebar.download_button(
    label="⬇️ Экспорт настроек",
    data=config_yaml_data,
    file_name="config.yaml",
    mime="text/yaml",
    use_container_width=True
)

uploaded_file = st.sidebar.file_uploader("⬆️ Импорт настроек", type=["yaml", "yml"])
if uploaded_file is not None:
    try:
        new_data = yaml.safe_load(uploaded_file)
        settings.save(new_data)
        st.sidebar.success("Настройки успешно импортированы!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Ошибка импорта: {e}")

# --- Main UI ---
st.title("⚙️ Настройки пайплайна")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🤖 AI Модели")
    whisper_model = st.selectbox(
        "Whisper model", 
        ['base', 'small', 'medium', 'large'], 
        index=['base', 'small', 'medium', 'large'].index(settings.WHISPER_MODEL)
    )
    llm_provider = st.selectbox(
        "LLM provider", 
        ['openrouter', 'groq'], 
        index=['openrouter', 'groq'].index(settings.LLM_PROVIDER)
    )
    
    llm_model = st.text_input(
        "LLM model", 
        value=settings.LLM_MODEL,
        help="Examples:\n- OpenRouter: google/gemini-2.5-flash-preview-05-20\n- Groq: llama-3.3-70b-versatile или llama3-70b-8192"
    )
    
    st.subheader("🎞 Обработка видео")
    device = st.selectbox("Device", ['cpu', 'cuda'], index=['cpu', 'cuda'].index(settings.DEVICE))
    if device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            st.warning("Устройство 'cuda' выбрано, но GPU недоступен! Приложение переключится на 'cpu'.")
            
    crop_mode = st.selectbox("Crop mode", ['smart_center', 'face_tracking'], index=['smart_center', 'face_tracking'].index(settings.CROP_MODE))
    output_res = st.selectbox("Output resolution", ['720x1280', '1080x1920'], index=['720x1280', '1080x1920'].index(settings.OUTPUT_RESOLUTION))
    
    st.markdown("##### FFmpeg настройки")
    ffmpeg_preset = st.selectbox("FFmpeg preset", ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'], index=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'].index(settings.FFMPEG_PRESET))
    ffmpeg_crf = st.slider("FFmpeg CRF (качество: меньше - лучше)", min_value=0, max_value=51, value=settings.FFMPEG_CRF)
    use_nvenc = st.checkbox("Использовать NVENC (NVIDIA GPU)", value=settings.USE_NVENC, help="Значительно ускоряет рендер, если есть видеокарта NVIDIA")

with col2:
    st.subheader("📊 Анализ")
    num_clips = st.slider("Number of clips", 1, 10, int(settings.NUM_CLIPS))
    min_dur = st.number_input("Min clip duration (seconds)", value=int(settings.MIN_CLIP_DURATION))
    max_dur = st.number_input("Max clip duration (seconds)", value=int(settings.MAX_CLIP_DURATION))
    
    st.subheader("📝 Субтитры")
    sub_style = st.selectbox("Subtitle style", ['title_only', 'word_by_word'], index=['title_only', 'word_by_word'].index(settings.SUBTITLE_STYLE))
    sub_font = st.slider("Subtitle font size", 30, 110, int(settings.SUBTITLE_FONT_SIZE))
    sub_color = st.color_picker("Subtitle color", settings.SUBTITLE_COLOR if str(settings.SUBTITLE_COLOR).startswith("#") else "#FFFFFF")
    sub_active_color = st.color_picker("Active word color", getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00") if str(getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00")).startswith("#") else "#FFFF00")
    sub_words_per_caption = st.slider(
        "Words per caption",
        1,
        5,
        int(getattr(settings, "SUBTITLE_WORDS_PER_CAPTION", 3)),
        help="Для шортов обычно лучше 2–4. Активное слово подсвечивается по таймингу."
    )
    sub_timing_offset = st.slider(
        "Subtitle timing offset (ms)",
        -300,
        300,
        int(getattr(settings, "SUBTITLE_TIMING_OFFSET_MS", -80)),
        step=10,
        help="Отрицательное значение показывает слово чуть раньше. Для динамичных шортов часто хорошо -80…-120 мс."
    )


def current_ui_config() -> dict:
    """Return the currently visible UI settings.

    Streamlit widgets update local variables immediately, but the processing
    pipeline reads from settings/config. Persisting this before a run prevents
    surprises when the user changes controls and presses Start without pressing Save first.
    """
    return {
        "whisper_model": whisper_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "device": device,
        "crop_mode": crop_mode,
        "output_resolution": output_res,
        "ffmpeg_preset": ffmpeg_preset,
        "ffmpeg_crf": ffmpeg_crf,
        "use_nvenc": use_nvenc,
        "num_clips": int(num_clips),
        "min_clip_duration": int(min_dur),
        "max_clip_duration": int(max_dur),
        "subtitle_style": sub_style,
        "subtitle_font_size": int(sub_font),
        "subtitle_color": sub_color,
        "subtitle_active_color": sub_active_color,
        "subtitle_words_per_caption": int(sub_words_per_caption),
        "subtitle_timing_offset_ms": int(sub_timing_offset),
    }


# Validations
if device == "cuda" and not gpu_avail:
    st.error("⚠️ Устройство 'cuda' выбрано, но GPU недоступен! Приложение переключится на 'cpu'.")
if min_dur > max_dur:
    st.error("❌ Min clip duration не может быть больше Max clip duration!")
if whisper_model == "large" and device == "cpu":
    st.warning("⚠️ Внимание: Выбрана модель 'large' на 'cpu'. Это будет работать очень медленно!")

if st.button("💾 Сохранить настройки", type="primary", use_container_width=True):
    if min_dur > max_dur:
        st.error("Исправьте ошибки перед сохранением.")
    else:
        settings.save(current_ui_config())
        st.success("Настройки успешно сохранены!")

st.divider()

# --- Processing Pipeline ---
st.title("🚀 Запустить обработку")
video_url = st.text_input("URL YouTube видео", placeholder="https://www.youtube.com/watch?v=...")

if st.button("▶️ Начать", type="primary", use_container_width=True):
    if not video_url:
        st.error("Пожалуйста, введите URL видео.")
    elif min_dur > max_dur:
        st.error("Исправьте ошибки в настройках перед запуском.")
    else:
        # Apply the current UI values even if the user did not click Save first.
        run_config = current_ui_config()
        settings.save(run_config)
        logger.info(
            "Starting run with num_clips=%s, duration=%s-%ss, provider=%s, model=%s, device=%s, crop=%s, subtitles=%s, subtitle_offset=%sms, words_per_caption=%s, resolution=%s, nvenc=%s",
            run_config["num_clips"],
            run_config["min_clip_duration"],
            run_config["max_clip_duration"],
            run_config["llm_provider"],
            run_config["llm_model"],
            run_config["device"],
            run_config["crop_mode"],
            run_config["subtitle_style"],
            run_config["subtitle_timing_offset_ms"],
            run_config["subtitle_words_per_caption"],
            run_config["output_resolution"],
            run_config["use_nvenc"],
        )

        # Clear log buffer
        st.session_state.log_stream.truncate(0)
        st.session_state.log_stream.seek(0)
        
        progress_bar = st.progress(0, text="Подготовка...")
        log_expander = st.expander("Журнал логов", expanded=True)
        log_text = log_expander.empty()
        
        job_id = str(uuid.uuid4())[:8]
        temp_dir = f"tmp/{job_id}"
        out_dir = f"output/{job_id}"
        
        def update_logs():
            log_text.text(st.session_state.log_stream.getvalue())

        try:
            # 1. Ingestion (0 -> 40%)
            progress_bar.progress(5, text="[1/4] Инициализация загрузчика...")
            ingestor = VideoIngestor(temp_dir=temp_dir)
            update_logs()
            
            progress_bar.progress(10, text="[1/4] Скачивание видео...")
            video_path = ingestor.download_video(video_url)
            update_logs()
            
            progress_bar.progress(20, text="[2/4] Извлечение аудио...")
            audio_path = ingestor.extract_audio(video_path)
            update_logs()
            
            progress_bar.progress(30, text="[2/4] Транскрибация аудио (может занять время)...")
            transcript = ingestor.transcribe(audio_path)
            update_logs()
            
            # 2. Analysis (40 -> 60%)
            progress_bar.progress(40, text="[3/4] Анализ контента через AI...")
            analyzer = HighlightAnalyzer()
            highlights = analyzer.find_highlights(transcript, num_clips=int(num_clips))
            update_logs()
            
            progress_bar.progress(50, text="[3/4] Точная подгонка по тишине...")
            highlights = analyzer.snap_to_silence(highlights, audio_path, transcript)
            update_logs()
            
            # 3. Rendering (60 -> 100%)
            progress_bar.progress(60, text="[4/4] Инициализация рендера...")
            res = tuple(map(int, settings.OUTPUT_RESOLUTION.split('x')))
            renderer = VerticalRenderer(output_dir=out_dir, resolution=res)
            
            final_clips = []
            total_clips = len(highlights)
            if total_clips == 0:
                st.warning("Не удалось найти подходящие хайлайты. Попробуйте уменьшить Min clip duration или выбрать другую LLM-модель.")
            
            for i, hl in enumerate(highlights):
                prog = 60 + int(40 * (i / max(total_clips, 1)))
                progress_bar.progress(prog, text=f"[4/4] Рендер клипа {i+1} из {total_clips}...")
                
                clip_path = os.path.join(out_dir, f"clip_{i+1}.mp4")
                renderer.render_clip(video_path, hl, clip_path, transcript=transcript)
                final_clips.append(clip_path)
                update_logs()
                
            progress_bar.progress(100, text="✅ Готово!")
            st.success(f"🎉 Обработка завершена! Создано клипов: {len(final_clips)}")
            
            # Display results
            cols = st.columns(min(3, len(final_clips) if final_clips else 1))
            for i, cp in enumerate(final_clips):
                with cols[i % 3]:
                    st.video(cp)
                    with open(cp, "rb") as file:
                        st.download_button(
                            label=f"💾 Скачать Клип {i+1}", 
                            data=file, 
                            file_name=os.path.basename(cp), 
                            mime="video/mp4",
                            use_container_width=True
                        )
                        
        except Exception as e:
            progress_bar.progress(100, text="❌ Ошибка обработки")
            st.error(f"Произошла ошибка: {e}")
            update_logs()
