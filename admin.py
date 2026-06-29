import io
import logging
import os
from pathlib import Path

import streamlit as st
import torch
import yaml

from config import CONFIG_FILE, settings
from core.analyzer import HighlightAnalyzer
from core.ingestion import VideoIngestor
from core.projects import (
    add_clip,
    clips_dir,
    create_project,
    list_projects,
    load_candidates,
    load_project,
    load_transcript,
    save_candidates,
    save_project,
    save_transcript,
    tmp_dir,
)
from core.renderer import VerticalRenderer

st.set_page_config(page_title="Klippr Studio", page_icon="🎬", layout="wide")

# --- Logging ---
if "log_stream" not in st.session_state:
    st.session_state.log_stream = io.StringIO()
    handler = logging.StreamHandler(st.session_state.log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and h.stream == st.session_state.log_stream for h in root_logger.handlers):
        root_logger.addHandler(handler)

logger = logging.getLogger(__name__)

if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None


def safe_rerun() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def reset_logs() -> None:
    st.session_state.log_stream.truncate(0)
    st.session_state.log_stream.seek(0)


def log_text_value() -> str:
    return st.session_state.log_stream.getvalue()


def safe_cuda_label() -> str:
    try:
        if torch.cuda.is_available():
            return "✅ " + torch.cuda.get_device_name(0)
    except Exception:
        return "⚠️ CUDA check failed"
    return "❌ CPU only"


def safe_index(options: list[str], value: object, default: int = 0) -> int:
    try:
        return options.index(str(value))
    except Exception:
        return default


def safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def format_time(seconds: float) -> str:
    try:
        seconds = max(0.0, float(seconds or 0))
    except Exception:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def candidate_score(clip: dict) -> float:
    try:
        return float(clip.get("total_score", clip.get("score", 0)))
    except Exception:
        return 0.0


def candidate_label(idx: int, clip: dict) -> str:
    score = candidate_score(clip)
    start = format_time(clip.get("start_time", 0))
    end = format_time(clip.get("end_time", 0))
    title = str(clip.get("title") or "Клип")[:80]
    return f"{idx + 1}. {start}–{end} | {score:.0f} | {title}"


def project_option_label(project: dict) -> str:
    status = project.get("status", "created")
    name = project.get("name", "Untitled")
    updated = str(project.get("updated_at", ""))[:16].replace("T", " ")
    return f"{name} · {status} · {updated}"


def load_active_project() -> dict | None:
    project_id = st.session_state.get("active_project_id")
    if not project_id:
        return None
    try:
        return load_project(project_id)
    except Exception as e:
        st.warning(f"Не удалось открыть проект {project_id}: {e}")
        st.session_state.active_project_id = None
        return None


def show_project_metrics(project: dict, candidates: list[dict]) -> None:
    clips = project.get("clips", []) if isinstance(project.get("clips", []), list) else []
    selected = project.get("selected_candidate_ids", []) if isinstance(project.get("selected_candidate_ids", []), list) else []
    cols = st.columns(5)
    cols[0].metric("Status", project.get("status", "created"))
    cols[1].metric("Candidates", len(candidates))
    cols[2].metric("Selected", len(selected))
    cols[3].metric("Rendered", len(clips))
    cols[4].metric("Updated", str(project.get("updated_at", ""))[:10])


def read_settings_from_sidebar() -> dict:
    whisper_options = ["base", "small", "medium", "large"]
    provider_options = ["openrouter", "groq"]
    device_options = ["cpu", "cuda"]
    crop_options = ["smart_center", "face_tracking"]
    resolution_options = ["720x1280", "1080x1920"]
    preset_options = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"]
    subtitle_options = ["title_only", "word_by_word"]

    with st.sidebar.expander("⚙️ Pipeline settings", expanded=False):
        whisper_model = st.selectbox(
            "Whisper model",
            whisper_options,
            index=safe_index(whisper_options, getattr(settings, "WHISPER_MODEL", "small"), 1),
            key="settings_whisper_model",
        )
        llm_provider = st.selectbox(
            "LLM provider",
            provider_options,
            index=safe_index(provider_options, getattr(settings, "LLM_PROVIDER", "groq"), 1),
            key="settings_llm_provider",
        )
        llm_model = st.text_input(
            "LLM model",
            value=str(getattr(settings, "LLM_MODEL", "llama-3.3-70b-versatile")),
            key="settings_llm_model",
            help="Для качества попробуйте groq / llama-3.3-70b-versatile. Для скорости — llama-3.1-8b-instant.",
        )
        device = st.selectbox(
            "Device",
            device_options,
            index=safe_index(device_options, getattr(settings, "DEVICE", "cuda"), 1),
            key="settings_device",
        )
        crop_mode = st.selectbox(
            "Crop mode",
            crop_options,
            index=safe_index(crop_options, getattr(settings, "CROP_MODE", "smart_center"), 0),
            key="settings_crop_mode",
        )
        output_res = st.selectbox(
            "Output resolution",
            resolution_options,
            index=safe_index(resolution_options, getattr(settings, "OUTPUT_RESOLUTION", "1080x1920"), 1),
            key="settings_output_res",
        )
        ffmpeg_preset = st.selectbox(
            "FFmpeg preset",
            preset_options,
            index=safe_index(preset_options, getattr(settings, "FFMPEG_PRESET", "fast"), 4),
            key="settings_ffmpeg_preset",
        )
        ffmpeg_crf = st.slider("FFmpeg CRF", 0, 51, safe_int(getattr(settings, "FFMPEG_CRF", 23), 23), key="settings_ffmpeg_crf")
        use_nvenc = st.checkbox("Use NVENC", value=bool(getattr(settings, "USE_NVENC", False)), key="settings_use_nvenc")
        min_dur = st.number_input("Min clip duration", value=safe_int(getattr(settings, "MIN_CLIP_DURATION", 20), 20), key="settings_min_dur")
        max_dur = st.number_input("Max clip duration", value=safe_int(getattr(settings, "MAX_CLIP_DURATION", 60), 60), key="settings_max_dur")
        sub_style = st.selectbox(
            "Subtitle style",
            subtitle_options,
            index=safe_index(subtitle_options, getattr(settings, "SUBTITLE_STYLE", "word_by_word"), 1),
            key="settings_sub_style",
        )
        sub_font = st.slider("Subtitle font size", 30, 110, safe_int(getattr(settings, "SUBTITLE_FONT_SIZE", 70), 70), key="settings_sub_font")
        raw_sub_color = str(getattr(settings, "SUBTITLE_COLOR", "#FFFFFF"))
        raw_active_color = str(getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00"))
        sub_color = st.color_picker("Subtitle color", raw_sub_color if raw_sub_color.startswith("#") else "#FFFFFF", key="settings_sub_color")
        sub_active_color = st.color_picker("Active word color", raw_active_color if raw_active_color.startswith("#") else "#FFFF00", key="settings_sub_active_color")
        sub_words_per_caption = st.slider(
            "Words per caption",
            1,
            5,
            safe_int(getattr(settings, "SUBTITLE_WORDS_PER_CAPTION", 3), 3),
            key="settings_sub_words_per_caption",
        )
        sub_timing_offset = st.slider(
            "Subtitle timing offset (ms)",
            -300,
            300,
            safe_int(getattr(settings, "SUBTITLE_TIMING_OFFSET_MS", -80), -80),
            step=10,
            key="settings_sub_timing_offset",
        )

        config = {
            "whisper_model": whisper_model,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
            "device": device,
            "crop_mode": crop_mode,
            "output_resolution": output_res,
            "ffmpeg_preset": ffmpeg_preset,
            "ffmpeg_crf": int(ffmpeg_crf),
            "use_nvenc": bool(use_nvenc),
            "min_clip_duration": int(min_dur),
            "max_clip_duration": int(max_dur),
            "subtitle_style": sub_style,
            "subtitle_font_size": int(sub_font),
            "subtitle_color": sub_color,
            "subtitle_active_color": sub_active_color,
            "subtitle_words_per_caption": int(sub_words_per_caption),
            "subtitle_timing_offset_ms": int(sub_timing_offset),
        }
        if st.button("Save settings", use_container_width=True, key="settings_save_btn"):
            if int(min_dur) > int(max_dur):
                st.error("Min duration cannot be greater than max duration.")
            else:
                settings.save(config)
                st.success("Settings saved")
        return config


# --- Sidebar ---
st.sidebar.title("Klippr Studio")
st.sidebar.caption(f"GPU: {safe_cuda_label()}")

if st.sidebar.button("🔄 Reset UI state", use_container_width=True, key="reset_ui_state_btn"):
    st.session_state.active_project_id = None
    reset_logs()
    safe_rerun()

projects = list_projects()
project_ids = [p.get("id") for p in projects if p.get("id")]
if projects and project_ids:
    default_idx = project_ids.index(st.session_state.active_project_id) if st.session_state.active_project_id in project_ids else 0
    selected_project_id = st.sidebar.selectbox(
        "Projects",
        project_ids,
        index=default_idx,
        key="sidebar_project_select",
        format_func=lambda pid: project_option_label(next((p for p in projects if p.get("id") == pid), {"name": pid})),
    )
    if selected_project_id != st.session_state.active_project_id:
        st.session_state.active_project_id = selected_project_id
        safe_rerun()
else:
    st.sidebar.info("Пока нет проектов. Создайте первый проект ниже.")

with st.sidebar.expander("➕ New project", expanded=not bool(projects)):
    new_project_name = st.text_input("New project name", value="New shorts project", key="new_project_name_input")
    new_project_url = st.text_input("New project YouTube URL", placeholder="https://www.youtube.com/watch?v=...", key="new_project_url_input")
    if st.button("Create project", use_container_width=True, key="create_project_btn"):
        project = create_project(new_project_name, new_project_url)
        st.session_state.active_project_id = project["id"]
        st.success("Project created")
        safe_rerun()

run_config = read_settings_from_sidebar()

with st.sidebar.expander("Backup", expanded=False):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_yaml_data = f.read()
        st.download_button("Export config.yaml", config_yaml_data, file_name="config.yaml", mime="text/yaml", use_container_width=True, key="export_config_btn")
    uploaded_file = st.file_uploader("Import config", type=["yaml", "yml"], key="import_config_uploader")
    if uploaded_file is not None:
        try:
            settings.save(yaml.safe_load(uploaded_file) or {})
            st.success("Settings imported")
            safe_rerun()
        except Exception as e:
            st.error(f"Import error: {e}")

project = load_active_project()

# --- Main Studio ---
st.title("🎬 Klippr Studio")
st.caption("Safe-mode Studio: Source → Analyze → Review → Render → Export")

if project is None:
    st.info("Создайте проект в сайдбаре, чтобы начать.")
    st.stop()

try:
    candidates = load_candidates(project["id"])
except Exception as e:
    st.warning(f"Не удалось загрузить candidates.json: {e}")
    candidates = []
try:
    transcript = load_transcript(project["id"])
except Exception as e:
    st.warning(f"Не удалось загрузить transcript.json: {e}")
    transcript = []

show_project_metrics(project, candidates)

source_tab, analyze_tab, review_tab, render_tab, export_tab, logs_tab = st.tabs([
    "1 Source",
    "2 Analyze",
    "3 Review",
    "4 Render",
    "5 Export",
    "Logs",
])

with source_tab:
    st.subheader("Source video")
    name = st.text_input("Project title", value=project.get("name", "Untitled project"), key=f"source_name_{project['id']}")
    source_url = st.text_input("Source YouTube URL", value=project.get("source_url", ""), placeholder="https://www.youtube.com/watch?v=...", key=f"source_url_{project['id']}")
    notes = st.text_area("Project notes", value=project.get("notes", ""), height=100, key=f"source_notes_{project['id']}")
    if st.button("Save source", type="primary", key=f"save_source_{project['id']}"):
        project["name"] = name.strip() or "Untitled project"
        project["source_url"] = source_url.strip()
        project["notes"] = notes
        project["settings_snapshot"] = run_config
        save_project(project)
        st.success("Source saved")
        safe_rerun()

    video_path = project.get("video_path")
    if video_path and Path(video_path).exists():
        st.success("Source video is downloaded.")
        st.video(video_path)
    else:
        st.info("Видео ещё не скачано. Перейдите во вкладку Analyze.")

with analyze_tab:
    st.subheader("Analyze video")
    st.write("Скачивание, аудио, транскрипт и AI-кандидаты сохраняются в папку проекта.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Transcript segments", len(transcript))
    col_b.metric("Candidates", len(candidates))
    col_c.metric("Model", run_config["llm_model"])

    if st.button("🔎 Analyze / regenerate candidates", type="primary", use_container_width=True, key=f"analyze_btn_{project['id']}"):
        if not project.get("source_url"):
            st.error("Сначала укажите YouTube URL во вкладке Source.")
        elif run_config["min_clip_duration"] > run_config["max_clip_duration"]:
            st.error("Min duration cannot be greater than max duration.")
        else:
            reset_logs()
            settings.save(run_config)
            project["settings_snapshot"] = run_config
            project["status"] = "analyzing"
            save_project(project)

            progress_bar = st.progress(0, text="Starting analysis...")
            log_box = st.empty()

            def update_logs():
                log_box.text(log_text_value())

            try:
                progress_bar.progress(5, text="Initializing ingestion...")
                ingestor = VideoIngestor(temp_dir=str(tmp_dir(project["id"])))
                update_logs()

                progress_bar.progress(15, text="Downloading video...")
                video_path = ingestor.download_video(project["source_url"])
                project["video_path"] = video_path
                save_project(project)
                update_logs()

                progress_bar.progress(30, text="Extracting audio...")
                audio_path = ingestor.extract_audio(video_path)
                project["audio_path"] = audio_path
                save_project(project)
                update_logs()

                progress_bar.progress(45, text="Transcribing with word timestamps...")
                transcript = ingestor.transcribe(audio_path)
                project["transcript_path"] = save_transcript(project["id"], transcript)
                save_project(project)
                update_logs()

                progress_bar.progress(65, text="Finding AI candidates...")
                analyzer = HighlightAnalyzer()
                candidates = analyzer.find_highlight_candidates(transcript)
                candidates = analyzer.snap_to_silence(candidates, audio_path, transcript)
                project["candidates_path"] = save_candidates(project["id"], candidates)
                project["selected_candidate_ids"] = [c.get("id") for c in candidates if c.get("id")]
                project["status"] = "candidates_ready"
                save_project(project)
                update_logs()

                progress_bar.progress(100, text="Candidates ready")
                st.success(f"Готово: найдено кандидатов {len(candidates)}")
                safe_rerun()
            except Exception as e:
                project["status"] = "analysis_error"
                save_project(project)
                progress_bar.progress(100, text="Analysis failed")
                st.error(f"Ошибка анализа: {e}")
                update_logs()

with review_tab:
    st.subheader("Review candidates")
    if not candidates:
        st.info("Кандидатов пока нет. Запустите Analyze.")
    else:
        table_rows = []
        for i, clip in enumerate(candidates):
            try:
                duration = round(float(clip.get("end_time", 0)) - float(clip.get("start_time", 0)), 1)
            except Exception:
                duration = 0.0
            table_rows.append({
                "№": i + 1,
                "time": f"{format_time(clip.get('start_time', 0))}–{format_time(clip.get('end_time', 0))}",
                "duration": clip.get("duration", duration),
                "score": candidate_score(clip),
                "hook": clip.get("hook", ""),
                "title": clip.get("title", ""),
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        for i, clip in enumerate(candidates):
            with st.expander(candidate_label(i, clip)):
                st.markdown(f"**Hook:** {clip.get('hook', '')}")
                st.markdown(f"**Reason:** {clip.get('reason', '')}")
                st.markdown(
                    f"**Scores:** hook={clip.get('hook_score', '-')}, standalone={clip.get('standalone_score', '-')}, "
                    f"payoff={clip.get('payoff_score', '-')}, retention={clip.get('retention_score', '-')}, "
                    f"clarity={clip.get('clarity_score', '-')}, total={candidate_score(clip):.0f}"
                )
                st.write(clip.get("text", ""))

        candidate_ids = [str(c.get("id", f"candidate_{i + 1}")) for i, c in enumerate(candidates)]
        project_selected = [str(cid) for cid in project.get("selected_candidate_ids", []) if str(cid) in candidate_ids]
        if not project_selected:
            project_selected = candidate_ids

        selected_ids = st.multiselect(
            "Selected candidates",
            options=candidate_ids,
            default=project_selected,
            key=f"selected_candidates_{project['id']}",
            format_func=lambda cid: candidate_label(candidate_ids.index(cid), candidates[candidate_ids.index(cid)]) if cid in candidate_ids else cid,
        )
        if st.button("Save selection", type="primary", key=f"save_selection_{project['id']}"):
            project["selected_candidate_ids"] = selected_ids
            project["status"] = "selection_ready" if selected_ids else project.get("status", "candidates_ready")
            save_project(project)
            st.success("Selection saved")
            safe_rerun()

with render_tab:
    st.subheader("Render selected clips")
    if not candidates:
        st.info("Сначала запустите Analyze.")
    else:
        selected_ids = set(str(cid) for cid in project.get("selected_candidate_ids", []))
        selected = [c for c in candidates if str(c.get("id")) in selected_ids]
        if not selected:
            st.warning("Сначала выберите кандидаты во вкладке Review.")
        else:
            st.write(f"Selected clips: {len(selected)}")
            for i, clip in enumerate(selected):
                st.caption(candidate_label(i, clip))

            if st.button("🎬 Render selected", type="primary", use_container_width=True, key=f"render_btn_{project['id']}"):
                if not project.get("video_path") or not Path(project["video_path"]).exists():
                    st.error("Source video not found. Run Analyze again.")
                else:
                    reset_logs()
                    settings.save(run_config)
                    project["status"] = "rendering"
                    save_project(project)
                    progress_bar = st.progress(0, text="Initializing render...")
                    log_box = st.empty()

                    def update_logs():
                        log_box.text(log_text_value())

                    try:
                        res = tuple(map(int, run_config["output_resolution"].split("x")))
                        renderer = VerticalRenderer(output_dir=str(clips_dir(project["id"])), resolution=res)
                        transcript = load_transcript(project["id"])
                        project["clips"] = []
                        selected_sorted = sorted(selected, key=lambda c: float(c.get("start_time", 0)))
                        for i, hl in enumerate(selected_sorted):
                            progress_bar.progress(int(100 * (i / max(len(selected_sorted), 1))), text=f"Rendering clip {i + 1}/{len(selected_sorted)}...")
                            clip_path = str(clips_dir(project["id"]) / f"clip_{i + 1:02d}.mp4")
                            renderer.render_clip(project["video_path"], hl, clip_path, transcript=transcript)
                            project = add_clip(project, clip_path, hl)
                            update_logs()

                        project["status"] = "rendered"
                        save_project(project)
                        progress_bar.progress(100, text="Rendered")
                        st.success(f"Готово: создано клипов {len(project.get('clips', []))}")
                        safe_rerun()
                    except Exception as e:
                        project["status"] = "render_error"
                        save_project(project)
                        progress_bar.progress(100, text="Render failed")
                        st.error(f"Ошибка рендера: {e}")
                        update_logs()

with export_tab:
    st.subheader("Export")
    clips = project.get("clips", []) if isinstance(project.get("clips", []), list) else []
    if not clips:
        st.info("Клипов пока нет. Перейдите во вкладку Render.")
    else:
        cols = st.columns(min(3, max(1, len(clips))))
        for i, clip in enumerate(clips):
            path = clip.get("path")
            with cols[i % len(cols)]:
                st.markdown(f"**{clip.get('title', f'Clip {i + 1}')}**")
                st.caption(f"{format_time(clip.get('start_time', 0))}–{format_time(clip.get('end_time', 0))}")
                if path and Path(path).exists():
                    st.video(path)
                    with open(path, "rb") as file:
                        st.download_button(
                            label=f"Download clip {i + 1}",
                            data=file,
                            file_name=os.path.basename(path),
                            mime="video/mp4",
                            use_container_width=True,
                            key=f"download_clip_{project['id']}_{i}",
                        )
                else:
                    st.warning("File missing")

with logs_tab:
    st.subheader("Logs")
    st.text_area("Current session logs", value=log_text_value(), height=500, key=f"logs_area_{project['id']}")
    if st.button("Clear logs", key=f"clear_logs_{project['id']}"):
        reset_logs()
        safe_rerun()
