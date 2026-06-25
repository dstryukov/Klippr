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


def reset_logs() -> None:
    st.session_state.log_stream.truncate(0)
    st.session_state.log_stream.seek(0)


def log_text_value() -> str:
    return st.session_state.log_stream.getvalue()


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def candidate_score(clip: dict) -> float:
    try:
        return float(clip.get("total_score", clip.get("score", 0)))
    except (TypeError, ValueError):
        return 0.0


def candidate_label(idx: int, clip: dict) -> str:
    score = candidate_score(clip)
    start = format_time(clip.get("start_time", 0))
    end = format_time(clip.get("end_time", 0))
    title = clip.get("title") or "Клип"
    return f"{idx + 1}. {start}–{end} | {score:.0f} | {title}"


def current_ui_config() -> dict:
    return {
        "whisper_model": whisper_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "device": device,
        "crop_mode": crop_mode,
        "output_resolution": output_res,
        "ffmpeg_preset": ffmpeg_preset,
        "ffmpeg_crf": int(ffmpeg_crf),
        "use_nvenc": bool(use_nvenc),
        "num_clips": int(num_clips),
        "highlight_candidate_count": int(candidate_count),
        "min_clip_duration": int(min_dur),
        "max_clip_duration": int(max_dur),
        "subtitle_style": sub_style,
        "subtitle_font_size": int(sub_font),
        "subtitle_color": sub_color,
        "subtitle_active_color": sub_active_color,
        "subtitle_words_per_caption": int(sub_words_per_caption),
        "subtitle_timing_offset_ms": int(sub_timing_offset),
    }


def project_option_label(project: dict) -> str:
    status = project.get("status", "created")
    name = project.get("name", "Untitled")
    updated = project.get("updated_at", "")[:16].replace("T", " ")
    return f"{name} · {status} · {updated}"


def load_active_project() -> dict | None:
    if not st.session_state.active_project_id:
        return None
    try:
        return load_project(st.session_state.active_project_id)
    except FileNotFoundError:
        st.session_state.active_project_id = None
        return None


def show_project_metrics(project: dict, candidates: list[dict]) -> None:
    clips = project.get("clips", [])
    cols = st.columns(5)
    cols[0].metric("Status", project.get("status", "created"))
    cols[1].metric("Candidates", len(candidates))
    cols[2].metric("Selected", len(project.get("selected_candidate_ids", [])))
    cols[3].metric("Rendered", len(clips))
    cols[4].metric("Updated", project.get("updated_at", "")[:10])


# --- Sidebar / Settings ---
st.sidebar.title("Klippr Studio")
gpu_avail = torch.cuda.is_available()
st.sidebar.caption(f"GPU: {'✅ ' + torch.cuda.get_device_name(0) if gpu_avail else '❌ CPU only'}")

projects = list_projects()
project_ids = [p["id"] for p in projects]
if projects:
    default_idx = project_ids.index(st.session_state.active_project_id) if st.session_state.active_project_id in project_ids else 0
    selected_project_id = st.sidebar.selectbox(
        "Projects",
        project_ids,
        index=default_idx,
        format_func=lambda pid: project_option_label(next((p for p in projects if p["id"] == pid), {"name": pid})),
    )
    if selected_project_id != st.session_state.active_project_id:
        st.session_state.active_project_id = selected_project_id
        st.rerun()
else:
    st.sidebar.info("Пока нет проектов. Создайте первый проект ниже.")

with st.sidebar.expander("➕ New project", expanded=not bool(projects)):
    new_project_name = st.text_input("Project name", value="New shorts project")
    new_project_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("Create project", use_container_width=True):
        project = create_project(new_project_name, new_project_url)
        st.session_state.active_project_id = project["id"]
        st.success("Project created")
        st.rerun()

with st.sidebar.expander("⚙️ Pipeline settings", expanded=False):
    whisper_model = st.selectbox(
        "Whisper model",
        ["base", "small", "medium", "large"],
        index=["base", "small", "medium", "large"].index(settings.WHISPER_MODEL),
    )
    llm_provider = st.selectbox(
        "LLM provider",
        ["openrouter", "groq"],
        index=["openrouter", "groq"].index(settings.LLM_PROVIDER),
    )
    llm_model = st.text_input(
        "LLM model",
        value=settings.LLM_MODEL,
        help="Для качества попробуйте groq / llama-3.3-70b-versatile. Для скорости — llama-3.1-8b-instant.",
    )
    device = st.selectbox("Device", ["cpu", "cuda"], index=["cpu", "cuda"].index(settings.DEVICE))
    crop_mode = st.selectbox("Crop mode", ["smart_center", "face_tracking"], index=["smart_center", "face_tracking"].index(settings.CROP_MODE))
    output_res = st.selectbox("Output resolution", ["720x1280", "1080x1920"], index=["720x1280", "1080x1920"].index(settings.OUTPUT_RESOLUTION))
    ffmpeg_preset = st.selectbox(
        "FFmpeg preset",
        ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        index=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"].index(settings.FFMPEG_PRESET),
    )
    ffmpeg_crf = st.slider("FFmpeg CRF", 0, 51, int(settings.FFMPEG_CRF))
    use_nvenc = st.checkbox("Use NVENC", value=bool(settings.USE_NVENC))
    num_clips = st.slider("Default clips", 1, 10, int(settings.NUM_CLIPS))
    candidate_count = st.slider("Candidates to preview", 4, 20, int(getattr(settings, "HIGHLIGHT_CANDIDATE_COUNT", 12)))
    min_dur = st.number_input("Min clip duration", value=int(settings.MIN_CLIP_DURATION))
    max_dur = st.number_input("Max clip duration", value=int(settings.MAX_CLIP_DURATION))
    sub_style = st.selectbox("Subtitle style", ["title_only", "word_by_word"], index=["title_only", "word_by_word"].index(settings.SUBTITLE_STYLE))
    sub_font = st.slider("Subtitle font size", 30, 110, int(settings.SUBTITLE_FONT_SIZE))
    sub_color = st.color_picker("Subtitle color", settings.SUBTITLE_COLOR if str(settings.SUBTITLE_COLOR).startswith("#") else "#FFFFFF")
    sub_active_color = st.color_picker(
        "Active word color",
        getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00") if str(getattr(settings, "SUBTITLE_ACTIVE_COLOR", "#FFFF00")).startswith("#") else "#FFFF00",
    )
    sub_words_per_caption = st.slider("Words per caption", 1, 5, int(getattr(settings, "SUBTITLE_WORDS_PER_CAPTION", 3)))
    sub_timing_offset = st.slider("Subtitle timing offset (ms)", -300, 300, int(getattr(settings, "SUBTITLE_TIMING_OFFSET_MS", -80)), step=10)
    if st.button("Save settings", use_container_width=True):
        if min_dur > max_dur:
            st.error("Min duration cannot be greater than max duration.")
        else:
            settings.save(current_ui_config())
            st.success("Settings saved")

with st.sidebar.expander("Backup", expanded=False):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_yaml_data = f.read()
        st.download_button("Export config.yaml", config_yaml_data, file_name="config.yaml", mime="text/yaml", use_container_width=True)
    uploaded_file = st.file_uploader("Import config", type=["yaml", "yml"])
    if uploaded_file is not None:
        try:
            settings.save(yaml.safe_load(uploaded_file) or {})
            st.success("Settings imported")
            st.rerun()
        except Exception as e:
            st.error(f"Import error: {e}")

# Make sure settings variables exist even if expander has not been interacted with.
try:
    current_ui_config
except NameError:
    pass

project = load_active_project()

# --- Main Studio ---
st.title("🎬 Klippr Studio")
st.caption("Opus-like workflow with your own UI: Source → Analyze → Review → Render → Export")

if project is None:
    st.info("Создайте проект в сайдбаре, чтобы начать.")
    st.stop()

candidates = load_candidates(project["id"])
transcript = load_transcript(project["id"])
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
    name = st.text_input("Project name", value=project.get("name", "Untitled project"))
    source_url = st.text_input("YouTube URL", value=project.get("source_url", ""), placeholder="https://www.youtube.com/watch?v=...")
    notes = st.text_area("Notes", value=project.get("notes", ""), height=100)
    if st.button("Save source", type="primary"):
        project["name"] = name.strip() or "Untitled project"
        project["source_url"] = source_url.strip()
        project["notes"] = notes
        project["settings_snapshot"] = current_ui_config()
        save_project(project)
        st.success("Source saved")
        st.rerun()

    if project.get("video_path") and Path(project["video_path"]).exists():
        st.success("Source video is downloaded.")
        st.video(project["video_path"])
    else:
        st.info("Видео ещё не скачано. Перейдите во вкладку Analyze.")

with analyze_tab:
    st.subheader("Analyze video")
    st.write("Скачивание, аудио, транскрипт и AI-кандидаты сохраняются в папку проекта.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Transcript segments", len(transcript))
    col_b.metric("Candidates", len(candidates))
    col_c.metric("Model", settings.LLM_MODEL)

    analyze_clicked = st.button("🔎 Analyze / regenerate candidates", type="primary", use_container_width=True)
    if analyze_clicked:
        if not project.get("source_url"):
            st.error("Сначала укажите YouTube URL во вкладке Source.")
        elif min_dur > max_dur:
            st.error("Min duration cannot be greater than max duration.")
        else:
            reset_logs()
            settings.save(current_ui_config())
            project["settings_snapshot"] = current_ui_config()
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
                candidates = analyzer.find_highlight_candidates(transcript, num_candidates=int(candidate_count))
                candidates = analyzer.snap_to_silence(candidates, audio_path, transcript)
                project["candidates_path"] = save_candidates(project["id"], candidates)
                project["selected_candidate_ids"] = [c.get("id") for c in candidates[: int(num_clips)] if c.get("id")]
                project["status"] = "candidates_ready"
                save_project(project)
                update_logs()

                progress_bar.progress(100, text="Candidates ready")
                st.success(f"Готово: найдено кандидатов {len(candidates)}")
                st.rerun()
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
            table_rows.append({
                "№": i + 1,
                "time": f"{format_time(clip.get('start_time', 0))}–{format_time(clip.get('end_time', 0))}",
                "duration": clip.get("duration", round(float(clip.get("end_time", 0)) - float(clip.get("start_time", 0)), 1)),
                "score": candidate_score(clip),
                "hook": clip.get("hook", ""),
                "title": clip.get("title", ""),
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        st.markdown("### Candidate details")
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

        default_ids = [cid for cid in project.get("selected_candidate_ids", []) if cid]
        if not default_ids:
            default_ids = [c.get("id") for c in candidates[: int(num_clips)] if c.get("id")]
        candidate_ids = [c.get("id") for c in candidates]
        selected_ids = st.multiselect(
            "Selected candidates",
            options=candidate_ids,
            default=[cid for cid in default_ids if cid in candidate_ids],
            format_func=lambda cid: candidate_label(candidate_ids.index(cid), candidates[candidate_ids.index(cid)]),
        )
        if st.button("Save selection", type="primary"):
            project["selected_candidate_ids"] = selected_ids
            project["status"] = "selection_ready" if selected_ids else project.get("status", "candidates_ready")
            save_project(project)
            st.success("Selection saved")
            st.rerun()

with render_tab:
    st.subheader("Render selected clips")
    if not candidates:
        st.info("Сначала запустите Analyze.")
    else:
        selected_ids = project.get("selected_candidate_ids", [])
        selected = [c for c in candidates if c.get("id") in selected_ids]
        if not selected:
            st.warning("Сначала выберите кандидаты во вкладке Review.")
        else:
            st.write(f"Selected clips: {len(selected)}")
            for i, clip in enumerate(selected):
                st.caption(candidate_label(i, clip))

            render_clicked = st.button("🎬 Render selected", type="primary", use_container_width=True)
            if render_clicked:
                if not project.get("video_path") or not Path(project["video_path"]).exists():
                    st.error("Source video not found. Run Analyze again.")
                else:
                    reset_logs()
                    project["status"] = "rendering"
                    save_project(project)
                    progress_bar = st.progress(0, text="Initializing render...")
                    log_box = st.empty()

                    def update_logs():
                        log_box.text(log_text_value())

                    try:
                        res = tuple(map(int, settings.OUTPUT_RESOLUTION.split("x")))
                        renderer = VerticalRenderer(output_dir=str(clips_dir(project["id"])), resolution=res)
                        transcript = load_transcript(project["id"])
                        project["clips"] = []
                        for i, hl in enumerate(sorted(selected, key=lambda c: c["start_time"])):
                            progress_bar.progress(int(100 * (i / max(len(selected), 1))), text=f"Rendering clip {i + 1}/{len(selected)}...")
                            safe_name = f"clip_{i + 1:02d}.mp4"
                            clip_path = str(clips_dir(project["id"]) / safe_name)
                            renderer.render_clip(project["video_path"], hl, clip_path, transcript=transcript)
                            project = add_clip(project, clip_path, hl)
                            update_logs()

                        project["status"] = "rendered"
                        save_project(project)
                        progress_bar.progress(100, text="Rendered")
                        st.success(f"Готово: создано клипов {len(project.get('clips', []))}")
                        st.rerun()
                    except Exception as e:
                        project["status"] = "render_error"
                        save_project(project)
                        progress_bar.progress(100, text="Render failed")
                        st.error(f"Ошибка рендера: {e}")
                        update_logs()

with export_tab:
    st.subheader("Export")
    clips = project.get("clips", [])
    if not clips:
        st.info("Клипов пока нет. Перейдите во вкладку Render.")
    else:
        cols = st.columns(min(3, len(clips)))
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
                        )
                else:
                    st.warning("File missing")

with logs_tab:
    st.subheader("Logs")
    st.text_area("Current session logs", value=log_text_value(), height=500)
    if st.button("Clear logs"):
        reset_logs()
        st.rerun()
