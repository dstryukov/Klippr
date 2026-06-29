import os
import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE = "config.yaml"

SUPPORTED_LLM_PROVIDERS = ["openrouter", "groq", "gemini", "fireworks"]

DEFAULT_CONFIG = {
    "whisper_model": "small",
    "llm_provider": "openrouter",
    "llm_model": "google/gemini-2.5-flash-preview-05-20",
    "device": "cuda",
    "crop_mode": "smart_center",
    "output_resolution": "1080x1920",
    "min_clip_duration": 30,
    "max_clip_duration": 90,
    "subtitle_style": "title_only",
    "subtitle_font_size": 70,
    "subtitle_color": "white",
    "subtitle_active_color": "#FFFF00",
    "subtitle_stroke_color": "black",
    "subtitle_words_per_caption": 3,
    "subtitle_timing_offset_ms": -80,
    "ffmpeg_preset": "fast",
    "ffmpeg_crf": 23,
    "use_nvenc": False,
    # Face tracking tuning
    "face_tracking_skip_frames": 5,
    "face_tracking_max_shift": 40,
    "face_tracking_lookahead": 5,
    # Diarization
    "enable_diarization": True,
    # Hook overlay (like Opus Clips)
    "hook_overlay_enabled": True,
    "hook_overlay_duration": 4,
    "hook_overlay_font_size": 80,
    "hook_overlay_color": "#FFFFFF",
    "hook_overlay_bg_color": "#000000",
    "hook_overlay_position": "top",
}

class EnvSettings(BaseSettings):
    """Loads sensitive API keys from .env"""
    OPENROUTER_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    GEMINI_API_KEY: SecretStr | None = None
    FIREWORKS_API_KEY: SecretStr | None = None
    HF_TOKEN: SecretStr | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

class SettingsManager:
    def __init__(self):
        # 1. Load keys from .env
        self.env = EnvSettings()
        
        # 2. Export secrets to os.environ so third-party libs (huggingface_hub, etc.) can use them
        if self.env.HF_TOKEN:
            os.environ.setdefault("HF_TOKEN", self.env.HF_TOKEN.get_secret_value())
        if self.env.OPENROUTER_API_KEY:
            os.environ.setdefault("OPENROUTER_API_KEY", self.env.OPENROUTER_API_KEY.get_secret_value())
        if self.env.GROQ_API_KEY:
            os.environ.setdefault("GROQ_API_KEY", self.env.GROQ_API_KEY.get_secret_value())
        if self.env.GEMINI_API_KEY:
            os.environ.setdefault("GEMINI_API_KEY", self.env.GEMINI_API_KEY.get_secret_value())
        
        # 3. Load other settings from config.yaml
        self._load_yaml()

    def _load_yaml(self):
        if not os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False)
            
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            
        # Ensure all default keys exist
        modified = False
        for k, v in DEFAULT_CONFIG.items():
            if k not in data:
                data[k] = v
                modified = True
                
        if modified:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)
                
        # Set as uppercase attributes
        for k, v in data.items():
            setattr(self, k.upper(), v)

    def save(self, new_settings: dict):
        """Saves new non-sensitive settings to yaml"""
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
            
        data.update(new_settings)
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        self._load_yaml()

    # Pass-through for secrets
    @property
    def OPENROUTER_API_KEY(self):
        return self.env.OPENROUTER_API_KEY

    @property
    def GROQ_API_KEY(self):
        return self.env.GROQ_API_KEY

    @property
    def GEMINI_API_KEY(self):
        return self.env.GEMINI_API_KEY

    @property
    def FIREWORKS_API_KEY(self):
        return self.env.FIREWORKS_API_KEY

    @property
    def HF_TOKEN(self):
        return self.env.HF_TOKEN

settings = SettingsManager()
