import os
import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE = "config.yaml"

DEFAULT_CONFIG = {
    "whisper_model": "small",
    "llm_provider": "openrouter",
    "llm_model": "google/gemini-2.5-flash-preview-05-20",
    "device": "cuda",
    "crop_mode": "smart_center",
    "output_resolution": "1080x1920",
    "num_clips": 3,
    "min_clip_duration": 30,
    "max_clip_duration": 90,
    "subtitle_style": "title_only",
    "subtitle_font_size": 70,
    "subtitle_color": "white",
    "subtitle_stroke_color": "black",
    "ffmpeg_preset": "fast",
    "ffmpeg_crf": 23,
    "use_nvenc": False
}

class EnvSettings(BaseSettings):
    """Loads sensitive API keys from .env"""
    OPENROUTER_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

class SettingsManager:
    def __init__(self):
        # 1. Load keys from .env
        self.env = EnvSettings()
        
        # 2. Load other settings from config.yaml
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

settings = SettingsManager()
