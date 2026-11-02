"""Settings: CLI flags > env (NEXTCHAP_*) > .env > ./nextchap.toml > user config.toml > defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

APP_NAME = "nextchap"
PROJECT_CONFIG = Path("nextchap.toml")

Reasoning = Literal["none", "minimal", "low", "medium", "high"]

HAIKU = "anthropic/claude-haiku-5.5"
DEEPSEEK_FLASH = "deepseek/deepseek-v4.1-flash-1017"


class RoleModel(BaseModel):
    model: str
    reasoning: Reasoning | None = "none"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 4000
    fallbacks: list[str] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    writer: RoleModel = RoleModel(
        model=HAIKU,
        reasoning="none",
        temperature=0.9,
        top_p=0.95,
        max_tokens=6000,
        fallbacks=[DEEPSEEK_FLASH],
    )
    planner: RoleModel = RoleModel(
        model=DEEPSEEK_FLASH,
        reasoning="low",
        temperature=0.7,
        max_tokens=6000,
        fallbacks=[HAIKU],
    )
    analyst: RoleModel = RoleModel(
        model=DEEPSEEK_FLASH,
        reasoning="none",
        temperature=0.1,
        max_tokens=3000,
        fallbacks=[HAIKU],
    )
    critic: RoleModel = RoleModel(
        model=HAIKU,
        reasoning="none",
        temperature=0.2,
        max_tokens=2000,
        fallbacks=[DEEPSEEK_FLASH],
    )

    def roles(self) -> dict[str, RoleModel]:
        return {
            "writer": self.writer,
            "planner": self.planner,
            "analyst": self.analyst,
            "critic": self.critic,
        }


class DefaultsConfig(BaseModel):
    """Personal defaults for the most common CLI choices (``nextchap.toml [defaults]``)."""

    chapters: int = Field(default=1, ge=1)
    max_cost: float | None = Field(default=None, ge=0)
    review: bool = False
    preview: bool = True
    # Optional so an explicit false in [generation] keeps working when [defaults] is unset.
    critic: bool | None = None
    export_epub: bool | None = None


class GenerationConfig(BaseModel):
    target_words: tuple[int, int] = (1500, 2000)
    min_words: int = 900
    arc_size: int = 20
    recent_full_count: int = 3
    bible_refresh_every: int = 5
    max_prompt_tokens: int = 90_000
    critic: bool = True
    max_rewrites: int = 1
    max_continuations: int = 2
    max_cost: float | None = None
    scene_separator: str = "* * *"
    export_epub: bool = True


class RetrievalConfig(BaseModel):
    embed_backend: Literal["local", "remote", "hash"] = "local"
    embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
    remote_embed_model: str = "openai/text-embedding-3-small"
    chunk_words: int = 400
    chunk_overlap_words: int = 60
    top_k: int = 6
    candidates: int = 30
    hybrid: bool = True
    dense_weight: float = 0.7
    rerank: bool = False
    rerank_model: str = "Qwen/Qwen3-Reranker-0.6B"


class CleanupConfig(BaseModel):
    extra_noise_patterns: list[str] = Field(default_factory=list)


class NetworkConfig(BaseModel):
    base_url: str = "https://openrouter.ai/api/v1"
    connect_timeout: float = 15.0
    read_timeout: float = 180.0
    max_attempts: int = 5
    app_title: str = "NextChap"
    app_url: str = "https://github.com/nextchap/nextchap"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEXTCHAP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NEXTCHAP_API_KEY", "OPENROUTER_API_KEY", "api_key"),
    )
    data_dir: Path | None = None
    models: ModelsConfig = ModelsConfig()
    defaults: DefaultsConfig = DefaultsConfig()
    generation: GenerationConfig = GenerationConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    cleanup: CleanupConfig = CleanupConfig()
    network: NetworkConfig = NetworkConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=PROJECT_CONFIG),
            TomlConfigSettingsSource(settings_cls, toml_file=user_config_path()),
        )

    @property
    def data_root(self) -> Path:
        """Project-local data root: books land in <root>/novel_books, logs in <root>/logs."""
        return self.data_dir or Path.cwd()

    @property
    def cache_dir(self) -> Path:
        return Path(platformdirs.user_cache_dir(APP_NAME))

    @property
    def log_dir(self) -> Path:
        return self.data_root / "logs"


def user_config_path() -> Path:
    return Path(platformdirs.user_config_dir(APP_NAME)) / "config.toml"


KEYRING_SERVICE = "nextchap-openrouter"


def resolve_api_key(settings: Settings) -> str | None:
    if settings.api_key is not None and settings.api_key.get_secret_value():
        return settings.api_key.get_secret_value()
    try:
        import keyring
    except ImportError:
        return None
    try:
        value = keyring.get_password(KEYRING_SERVICE, "default")
    except Exception:
        return None
    return str(value) if value else None
