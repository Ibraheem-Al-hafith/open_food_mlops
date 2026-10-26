"""Runtime configuration for the UI, loaded from ``UI_*`` environment variables."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class UiSettings(BaseSettings):
    """Settings of the Gradio application (env prefix ``UI_``)."""

    model_config = SettingsConfigDict(env_prefix="UI_", extra="ignore", case_sensitive=False)

    api_url: str = Field(
        default="http://fastapi:8000",
        description="Base URL of the serving API, reachable from the UI container.",
    )
    host: str = Field(default="0.0.0.0", description="Interface the UI binds to.")
    port: int = Field(default=7860, ge=1, le=65535, description="Port the UI binds to.")
    root_path: str = Field(
        default="",
        description="URL prefix when served behind a reverse proxy, e.g. '/ui'.",
    )
    request_timeout_s: float = Field(default=60.0, gt=0, description="HTTP timeout per call.")
    max_batch_rows: int = Field(default=5000, ge=1, description="Largest accepted upload.")
    batch_chunk_size: int = Field(
        default=500, ge=1, description="Rows sent per API call (drives progress updates)."
    )
    auth_username: Optional[str] = Field(default=None, description="Enables basic auth when set.")
    auth_password: Optional[SecretStr] = Field(default=None)

    @field_validator("auth_username", "auth_password", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        """Treat blank variables (``UI_AUTH_USERNAME=``) as unset."""
        return None if value == "" else value

    @field_validator("api_url", "root_path")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def auth(self) -> Optional[tuple[str, str]]:
        """Credentials for Gradio's ``auth`` argument, or ``None`` when disabled."""
        if self.auth_username and self.auth_password:
            return self.auth_username, self.auth_password.get_secret_value()
        return None
