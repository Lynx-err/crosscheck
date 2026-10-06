"""Configuration models and YAML loading for CrossCheck."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class ModelConfig(BaseModel):
    """Configuration for one model endpoint in the ensemble."""

    model_config = {"extra": "forbid"}

    id: str = Field(..., min_length=1)
    provider: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    base_url: str = Field(..., min_length=1)
    family: str | None = None
    parameter_size: str | None = None
    api_key: str = Field(default="ollama")
    role: str = Field(default="verifier")
    enabled: bool = True

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"primary", "verifier"}:
            raise ValueError("role must be either 'primary' or 'verifier'")
        return normalized


class ModelRegistry(BaseModel):
    """List of model configurations read from YAML."""

    models: list[ModelConfig]
    nli_model: str = Field(..., min_length=1)

    def get(self, model_id: str) -> ModelConfig:
        for model in self.models:
            if model.id == model_id:
                return model
        raise KeyError(f"Unknown model id: {model_id}")

    def primary_models(self) -> list[ModelConfig]:
        return [model for model in self.models if model.role == "primary" and model.enabled]

    def verifier_models(self) -> list[ModelConfig]:
        return [model for model in self.models if model.role == "verifier" and model.enabled]


def load_model_registry(config_path: str | Path | None = None) -> ModelRegistry:
    """Load the YAML registry and validate the model definitions."""

    if config_path is None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "models.yaml"

    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model config not found: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(f"Model config must contain a top-level 'models' list: {resolved_path}")

    try:
        return ModelRegistry.model_validate(raw)
    except ValidationError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid model config in {resolved_path}: {exc}") from exc


def default_registry() -> ModelRegistry:
    """Return the project default registry."""

    return load_model_registry()
