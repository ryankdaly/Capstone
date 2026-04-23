"""Unit tests for configuration loading."""

from pathlib import Path

from backend.config import HpemaConfig, load_config


class TestConfig:
    def test_defaults(self):
        config = HpemaConfig()
        assert config.models.actor.model == "gpt-oss-120b"
        assert config.pipeline.max_iterations == 3
        assert config.verification.prover == "dafny"

    def test_load_missing_file(self, tmp_path):
        config = load_config(tmp_path / "nonexistent.yaml")
        assert config.models.actor.endpoint == "https://llm-api.arc.vt.edu/api/v1"

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
models:
  actor:
    endpoint: "http://custom:8001/v1"
    model: "custom-model"
pipeline:
  max_iterations: 5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.models.actor.endpoint == "http://custom:8001/v1"
        assert config.models.actor.model == "custom-model"
        assert config.pipeline.max_iterations == 5
        # Defaults still work for unset fields
        assert config.models.checker.model == "gpt-oss-120b"
