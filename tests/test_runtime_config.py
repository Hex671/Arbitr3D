"""Credential/config checks: no GPU, data, installed SDK, or network needed."""

import os
import unittest
from unittest.mock import patch

from utils.runtime_config import resolve_api_key, resolve_base_url, resolve_model_name


class RuntimeConfigTests(unittest.TestCase):
    def test_default_and_custom_environment_names(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "default", "VISION_KEY": "custom"}, clear=True):
            self.assertEqual(resolve_api_key({}), "default")
            self.assertEqual(resolve_api_key({"api_key_env": "VISION_KEY"}), "custom")

    def test_local_server_requires_explicit_placeholder(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Set OPENAI_API_KEY"):
                resolve_api_key({})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY"}, clear=True):
            self.assertEqual(resolve_api_key({}), "EMPTY")

    def test_literal_credential_is_rejected_without_echoing_it(self):
        credential = "sk-" + "example-credential"
        with self.assertRaises(ValueError) as context:
            resolve_api_key({"api_key_env": credential})
        self.assertNotIn(credential, str(context.exception))

    def test_empty_credential_is_rejected(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "  "}, clear=True):
            with self.assertRaises(ValueError):
                resolve_api_key({})

    def test_endpoint_and_model_overrides(self):
        config = {"base_url": "http://localhost:8000/v1", "model_name": "config-model"}
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_base_url(config), config["base_url"])
            self.assertEqual(resolve_model_name(config), "config-model")
        with patch.dict(os.environ, {"MLLM_BASE_URL": "http://localhost:9000/v1", "MLLM_MODEL_NAME": "env-model"}, clear=True):
            self.assertEqual(resolve_base_url(config), "http://localhost:9000/v1")
            self.assertEqual(resolve_model_name(config), "env-model")


if __name__ == "__main__":
    unittest.main()
