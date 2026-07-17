import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from cemg.classify import generalize_params, regex_normalize_string
from cemg.graph import make_action_signature


class TestRegexNormalizeString:

    def test_normal_string_unchanged(self):
        assert regex_normalize_string("hello world") == "hello world"

    def test_normalize_uuid(self):
        uuid_str = "123e4567-e89b-12d3-a456-426614174000"
        text = f"User ID is {uuid_str}"
        assert regex_normalize_string(text) == "User ID is [UUID]"

    def test_normalize_timestamp(self):
        ts_str = "2026-07-18T02:30:45Z"
        text = f"Time: {ts_str}"
        assert regex_normalize_string(text) == "Time: [TIMESTAMP]"

        ts_str2 = "2026-07-18 02:30:45"
        text2 = f"Time: {ts_str2}"
        assert regex_normalize_string(text2) == "Time: [TIMESTAMP]"

    def test_normalize_numbers(self):
        assert regex_normalize_string("Version 12.3.4") == "Version [NUM].[NUM].[NUM]"
        assert regex_normalize_string("Step 42 completed in 100.5 seconds") == "Step [NUM] completed in [NUM].[NUM] seconds"


class TestGeneralizeParams:

    def test_does_not_mutate_original(self):
        original = {"path": "data/v1/config.json", "count": 42}
        generalized = generalize_params("read_file", original)
        assert original["count"] == 42
        assert original["path"] == "data/v1/config.json"
        assert generalized["count"] == "[NUM]"

    def test_generalizes_nested_structures(self):
        params = {
            "metadata": {
                "user_id": "123e4567-e89b-12d3-a456-426614174000",
                "score": 9.5
            },
            "history": [1, 2, 3]
        }
        res = generalize_params("custom_tool", params)
        assert res["metadata"]["user_id"] == "[UUID]"
        assert res["metadata"]["score"] == "[NUM]"
        assert res["history"] == ["[NUM]", "[NUM]", "[NUM]"]

    def test_read_file_path_normalization(self):
        # Different filenames in the same folder with different versions should group
        p1 = {"path": "data/v1/file_a.json"}
        p2 = {"path": "data/v2/file_b.json"}
        
        g1 = generalize_params("read_file", p1)
        g2 = generalize_params("read_file", p2)
        
        # Path override extracts: directory + /*extension
        # data/v1/file_a.json -> data/v1/*.json -> regex: data/v[NUM]/*.json
        # data/v2/file_b.json -> data/v2/*.json -> regex: data/v[NUM]/*.json
        assert g1["path"] == "data/v[NUM]/*.json"
        assert g2["path"] == "data/v[NUM]/*.json"


class TestMakeActionSignature:

    def test_signature_is_grouped(self):
        # Different paths but same folder structure/extensions must result in the same signature hash
        sig1 = make_action_signature("read_file", {"path": "data/v1/config.json"})
        sig2 = make_action_signature("read_file", {"path": "data/v2/settings.json"})
        assert sig1 == sig2

    def test_signature_differentiates_extensions(self):
        # Different extensions in the same folder must produce different signatures
        sig1 = make_action_signature("read_file", {"path": "data/v1/config.json"})
        sig2 = make_action_signature("read_file", {"path": "data/v1/config.yaml"})
        assert sig1 != sig2

    def test_signature_differentiates_directories(self):
        # Different directories must produce different signatures
        sig1 = make_action_signature("read_file", {"path": "data/v1/config.json"})
        sig2 = make_action_signature("read_file", {"path": "reports/config.json"})
        assert sig1 != sig2
