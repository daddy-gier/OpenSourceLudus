import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def spec_dir(temp_dir: Path) -> Path:
    spec = temp_dir / "spec"
    spec.mkdir()
    return spec


@pytest.fixture
def project_dir(temp_dir: Path) -> Path:
    project = temp_dir / "project"
    project.mkdir()
    return project


@pytest.fixture
def spec_with_plan(spec_dir: Path) -> Path:
    plan = {
        "spec_name": "test-spec",
        "qa_signoff": {
            "status": "pending",
            "qa_session": 0,
        },
    }
    plan_file = spec_dir / "implementation_plan.json"
    with plan_file.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle)
    return spec_dir
