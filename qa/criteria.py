from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


IMPLEMENTATION_PLAN_FILENAME = "implementation_plan.json"


def load_implementation_plan(spec_dir: Path) -> Dict[str, Any]:
    plan_path = spec_dir / IMPLEMENTATION_PLAN_FILENAME
    if not plan_path.exists():
        return {}
    try:
        with plan_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return {}


def save_implementation_plan(spec_dir: Path, plan: Dict[str, Any]) -> Path:
    plan_path = spec_dir / IMPLEMENTATION_PLAN_FILENAME
    with plan_path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, sort_keys=True)
    return plan_path
