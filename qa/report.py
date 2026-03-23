from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .criteria import load_implementation_plan, save_implementation_plan

RECURRING_ISSUE_THRESHOLD = 3
ISSUE_SIMILARITY_THRESHOLD = 0.8

_TEST_CONFIG_FILES = {
    "pytest.ini",
    "pyproject.toml",
    "setup.cfg",
    "jest.config.js",
    "jest.config.ts",
    "vitest.config.js",
    "vitest.config.ts",
    "karma.conf.js",
    "cypress.config.js",
    "playwright.config.ts",
    ".rspec",
}

_TEST_FILE_PATTERNS = [
    "test_*.py",
    "*_test.py",
    "*.spec.js",
    "*.spec.ts",
    "*.test.js",
    "*.test.ts",
]


def check_test_discovery(spec_dir: Path) -> Optional[Dict[str, Any]]:
    discovery_path = spec_dir / "test_discovery.json"
    if not discovery_path.exists():
        return None
    try:
        with discovery_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def _has_test_config(project_dir: Path) -> bool:
    return any((project_dir / filename).exists() for filename in _TEST_CONFIG_FILES)


def _iter_test_files(project_dir: Path) -> Iterable[Path]:
    for pattern in _TEST_FILE_PATTERNS:
        yield from project_dir.rglob(pattern)

    spec_helper = project_dir / "spec" / "spec_helper.rb"
    if spec_helper.exists():
        yield spec_helper


def is_no_test_project(spec_dir: Path, project_dir: Path) -> bool:
    discovery = check_test_discovery(spec_dir)
    if discovery is not None:
        frameworks = discovery.get("frameworks", [])
        return len(frameworks) == 0

    if _has_test_config(project_dir):
        return False

    for test_file in _iter_test_files(project_dir):
        if test_file.name == "conftest.py":
            continue
        return False

    return True


def _extract_acceptance_criteria(spec_dir: Path) -> List[str]:
    spec_path = spec_dir / "spec.md"
    if not spec_path.exists():
        return []

    lines = spec_path.read_text(encoding="utf-8").splitlines()
    criteria_lines: List[str] = []
    in_section = False
    for line in lines:
        if line.strip().lower().startswith("## "):
            if in_section:
                break
            in_section = line.strip().lower() == "## acceptance criteria"
            continue
        if in_section and line.strip().startswith("-"):
            criteria_lines.append(line.strip().lstrip("-").strip())
    return criteria_lines


def create_manual_test_plan(spec_dir: Path, spec_name: str) -> Path:
    plan_path = spec_dir / "MANUAL_TEST_PLAN.md"
    criteria = _extract_acceptance_criteria(spec_dir)
    if not criteria:
        criteria = ["Core functionality works as expected"]

    timestamp = datetime.now(timezone.utc).isoformat()
    checklist = "\n".join(f"- [ ] {item}" for item in criteria)

    content = f"""# Manual Test Plan: {spec_name}

**Generated**: {timestamp}
**Reason**: No automated test framework detected

## Overview
Provide a high-level description of the feature and the intended behavior.

## Pre-Test Setup
- [ ] Required environment variables are configured
- [ ] Test data is available

## Functional Tests

### Happy Path
- [ ] Primary use case works correctly

### Edge Cases
- [ ] Empty input handling
- [ ] Large input handling

### Error Handling
- [ ] Invalid inputs are handled gracefully

## Non-Functional Tests

### Performance
- [ ] Response times remain acceptable under load

### Security
- [ ] Input validation blocks malicious input

## Browser/Environment Testing
- [ ] Verify behavior in target browsers/environments

## Acceptance Criteria Checklist
{checklist}

## Sign-off
- [ ] QA sign-off
- [ ] Stakeholder sign-off
"""

    plan_path.write_text(content, encoding="utf-8")
    return plan_path


def get_iteration_history(spec_dir: Path) -> List[Dict[str, Any]]:
    plan = load_implementation_plan(spec_dir)
    history = plan.get("qa_iteration_history", [])
    if isinstance(history, list):
        return history
    return []


def _count_issue_types(issues: Iterable[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for issue in issues:
        issue_type = issue.get("type") or "unknown"
        counts[issue_type] += 1
    return counts


def record_iteration(
    spec_dir: Path,
    iteration: int,
    status: str,
    issues: List[Dict[str, Any]],
    duration_seconds: Optional[float] = None,
) -> bool:
    plan = load_implementation_plan(spec_dir)
    history = plan.get("qa_iteration_history", [])
    if not isinstance(history, list):
        history = []

    record: Dict[str, Any] = {
        "iteration": iteration,
        "status": status,
        "issues": issues,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if duration_seconds is not None:
        record["duration_seconds"] = round(duration_seconds, 2)

    history.append(record)
    plan["qa_iteration_history"] = history

    stats = plan.get("qa_stats", {})
    stats["total_iterations"] = stats.get("total_iterations", 0) + 1
    stats["last_iteration"] = iteration
    stats["last_status"] = status
    issues_by_type = Counter(stats.get("issues_by_type", {}))
    issues_by_type.update(_count_issue_types(issues))
    stats["issues_by_type"] = dict(issues_by_type)
    plan["qa_stats"] = stats

    save_implementation_plan(spec_dir, plan)
    return True


def _normalize_issue_key(issue: Dict[str, Any]) -> str:
    title = issue.get("title") or ""
    file_name = issue.get("file") or ""
    line = issue.get("line") or ""

    title = str(title).strip().lower()
    for prefix in ("error:", "warning:", "exception:", "failed:", "failure:"):
        if title.startswith(prefix):
            title = title[len(prefix):].strip()
            break

    return "|".join([title, str(file_name), str(line)])


def _issue_similarity(issue_a: Dict[str, Any], issue_b: Dict[str, Any]) -> float:
    key_a = _normalize_issue_key(issue_a)
    key_b = _normalize_issue_key(issue_b)
    if key_a == key_b:
        return 1.0
    return SequenceMatcher(None, key_a, key_b).ratio()


def has_recurring_issues(
    current_issues: Iterable[Dict[str, Any]],
    history: Iterable[Dict[str, Any]],
    threshold: int = RECURRING_ISSUE_THRESHOLD,
) -> Tuple[bool, List[Dict[str, Any]]]:
    recurring: List[Dict[str, Any]] = []

    history_issues: List[Dict[str, Any]] = []
    for record in history:
        history_issues.extend(record.get("issues", []))

    for issue in current_issues:
        occurrence_count = 1
        for past_issue in history_issues:
            if _issue_similarity(issue, past_issue) >= ISSUE_SIMILARITY_THRESHOLD:
                occurrence_count += 1
        if occurrence_count >= threshold:
            recurring.append({
                "issue": issue,
                "occurrence_count": occurrence_count,
            })

    return (len(recurring) > 0, recurring)


def get_recurring_issue_summary(history: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total_issues = 0
    issue_counter: Counter = Counter()
    iterations_approved = 0
    iterations_rejected = 0

    for record in history:
        if record.get("status") == "approved":
            iterations_approved += 1
        if record.get("status") == "rejected":
            iterations_rejected += 1

        issues = record.get("issues", []) or []
        total_issues += len(issues)
        for issue in issues:
            title = issue.get("title") or "Unknown"
            issue_counter[title] += 1

    unique_issues = len(issue_counter)
    most_common = [
        {"title": title, "occurrences": count}
        for title, count in issue_counter.most_common()
    ]

    total_iterations = iterations_approved + iterations_rejected
    fix_success_rate = (
        iterations_approved / total_iterations if total_iterations else 0
    )

    return {
        "total_issues": total_issues,
        "unique_issues": unique_issues,
        "most_common": most_common,
        "iterations_approved": iterations_approved,
        "iterations_rejected": iterations_rejected,
        "fix_success_rate": fix_success_rate,
    }
