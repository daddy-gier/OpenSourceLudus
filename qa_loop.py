from __future__ import annotations

from qa.criteria import load_implementation_plan, save_implementation_plan
from qa.report import (
    ISSUE_SIMILARITY_THRESHOLD,
    RECURRING_ISSUE_THRESHOLD,
    check_test_discovery,
    create_manual_test_plan,
    get_iteration_history,
    get_recurring_issue_summary,
    has_recurring_issues,
    is_no_test_project,
    record_iteration,
    _issue_similarity,
    _normalize_issue_key,
)

__all__ = [
    "get_iteration_history",
    "record_iteration",
    "_normalize_issue_key",
    "_issue_similarity",
    "has_recurring_issues",
    "get_recurring_issue_summary",
    "check_test_discovery",
    "is_no_test_project",
    "create_manual_test_plan",
    "RECURRING_ISSUE_THRESHOLD",
    "ISSUE_SIMILARITY_THRESHOLD",
    "load_implementation_plan",
    "save_implementation_plan",
]
