"""
Tests for Review System Integration
====================================

Integration tests for complete review workflows:
- Full approval flow from start to finish
- Build readiness checks (run.py simulation)
- Rejection workflows
- Multi-session scenarios
"""

from pathlib import Path

import pytest

from review import ReviewState, REVIEW_STATE_FILE
from tests.review_fixtures import review_spec_dir, complete_spec_dir


class TestFullReviewFlow:
    """Integration tests for basic review workflow."""

    def test_full_approval_flow(self, review_spec_dir: Path) -> None:
        """Test complete approval flow."""
        # 1. Initially not approved
        state = ReviewState.load(review_spec_dir)
        assert not state.is_approved()

        # 2. Add feedback
        state.add_feedback("Needs minor changes", review_spec_dir)

        # 3. Approve
        state.approve(review_spec_dir, approved_by="reviewer")

        # 4. Verify state
        assert state.is_approved()
        assert state.is_approval_valid(review_spec_dir)

        # 5. Reload and verify persisted
        reloaded = ReviewState.load(review_spec_dir)
        assert reloaded.is_approved()
        assert reloaded.approved_by == "reviewer"
        assert len(reloaded.feedback) == 1

    def test_approval_invalidation_on_change(self, review_spec_dir: Path) -> None:
        """Test that spec changes invalidate approval."""
        # 1. Approve initially
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approval_valid(review_spec_dir)

        # 2. Modify spec.md
        spec_file = review_spec_dir / "spec.md"
        original_content = spec_file.read_text()
        spec_file.write_text(original_content + "\n## New Section\n\nAdded content.")

        # 3. Approval should now be invalid
        assert not state.is_approval_valid(review_spec_dir)

        # 4. Re-approve with new hash
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approval_valid(review_spec_dir)

    def test_rejection_flow(self, review_spec_dir: Path) -> None:
        """Test rejection workflow."""
        # 1. Approve first
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="user")
        assert state.is_approved()

        # 2. Reject
        state.reject(review_spec_dir)

        # 3. Verify state
        assert not state.is_approved()

        # 4. Reload and verify
        reloaded = ReviewState.load(review_spec_dir)
        assert not reloaded.is_approved()

    def test_auto_approve_flow(self, review_spec_dir: Path) -> None:
        """Test auto-approve workflow."""
        state = ReviewState()
        state.approve(review_spec_dir, approved_by="auto")

        assert state.is_approved()
        assert state.approved_by == "auto"
        assert state.is_approval_valid(review_spec_dir)

    def test_multiple_review_sessions(self, review_spec_dir: Path) -> None:
        """Test multiple review sessions increment count correctly."""
        state = ReviewState()
        assert state.review_count == 0

        # First review - approve
        state.approve(review_spec_dir, approved_by="user1")
        assert state.review_count == 1

        # Modify spec to invalidate
        (review_spec_dir / "spec.md").write_text("Changed content")
        state.invalidate(review_spec_dir)

        # Second review - reject
        state.reject(review_spec_dir)
        assert state.review_count == 2

        # Third review - approve again
        state.approve(review_spec_dir, approved_by="user2")
        assert state.review_count == 3


class TestFullReviewWorkflowIntegration:
    """
    Integration tests for the complete review workflow.

    These tests verify the full flow from spec creation through
    approval, build readiness check, and invalidation scenarios.
    """

    def test_full_review_flow(self, complete_spec_dir: Path) -> None:
        """
        Test the complete review flow from start to finish.

        This test verifies:
        1. Initial state is not approved
        2. Approval creates review_state.json
        3. After approval, is_approval_valid returns True
        4. Modifying spec invalidates approval
        5. Re-approval works correctly
        """
        # 1. Initial state - no approval
        state = ReviewState.load(complete_spec_dir)
        assert not state.is_approved()
        assert not state.is_approval_valid(complete_spec_dir)

        # Verify review_state.json doesn't exist yet
        state_file = complete_spec_dir / REVIEW_STATE_FILE
        assert not state_file.exists()

        # 2. User adds feedback before approving
        state.add_feedback("Please clarify the API response format", complete_spec_dir)

        # 3. User approves
        state.approve(complete_spec_dir, approved_by="developer")

        # Verify state file was created
        assert state_file.exists()

        # 4. Verify approval is valid
        assert state.is_approved()
        assert state.is_approval_valid(complete_spec_dir)
        assert state.approved_by == "developer"
        assert state.approved_at != ""
        assert state.spec_hash != ""
        assert state.review_count == 1
        assert len(state.feedback) == 1

        # 5. Simulate run.py check - should pass
        reloaded = ReviewState.load(complete_spec_dir)
        assert reloaded.is_approval_valid(complete_spec_dir)

        # 6. Modify spec.md (simulating user edit)
        spec_file = complete_spec_dir / "spec.md"
        original_content = spec_file.read_text()
        spec_file.write_text(
            original_content + "\n\n## Additional Notes\n\nSome extra information.\n"
        )

        # 7. Approval should now be invalid (spec changed)
        assert not reloaded.is_approval_valid(complete_spec_dir)

        # 8. Reload and verify still shows approved but invalid
        fresh_state = ReviewState.load(complete_spec_dir)
        assert fresh_state.approved is True  # Still marked approved
        assert not fresh_state.is_approval_valid(complete_spec_dir)  # But not valid

        # 9. Re-approve after changes
        fresh_state.approve(complete_spec_dir, approved_by="developer")
        assert fresh_state.is_approval_valid(complete_spec_dir)
        assert fresh_state.review_count == 2

    def test_run_py_approval_check_simulation(self, complete_spec_dir: Path) -> None:
        """
        Test the approval check logic as run.py would use it.

        This simulates the exact check that run.py performs before
        starting a build.
        """
        # Initial state - run.py would block
        review_state = ReviewState.load(complete_spec_dir)
        build_should_proceed = review_state.is_approval_valid(complete_spec_dir)
        assert not build_should_proceed, "Build should be blocked without approval"

        # After approval - run.py would proceed
        review_state.approve(complete_spec_dir, approved_by="user")
        build_should_proceed = review_state.is_approval_valid(complete_spec_dir)
        assert build_should_proceed, "Build should proceed after approval"
