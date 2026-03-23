import json
from pathlib import Path
from unittest.mock import patch

import pytest

from spec.phase_executor import PhaseExecutor, MAX_RETRIES


class TestPhaseSelfCritique:
    """Tests for phase_self_critique method."""

    @pytest.mark.asyncio
    async def test_self_critique_no_spec(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Self-critique fails if spec.md doesn't exist."""
        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        result = await executor.phase_self_critique()

        assert result.success is False
        assert "spec.md does not exist" in result.errors[0]

    @pytest.mark.asyncio
    async def test_self_critique_already_completed(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Self-critique returns early if already completed."""
        (spec_dir / "spec.md").write_text("# Test Spec")
        (spec_dir / "critique_report.json").write_text(
            json.dumps(
                {
                    "issues_fixed": True,
                    "no_issues_found": False,
                }
            )
        )

        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        result = await executor.phase_self_critique()

        assert result.success is True
        assert result.retries == 0


class TestPhasePlanning:
    """Tests for phase_planning method."""

    @pytest.mark.asyncio
    async def test_planning_file_exists_valid(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Planning phase returns early if valid plan exists."""
        (spec_dir / "implementation_plan.json").write_text(
            json.dumps({"phases": [{"phase": 1, "subtasks": []}]})
        )

        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(plan_valid=True),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        result = await executor.phase_planning()

        assert result.success is True
        assert result.phase == "planning"
        assert result.retries == 0


class TestPhaseValidation:
    """Tests for phase_validation method."""

    @pytest.mark.asyncio
    async def test_validation_all_pass(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Validation phase passes when all validations pass."""
        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(
                spec_valid=True,
                plan_valid=True,
                context_valid=True,
                all_valid=True,
            ),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        result = await executor.phase_validation()

        assert result.success is True
        assert result.phase == "validation"

    @pytest.mark.asyncio
    async def test_validation_retries_on_failure(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Validation phase retries with auto-fix agent on failure."""
        # Create agent mock that simulates failure
        agent_fn = mock_run_agent_fn(success=False, output="Fix failed")

        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(all_valid=False),
            run_agent_fn=agent_fn,
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        result = await executor.phase_validation()

        assert result.success is False
        assert result.retries == MAX_RETRIES


class TestRunScript:
    """Tests for _run_script helper method."""

    def test_run_script_not_found(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """_run_script returns False when script not found."""
        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        success, output = executor._run_script("nonexistent.py", [])

        assert success is False
        assert "not found" in output.lower()


class TestMaxRetriesConstant:
    """Tests for MAX_RETRIES configuration."""

    def test_max_retries_is_positive(self):
        """MAX_RETRIES is a positive integer."""
        assert MAX_RETRIES > 0
        assert isinstance(MAX_RETRIES, int)

    def test_max_retries_reasonable(self):
        """MAX_RETRIES is a reasonable value."""
        assert 1 <= MAX_RETRIES <= 10


class TestPhaseWorkflow:
    """Integration tests for phase workflow patterns."""

    @pytest.mark.asyncio
    async def test_phases_are_idempotent(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Running a phase twice with existing output is idempotent."""
        # Pre-create files
        (spec_dir / "requirements.json").write_text(
            json.dumps({"task_description": "Test"})
        )
        (spec_dir / "context.json").write_text(
            json.dumps({"task_description": "Test"})
        )

        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        # Run phases twice
        result1 = await executor.phase_requirements(interactive=False)
        result2 = await executor.phase_requirements(interactive=False)

        assert result1.success is True
        assert result2.success is True
        assert result1.retries == 0
        assert result2.retries == 0

    @pytest.mark.asyncio
    async def test_phases_log_to_task_logger(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Phases log messages to task logger."""
        (spec_dir / "project_index.json").write_text(json.dumps({"files": []}))

        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        with patch("spec.discovery.run_discovery_script", return_value=(True, "Success")):
            with patch(
                "spec.discovery.get_project_index_stats",
                return_value={"file_count": 10},
            ):
                await executor.phase_discovery()

        # Verify logger was called
        assert mock_task_logger.log.called

    @pytest.mark.asyncio
    async def test_phases_print_status(
        self,
        temp_dir: Path,
        spec_dir: Path,
        mock_run_agent_fn,
        mock_task_logger,
        mock_ui_module,
        mock_spec_validator,
    ):
        """Phases print status messages via UI module."""
        executor = PhaseExecutor(
            project_dir=temp_dir,
            spec_dir=spec_dir,
            task_description="Test task",
            spec_validator=mock_spec_validator(),
            run_agent_fn=mock_run_agent_fn(),
            task_logger=mock_task_logger,
            ui_module=mock_ui_module,
        )

        await executor.phase_requirements(interactive=False)

        # Verify UI print_status was called
        assert mock_ui_module.print_status.called
