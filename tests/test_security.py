#!/usr/bin/env python3
"""
Tests for Security System
=========================

Tests the security.py module functionality including:
- Command extraction and parsing
- Command allowlist validation
- Sensitive command validators (rm, chmod, pkill, etc.)
- Security hook behavior
"""

import pytest

from security import (
    extract_commands,
    split_command_segments,
    validate_command,
    validate_pkill_command,
    validate_kill_command,
    validate_chmod_command,
    validate_rm_command,
    validate_git_commit,
    validate_dropdb_command,
    validate_dropuser_command,
    validate_psql_command,
    validate_mysql_command,
    validate_redis_cli_command,
    validate_mongosh_command,
    validate_mysqladmin_command,
    get_command_for_validation,
    reset_profile_cache,
)
from project_analyzer import SecurityProfile, BASE_COMMANDS


class TestCommandExtraction:
    """Tests for command extraction from shell strings."""

    def test_simple_command(self):
        """Extracts single command correctly."""
        commands = extract_commands("ls -la")
        assert commands == ["ls"]

    def test_command_with_path(self):
        """Extracts command from path."""
        commands = extract_commands("/usr/bin/python script.py")
        assert commands == ["python"]

    def test_piped_commands(self):
        """Extracts all commands from pipeline."""
        commands = extract_commands("cat file.txt | grep pattern | wc -l")
        assert commands == ["cat", "grep", "wc"]

    def test_chained_commands_and(self):
        """Extracts commands from && chain."""
        commands = extract_commands("cd /tmp && ls && pwd")
        assert commands == ["cd", "ls", "pwd"]

    def test_chained_commands_or(self):
        """Extracts commands from || chain."""
        commands = extract_commands("test -f file || echo 'not found'")
        assert commands == ["test", "echo"]

    def test_semicolon_separated(self):
        """Extracts commands separated by semicolons."""
        commands = extract_commands("echo hello; echo world; ls")
        assert commands == ["echo", "echo", "ls"]

    def test_mixed_operators(self):
        """Handles mixed operators correctly."""
        commands = extract_commands("cmd1 && cmd2 || cmd3; cmd4 | cmd5")
        assert commands == ["cmd1", "cmd2", "cmd3", "cmd4", "cmd5"]

    def test_skips_flags(self):
        """Doesn't include flags as commands."""
        commands = extract_commands("ls -la --color=auto")
        assert commands == ["ls"]

    def test_skips_variable_assignments(self):
        """Skips variable assignments."""
        commands = extract_commands("VAR=value echo $VAR")
        assert commands == ["echo"]

    def test_handles_quotes(self):
        """Handles quoted arguments."""
        commands = extract_commands('echo "hello world" && grep "pattern with spaces"')
        assert commands == ["echo", "grep"]

    def test_empty_string(self):
        """Returns empty list for empty string."""
        commands = extract_commands("")
        assert commands == []

    def test_malformed_command(self):
        """Returns empty list for malformed command (fail-safe)."""
        commands = extract_commands("echo 'unclosed quote")
        assert commands == []


class TestSplitCommandSegments:
    """Tests for splitting command strings into segments."""

    def test_single_command(self):
        """Single command returns one segment."""
        segments = split_command_segments("ls -la")
        assert segments == ["ls -la"]

    def test_and_chain(self):
        """Splits on &&."""
        segments = split_command_segments("cd /tmp && ls")
        assert segments == ["cd /tmp", "ls"]

    def test_or_chain(self):
        """Splits on ||."""
        segments = split_command_segments("test -f file || echo error")
        assert segments == ["test -f file", "echo error"]

    def test_semicolon(self):
        """Splits on semicolons."""
        segments = split_command_segments("echo a; echo b; echo c")
        assert segments == ["echo a", "echo b", "echo c"]


class TestPkillValidator:
    """Tests for pkill command validation."""

    def test_allowed_process_node(self):
        """Allows killing node processes."""
        allowed, reason = validate_pkill_command("pkill -f node")
        assert allowed is True

    def test_allowed_process_python(self):
        """Allows killing python processes."""
        allowed, reason = validate_pkill_command("pkill python")
        assert allowed is True

    def test_allowed_process_vite(self):
        """Allows killing vite processes."""
        allowed, reason = validate_pkill_command("pkill vite")
        assert allowed is True

    def test_blocked_system_process(self):
        """Blocks killing system processes."""
        allowed, reason = validate_pkill_command("pkill init")
        assert allowed is False
        assert "dev processes" in reason

    def test_blocked_arbitrary_process(self):
        """Blocks killing arbitrary processes."""
        allowed, reason = validate_pkill_command("pkill systemd")
        assert allowed is False


class TestKillValidator:
    """Tests for kill command validation."""

    def test_allowed_specific_pid(self):
        """Allows killing specific PID."""
        allowed, reason = validate_kill_command("kill 12345")
        assert allowed is True

    def test_allowed_with_signal(self):
        """Allows kill with signal."""
        allowed, reason = validate_kill_command("kill -9 12345")
        assert allowed is True

    def test_blocked_kill_all(self):
        """Blocks kill -1 (kill all)."""
        allowed, reason = validate_kill_command("kill -9 -1")
        assert allowed is False
        assert "all processes" in reason

    def test_blocked_kill_group_zero(self):
        """Blocks kill 0 (process group)."""
        allowed, reason = validate_kill_command("kill 0")
        assert allowed is False
