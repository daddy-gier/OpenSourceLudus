# Contributing to Auto Claude

Thank you for your interest in contributing to Auto Claude! This document
provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Setup](#development-setup)
  - [Python Backend](#python-backend)
  - [Electron Frontend](#electron-frontend)
- [Running from Source](#running-from-source)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Code Style](#code-style)
- [Testing](#testing)
- [Continuous Integration](#continuous-integration)
- [Git Workflow](#git-workflow)
  - [Branch Overview](#branch-overview)
  - [Main Branches](#main-branches)
  - [Supporting Branches](#supporting-branches)
  - [Branch Naming](#branch-naming)
  - [Where to Branch From](#where-to-branch-from)
  - [Pull Request Targets](#pull-request-targets)
  - [Release Process (Maintainers)](#release-process-maintainers)
  - [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Architecture Overview](#architecture-overview)

## Prerequisites

Before contributing, ensure you have the following installed:

- **Python 3.12+** - For the backend framework
- **Node.js 24+** - For the Electron frontend
- **npm 10+** - Package manager for the frontend (comes with Node.js)
- **uv** (recommended) or **pip** - Python package manager
- **Git** - Version control

### Installing Python 3.12

**Windows:**
```bash
winget install Python.Python.3.12
```

**macOS:**
```bash
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install python3.12 python3.12-venv
```

## Quick Start

The fastest way to get started:

```bash
# Clone the repository
git clone https://github.com/AndyMik90/Auto-Claude.git
cd Auto-Claude

# Install all dependencies (cross-platform)
npm run install:all

# Run in development mode
npm run dev

# Or build and run production
npm start
```

## Development Setup

The project consists of two main components:

1. **Python Backend** (`apps/backend/`) - The core autonomous coding framework
2. **Electron Frontend** (`apps/frontend/`) - Optional desktop UI

### Python Backend

The recommended way is to use `npm run install:backend`, but you can also
set up manually:

```bash
# Navigate to the backend directory
cd apps/backend

# Create virtual environment
# Windows:
py -3.12 -m venv .venv
.venv\Scripts\activate

# macOS/Linux:
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r ../../tests/requirements-test.txt

# Set up environment
cp .env.example .env
# Edit .env and add your CLAUDE_CODE_OAUTH_TOKEN (get it via: claude setup-token)
```

### Electron Frontend

```bash
# Navigate to the frontend directory
cd apps/frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Package for distribution
npm run package
```

## Running from Source

If you want to run Auto Claude from source (for development or testing
unreleased features), follow these steps:

### Step 1: Clone and Set Up

```bash
git clone https://github.com/AndyMik90/Auto-Claude.git
cd Auto-Claude/apps/backend

# Using uv (recommended)
uv venv && uv pip install -r requirements.txt

# Or using standard Python
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set up environment
cd apps/backend
cp .env.example .env
# Edit .env and add your CLAUDE_CODE_OAUTH_TOKEN (get it via: claude setup-token)
```

### Step 2: Run the Desktop UI

```bash
cd ../frontend

# Install dependencies
npm install

# Development mode (hot reload)
npm run dev

# Or production build
npm run build && npm run start
```

## Pre-commit Hooks

Install and run any configured hooks prior to opening a PR.

## Code Style

Follow existing project conventions and keep changes focused and readable.

## Testing

```bash
# Python (from repository root)
npm run test:backend

# Frontend
cd apps/frontend && npm test && npm run lint && npm run typecheck
```

## Continuous Integration

CI runs the same checks as the testing commands above.

## Git Workflow

### Branch Overview

```
main ─────●─────●─────●─────●───── (production)
          ↑     ↑     ↑     ↑
develop ──●─────●─────●─────●───── (integration)
          ↑     ↑     ↑
feature/123 ────●
feature/124 ──────────●
hotfix/125 ─────────────────●───── (from main, merge to both)
```

### Main Branches

- `main` - production releases
- `develop` - integration branch for upcoming releases

### Supporting Branches

- `feature/*` - new features (branch from `develop`)
- `hotfix/*` - urgent production fixes (branch from `main`)

### Branch Naming

Use descriptive branch names:
- `feature/add-dark-mode`
- `hotfix/150-critical-fix`

### Where to Branch From

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Pull Request Targets

- `feature/*` → `develop`
- `hotfix/*` → `main` (then sync back to `develop`)

### Release Process (Maintainers)

See [RELEASE_PROCESS.md](RELEASE_PROCESS.md) for the full flow.

### Commit Messages

Write clear, concise commit messages that explain the **why** behind changes.

**Format:**
```
<type>: <subject>

<body>

<footer>
```

- **type**: feat, fix, docs, style, refactor, test, chore
- **subject**: Short description (50 chars max, imperative mood)
- **body**: Detailed explanation if needed (wrap at 72 chars)
- **footer**: Reference issues, breaking changes

**Examples:**
```bash
# Good
git commit -m "Add retry logic for failed API calls\n\nImplements exponential backoff for transient failures.\nFixes #123"

# Avoid
git commit -m "fix stuff"
git commit -m "WIP"
```

## Pull Request Process

1. **Fork the repository** and create your branch from `develop` (not main!)
2. **Make your changes** following the code style guidelines
3. **Test thoroughly** (see [Testing](#testing))
4. **Update documentation** if your changes affect:
   - Public APIs
   - Configuration options
   - User-facing behavior
5. **Create the Pull Request**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include screenshots for UI changes
   - List any breaking changes
6. **PR Title Format**:
   ```
   <type>: <description>
   ```
   Examples:
   - `feat: Add support for custom prompts`
   - `fix: Resolve memory leak in worker process`
   - `docs: Update installation instructions`
7. **Review Process**:
   - Address reviewer feedback promptly
   - Keep the PR focused on a single concern
   - Squash commits if requested

## Issue Reporting

### Bug Reports

When reporting a bug, include:

1. **Clear title** describing the issue
2. **Environment details**:
   - OS and version
   - Python version
   - Node.js version (for UI issues)
   - Auto Claude version
3. **Steps to reproduce** the issue
4. **Expected behavior** vs **actual behavior**
5. **Error messages** or logs (if applicable)
6. **Screenshots** (for UI issues)

### Feature Requests

When requesting a feature:

1. **Describe the problem** you're trying to solve
2. **Explain your proposed solution**
3. **Consider alternatives** you've thought about
4. **Provide context** on your use case

## Architecture Overview

Auto Claude consists of two main parts:

### Python Backend (`apps/backend/`)

The core autonomous coding framework:

- **Entry Points**: `run.py` (build runner), `spec_runner.py` (spec creator)
- **Agent System**: `agent.py`, `client.py`, `prompts/`
- **Execution**: `coordinator.py` (parallel), `worktree.py` (isolation)
- **Memory**: `memory.py` (file-based), `graphiti_memory.py` (graph-based)
- **QA**: `qa_loop.py`, `prompts/qa_*.md`

### Electron Frontend (`apps/frontend/`)

Desktop interface:

- **Main Process**: `src/main/` - Electron main process, IPC handlers
- **Renderer**: `src/renderer/` - React UI components
- **Shared**: `src/shared/` - Types and utilities

For detailed architecture information, see `CLAUDE.md`.
