# OpenSourceLudus

AI Copilot scripts and CI templates for Unity and Unreal Engine projects.

## Contents

- `.github/workflows/unreal-full-ci.yml`: multi-engine Unreal CI matrix with build, automation tests, optional PIE smoke tests, and an AI-branch governance status check.
- `.github/CODEOWNERS`: baseline rules for AI branch review requirements.
- `Scripts/create_revoltgpt_widget.py`: Unreal Editor Python script to create an Editor Utility Widget for RevoltGPT.
- `Config/revoltgpt.conf`: placeholder config for RevoltGPT API settings.

## Unreal Editor Utility Widget

Run the script in the Unreal Editor Python console with **Editor Scripting Utilities** enabled:

```python
import runpy
runpy.run_path("Scripts/create_revoltgpt_widget.py")
```

The script creates `/Game/RevoltGPT/UI/EUW_RevoltGPT_Panel` and adds named widgets:
- `PromptTextBox`
- `SendButton`
- `ResponseText`

## RevoltGPT Config

Update the placeholder settings in `Config/revoltgpt.conf` before running any API calls.
