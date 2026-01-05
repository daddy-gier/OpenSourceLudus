# NYGHTSHADE Unreal Plugin
Automate Unreal Engine with AI, Cursor, and Local Models

NYGHTSHADE Unreal Plugin is an Unreal Engine editor plugin built for developers who want to dramatically accelerate production using AI-driven automation.

Instead of manually opening assets, clicking through menus, or tweaking hundreds of properties by hand, NYGHTSHADE creates a direct, deterministic interface between AI tools and the Unreal Editor — allowing intelligent systems to do the repetitive work for you.

## Documentation Package (README + Docs Structure)

### Overview
NYGHTSHADE Unreal Plugin enables AI-driven automation, asset introspection, and large-scale project manipulation through a deterministic, machine-readable interface.

NYGHTSHADE allows AI tools (Cursor, local LLMs, Python agents, or custom systems) to safely inspect and modify Unreal projects without manual editor interaction.

### Key Principles
- AI-first, editor-native automation
- Deterministic and auditable execution
- Local and offline model support
- No cloud APIs or subscriptions
- Safe by design (validation, logging, rollback)

### Installation
1. Copy the `NYGHTSHADEUnrealPlugin` folder into:
   - `YourProject/Plugins/`
2. Enable NYGHTSHADE Unreal Plugin in the Unreal Editor.
3. Restart the editor.

### Architecture
```
NYGHTSHADEUnrealPlugin/
├─ Core/
│  ├─ AssetIntrospection
│  ├─ CommandExecution
│  ├─ Validation & Safety
│
├─ AIBridge/
│  ├─ JSON Command Interface
│  ├─ Local Socket / File IO
│
├─ Logging/
│  ├─ Change Tracking
│  ├─ Rollback Support
```

### AI Integration Model
NYGHTSHADE exposes structured commands that any AI system can generate.

- JSON-based
- Strongly typed
- Validated before execution
- Optional dry-run mode

Example:
```
{
  "command": "bulk_update",
  "asset_type": "WeaponData",
  "filters": {
    "folder": "/Game/Weapons"
  },
  "changes": {
    "damage": "*0.9"
  },
  "dry_run": true
}
```

### Safety Model
- Read-only vs write commands
- Property whitelisting
- Dry-run diffs
- Full change logs
- Undo / rollback support

AI can act — but never blindly.

### Typical Use Cases
- Weapon balancing
- AI tuning
- NPC stat normalization
- Variant generation
- Large-scale refactors
- Rapid prototyping pipelines

### License & Support
- Editor-only plugin
- No runtime impact
- Marketplace-friendly
- Pro and Enterprise tiers available

## Launch Announcement

🚀 **NYGHTSHADE Unreal Plugin – AI Automation for Unreal Engine**

NYGHTSHADE Unreal Plugin is now available.

NYGHTSHADE connects Unreal Engine directly to AI-driven tools, allowing automation systems, local LLMs, and Cursor-style workflows to safely inspect and modify Unreal projects at scale.

No cloud APIs.
No subscriptions.
No editor micromanagement.

Instead of opening hundreds of assets and tweaking properties by hand, NYGHTSHADE gives AI systems a deterministic, auditable interface into the Unreal Editor.

If you’ve ever thought:

> “AI should be doing this instead of me”

NYGHTSHADE is the missing link.

⚠️ Early pricing is temporary. As automation and AI-native features expand, the price will increase.

## Pricing Tiers

🟢 **Indie License – $5 (Early) / $5**
- Full core plugin
- Local and offline AI support
- Asset introspection
- Bulk editing
- Logging and rollback
- Lifetime updates for v1.x

Best for: Solo devs, indie teams

🔵 **Pro License – $20 / seat**
- Everything in Indie
- Advanced automation commands
- Conditional workflows
- Command presets
- Priority updates

Best for: Small teams, serious production

🟣 **Studio / Enterprise**
- Custom pricing
- Multi-user permissions
- Audit trails
- CI-style automation
- Headless editor workflows
- Long-term support builds
- Direct support and onboarding

Best for: Studios, AAA pipelines, enterprise teams

## Demo Script (Video or Live Demo)

**Demo: “AI Rebalances an Entire Weapon System”**

**Scene 1 – Problem**
“This project has 120 weapon assets. Rebalancing them manually takes hours.”

**Scene 2 – AI Analysis**
- Cursor scans project
- Identifies all weapon assets
- Groups by category

**Scene 3 – Dry Run**
```
{
  "command": "bulk_update",
  "asset_type": "WeaponData",
  "changes": {
    "damage": "*0.92",
    "recoil": "*1.1"
  },
  "dry_run": true
}
```

**Scene 4 – Review**
- Show diff
- Show safety validation
- Confirm execution

**Scene 5 – Result**
“120 assets updated in seconds. Logged. Reversible. Safe.”

**Closing**
“This is AI as a real Unreal Engine assistant.”

## Cursor Prompt Pack

**Prompt 1 – Project Discovery**
Inspect this Unreal project using NYGHTSHADE.
List all major asset categories and summarize their purpose.
Output structured JSON.

**Prompt 2 – Balance Proposal**
Analyze weapon assets.
Propose balance changes to normalize DPS while preserving weapon identity.
Do not execute changes yet.

**Prompt 3 – Safe Execution**
Generate a NYGHTSHADE command that applies the approved balance changes.
Use dry-run mode first.

**Prompt 4 – Variant Generation**
Create three weapon variants per base weapon.
Follow existing naming conventions and tuning patterns.

**Prompt 5 – Audit & Cleanup**
Scan the project for inconsistent or missing gameplay values.
Generate a report and suggest automated fixes.

## Marketplace Screenshots Copy

1. **Hero Shot – Automation Overview**
   - Title: “Automate Unreal Engine with AI”
   - Caption: “NYGHTSHADE turns repetitive editor work into one-command automation.”

2. **Asset Discovery**
   - Title: “Instant Project Insight”
   - Caption: “List, filter, and group assets by type, folder, or tag.”

3. **Bulk Editing**
   - Title: “Mass Updates in Seconds”
   - Caption: “Apply consistent tuning across hundreds of assets safely.”

4. **Dry Run & Safety**
   - Title: “Preview Changes Before They Land”
   - Caption: “Diffs, validation, and rollback keep every change safe.”

5. **Local LLM Support**
   - Title: “Offline and Token-Free”
   - Caption: “Works with local models like LM Studio, Ollama, or custom servers.”

## Technical Whitepaper (Summary)

### Abstract
NYGHTSHADE Unreal Plugin provides a deterministic automation layer inside the Unreal Editor, enabling AI-driven systems to perform safe, large-scale asset operations without manual editor interaction or cloud dependencies.

### Problem
Unreal Engine projects scale beyond manual workflows. Traditional tooling is UI-driven and difficult to automate, while generic AI tools lack editor-level control and safety guarantees.

### Solution
NYGHTSHADE offers:
- A structured command interface for automation.
- Asset introspection with type-safe metadata extraction.
- Deterministic execution with validation, logging, and rollback.
- Offline AI compatibility for local or self-hosted models.

### Core Technical Concepts
- **Command Schema:** Strongly typed JSON commands with validation and dry-run support.
- **Safety Layer:** Read/write permissions, property whitelisting, and change logs.
- **Automation Engine:** Batch execution with deterministic results and editor-safe transactions.

### Impact
NYGHTSHADE enables faster iteration cycles, reduces human error, and provides a future-proof path for AI-assisted Unreal development.

## Enterprise Sales Deck (Slide-by-Slide)

1. **Title Slide**
   - “NYGHTSHADE — AI-Native Automation for Unreal Engine”

2. **Problem**
   - Large Unreal projects don’t scale manually.
   - Designers and engineers waste time on repetitive edits.

3. **Why Existing Tools Fail**
   - Editor Utility Widgets are manual and UI-driven.
   - Generic AI tools lack Unreal awareness.
   - Custom studio tools are expensive to build and maintain.

4. **Solution**
   - NYGHTSHADE provides deterministic automation inside the editor.
   - AI tools can safely inspect and modify projects at scale.

5. **How It Works**
   - Structured command interface
   - Validation before execution
   - Dry-run diffs and rollback

6. **Use Cases**
   - Balance passes
   - Content generation
   - Rapid prototyping
   - Nightly validation jobs

7. **Security & Compliance**
   - Offline-capable
   - Full audit trails
   - Permission controls

8. **ROI**
   - Hours of manual work reduced to minutes.
   - Fewer errors, faster iteration, lower tooling costs.

9. **Enterprise Tier**
   - Headless automation
   - CI-style pipelines
   - SLA and long-term support

10. **Closing**
    - “NYGHTSHADE turns Unreal Engine into an automation-ready platform.”

## Sample Repo Structure
```
NYGHTSHADEUnrealPlugin/
├─ Source/
│  ├─ NYGHTSHADECore/
│  ├─ NYGHTSHADEAutomation/
│  ├─ NYGHTSHADEBridge/
│  └─ NYGHTSHADELogging/
├─ Resources/
├─ Content/
├─ Docs/
│  ├─ README.md
│  ├─ Commands.md
│  ├─ Safety.md
│  └─ Examples/
├─ Scripts/
│  ├─ python/
│  └─ examples/
└─ NYGHTSHADEUnrealPlugin.uplugin
```
```
