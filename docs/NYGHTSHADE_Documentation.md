# NYGHTSHADE Unreal Plugin — Documentation

## Overview
NYGHTSHADE Unreal Plugin is an Unreal Engine editor plugin that enables AI-driven automation, large-scale asset manipulation, and structured project introspection through a deterministic, machine-readable interface.

NYGHTSHADE allows AI tools such as Cursor, local LLMs, Python agents, and offline models to safely inspect, modify, generate, and rebalance Unreal projects without manual editor interaction.

## Core Principles
- AI-first, editor-native automation
- Deterministic and auditable execution
- Fully local & offline compatible
- No cloud APIs or subscriptions
- Safe by design (validation, logging, rollback)

## Installation
1. Copy the plugin folder:

```
NYGHTSHADEUnrealPlugin/
```

2. Into your project:

```
YourProject/Plugins/
```

3. Enable **NYGHTSHADE Unreal Plugin** in the Unreal Editor.
4. Restart the editor.

## Architecture
```
NYGHTSHADEUnrealPlugin/
├─ Core/
│  ├─ AssetIntrospection
│  ├─ CommandExecution
│  ├─ Validation
│
├─ AIBridge/
│  ├─ JSON Command Interface
│  ├─ Local Socket / File IO
│
├─ Logging/
│  ├─ Change Tracking
│  ├─ Rollback Support
```

## AI Integration Model
NYGHTSHADE exposes structured JSON commands that any AI system can generate.

```json
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

### Guarantees
- Strongly typed
- Validated before execution
- Optional dry-run mode
- Fully logged and reversible

## Safety Model
- Read-only vs write operations
- Property whitelisting
- Dry-run diffs
- Full change logs
- Undo / rollback support

AI can act — but never blindly.

## Typical Use Cases
- Weapon and gameplay balancing
- AI behavior tuning
- NPC stat normalization
- Variant generation
- Large-scale refactors
- Rapid prototyping pipelines

## License & Support
- Editor-only plugin
- No runtime dependency
- Marketplace-ready
- Indie, Pro, and Studio tiers available

## Launch Announcement
NYGHTSHADE Unreal Plugin is now available.

NYGHTSHADE connects Unreal Engine directly to AI-driven automation systems, enabling Cursor, local LLMs, and offline agents to safely inspect and modify Unreal projects at scale.

No cloud APIs.
No subscriptions.
No editor micromanagement.

Instead of opening hundreds of assets and tweaking properties by hand, NYGHTSHADE gives AI a deterministic, auditable interface into the Unreal Editor.

If you’ve ever thought:

“AI should be doing this instead of me.”

NYGHTSHADE is the missing link.

⚠️ Early pricing is temporary.
More automation features = higher price.

## Pricing Tiers
### 🟢 Indie License
$5 (Early) / $5

- Full core plugin
- Local & offline AI support
- Asset introspection
- Bulk editing
- Logging & rollback
- Lifetime v1.x updates

Best for: Solo devs, indie teams

### 🔵 Pro License
$20 / seat

- Everything in Indie
- Advanced automation commands
- Conditional workflows
- Command presets
- Priority updates

Best for: Small teams, serious production

### 🟣 Studio / Enterprise
Custom pricing

- Multi-user permissions
- Full audit trails
- CI-style Unreal automation
- Headless editor workflows
- Long-term support builds
- Direct onboarding & support

## Demo Script
**“AI Rebalances an Entire Weapon System”**

**Scene 1 — Problem**
“This project has 120 weapon assets. Manual balancing takes hours.”

**Scene 2 — AI Analysis**
- Cursor scans project
- Identifies weapon assets
- Groups by class

**Scene 3 — Dry Run**
```json
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

**Scene 4 — Review**
- Show diff
- Show validation
- Confirm execution

**Scene 5 — Result**
“120 assets updated in seconds. Logged. Reversible. Safe.”

**Closing**
“This is AI as a real Unreal Engine assistant.”

## Cursor Prompt Pack (NYGHTSHADE)
**Prompt 1 — Project Discovery**
Inspect this Unreal project using NYGHTSHADE.
List all major asset categories and summarize their purpose.
Output structured JSON.

**Prompt 2 — Balance Proposal**
Analyze weapon assets.
Propose balance changes to normalize DPS while preserving identity.
Do not execute changes.

**Prompt 3 — Safe Execution**
Generate a NYGHTSHADE command to apply the approved changes.
Use dry-run mode first.

**Prompt 4 — Variant Generation**
Create three weapon variants per base weapon.
Follow existing naming conventions.

**Prompt 5 — Audit & Cleanup**
Scan the project for inconsistent or missing gameplay values.
Generate a report and suggest automated fixes.
