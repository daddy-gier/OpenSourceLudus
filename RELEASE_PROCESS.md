# Release Process

This document describes how releases are created for Auto Claude.

## Versioning

**Version Format:**
```
X.Y.Z-beta.N   (e.g., 2.8.0-beta.1, 2.8.0-beta.2)
X.Y.Z-alpha.N  (e.g., 2.8.0-alpha.1)
X.Y.Z-rc.N     (e.g., 2.8.0-rc.1)
```

We follow Semantic Versioning:

- **MAJOR (X.0.0):** Breaking changes, incompatible API changes
- **MINOR (0.X.0):** New features, backwards compatible
- **PATCH (0.0.X):** Bug fixes, backwards compatible

## Beta Updates (User-Facing)

Users can opt into beta updates in **Settings → Updates → “Beta Updates”**.
When enabled, the app will check for and install beta versions. Users can
switch back to stable at any time.

## Release Flow

Auto Claude uses an automated release pipeline that ensures releases are
only published after all builds succeed.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RELEASE FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   develop branch                    main branch                              │
│   ──────────────                    ───────────                              │
│        │                                 │                                   │
│        │  1. bump-version.js             │                                   │
│        │     (creates commit)            │                                   │
│        │                                 │                                   │
│        ▼                                 │                                   │
│   ┌─────────┐                           │                                   │
│   │ v2.8.0  │  2. Create PR             │                                   │
│   │ commit  │ ────────────────────►     │                                   │
│   └─────────┘                           │                                   │
│                                          │                                   │
│                           3. Merge PR    ▼                                   │
│                                    ┌──────────┐                              │
│                                    │ v2.8.0   │                              │
│                                    │ on main  │                              │
│                                    └────┬─────┘                              │
│                                         │                                    │
│                     ┌───────────────────┴───────────────────┐               │
│                     │     GitHub Actions (automatic)         │               │
│                     ├───────────────────────────────────────┤               │
│                     │ 4. prepare-release.yml                 │               │
│                     │    - Detects version > latest tag      │               │
│                     │    - Creates tag v2.8.0                │               │
│                     │                                        │               │
│                     │ 5. release.yml (triggered by tag)      │               │
│                     │    - Builds macOS (Intel + ARM)        │               │
│                     │    - Builds Windows                    │               │
│                     │    - Builds Linux                      │               │
│                     │    - Generates changelog               │               │
│                     │    - Creates GitHub release            │               │
│                     │    - Updates README                    │               │
│                     └───────────────────────────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## For Maintainers: Creating a Release

### Step 1: Bump the Version

On your development branch (typically `develop` or a feature branch):

```bash
# Navigate to project root
cd /path/to/auto-claude

# Bump version (choose one)
node scripts/bump-version.js patch   # 2.7.1 -> 2.7.2 (bug fixes)
node scripts/bump-version.js minor   # 2.7.1 -> 2.8.0 (new features)
node scripts/bump-version.js major   # 2.7.1 -> 3.0.0 (breaking changes)
node scripts/bump-version.js 2.8.0   # Set specific version
```

This will:

- Update `apps/frontend/package.json`
- Update `package.json` (root)
- Update `apps/backend/__init__.py`
- Create a commit with message `chore: bump version to X.Y.Z`

### Step 2: Push and Create PR

```bash
# Push your branch
git push origin your-branch

# Create PR to main (via GitHub UI or gh CLI)
gh pr create --base main --title "Release v2.8.0"
```

### Step 3: Merge to Main

Once the PR is approved and merged to main, GitHub Actions will automatically:

- Detect the version bump (`prepare-release.yml`)
- Create a git tag (e.g., `v2.8.0`)
- Trigger the release workflow (`release.yml`)
- Build binaries for all platforms
- Generate changelog from merged PRs (using release-drafter)
- Scan binaries with VirusTotal
- Create GitHub release with all artifacts
- Update README with new version badge and download links

### Step 4: Verify

After merging, check:

- GitHub Actions - ensure all workflows pass
- Releases - verify release was created
- README - confirm version updated

## Hotfix Workflow

For urgent production fixes that can’t wait for the normal release cycle:

### 1. Create hotfix from main

```bash
git checkout main
git pull origin main
git checkout -b hotfix/150-critical-fix
```

### 2. Fix the issue

```bash
# ... make changes ...
git commit -m "hotfix: fix critical crash on startup"
```

### 3. Open PR to main (fast-track review)

```bash
gh pr create --base main --title "hotfix: fix critical crash on startup"
```

### 4. After merge to main, sync to develop

```bash
git checkout develop
git pull origin develop
git merge main
git push origin develop
```

> **Note:** Hotfixes branch FROM `main` and merge TO `main` first, then
> sync back to `develop` to keep branches aligned.

## Changelog Generation

Changelogs are automatically generated from merged PRs using Release Drafter.

### PR Labels for Changelog Categories

| Label(s) | Category |
|---------|----------|
| feature, enhancement | New Features |
| bug, fix | Bug Fixes |
| improvement, refactor | Improvements |
| documentation | Documentation |
| (any other) | Other Changes |

## Workflows

| Workflow | Trigger | Purpose |
|---------|---------|---------|
| prepare-release.yml | Push to main | Detects version bump, creates tag |
| release.yml | Tag v* pushed | Builds binaries, creates release |
| validate-version.yml | Tag v* pushed | Validates tag matches package.json |
| update-readme (in release.yml) | After release | Updates README with new version |

## Troubleshooting

### Release didn't trigger after merge

Check if version in `package.json` is greater than latest tag:

```bash
git tag -l 'v*' --sort=-version:refname | head -1
cat apps/frontend/package.json | grep version
```

Ensure the merge commit touched `package.json`:

```bash
git diff HEAD~1 --name-only | grep package.json
```

### Build failed after tag was created

- The release won't be published if builds fail.
- Fix the issue and create a new patch version.
- Don't reuse failed version numbers.

### README shows wrong version

README is only updated after successful release. If release failed, README
keeps the previous version (this is intentional).

## Manual Release (Emergency Only)

In rare cases where you need to bypass the automated flow:

```bash
git tag -a v2.8.0 -m "Release v2.8.0"
git push origin v2.8.0
```

> **Warning:** Only do this if you're certain the version in `package.json`
> matches the tag.

## Security

All releases are:

- Scanned with VirusTotal before publishing
- Include SHA256 checksums for verification
- Code-signed where applicable (macOS)
