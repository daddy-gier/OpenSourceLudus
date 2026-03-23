#!/bin/bash
# ============================================
# Nyghtshade-BraxtonConfig v1.0.0 Git Setup
# ============================================

# === CONFIGURE THESE ===
PROJECT_NAME="Nyghtshade-BraxtonConfig"
VERSION="v1.0.0"
REMOTE_URL="<your-git-url>" # <-- Replace this with your Git repo URL
ZIP_FILE="${PROJECT_NAME}_${VERSION}.zip"

# === STEP 1: Unzip project ===
echo "Unzipping project..."
unzip -o "$ZIP_FILE" -d "$PROJECT_NAME"
cd "$PROJECT_NAME" || exit

# === STEP 2: Replace 'revolt' with 'BRAXTONCONFIG' ===
echo "Replacing all occurrences of 'revolt' with 'BRAXTONCONFIG'..."
grep -rl "revolt" . | xargs sed -i '' 's/revolt/BRAXTONCONFIG/g'

# === STEP 3: Initialize Git repo ===
echo "Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: ${PROJECT_NAME} ${VERSION}"

# === STEP 4: Ensure main branch ===
git branch -M main

# === STEP 5: Add remote ===
git remote add origin "$REMOTE_URL"

# === STEP 6: Push to Git ===
echo "Pushing to remote repository..."
git push -u origin main

# === STEP 7: Tag release ===
git tag -a "$VERSION" -m "Release ${VERSION} of ${PROJECT_NAME}"
git push origin "$VERSION"

echo "✅ Nyghtshade-BraxtonConfig v1.0.0 is fully set up on Git!"
