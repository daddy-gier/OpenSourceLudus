#!/bin/bash
# ============================================
# Nyghtshade-BraxtonConfig v1.0.0 Git + Piniko Setup
# ============================================

# === CONFIGURE THESE ===
PROJECT_NAME="Nyghtshade-BraxtonConfig"
VERSION="v1.0.0"
ZIP_FILE="${PROJECT_NAME}_${VERSION}.zip"
GIT_REMOTE_URL="<your-git-url>" # <-- Replace with your Git repo URL
PINIKO_API_KEY="<your-piniko-api-key>" # <-- Replace with your Piniko API key
PINIKO_PROJECT_ID="<your-piniko-project-id>" # <-- Replace with your Piniko project ID

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

# === STEP 5: Add remote and push ===
git remote add origin "$GIT_REMOTE_URL"
echo "Pushing to Git repository..."
git push -u origin main
git tag -a "$VERSION" -m "Release ${VERSION} of ${PROJECT_NAME}"
git push origin "$VERSION"

# === STEP 6: Publish to Piniko ===
echo "Publishing to Piniko..."
curl -X POST "https://api.piniko.com/v1/projects/${PINIKO_PROJECT_ID}/uploads" \
  -H "Authorization: Bearer $PINIKO_API_KEY" \
  -F "file=@../$ZIP_FILE" \
  -F "version=$VERSION" \
  -F "notes=Release ${VERSION} of ${PROJECT_NAME}"

echo "✅ Nyghtshade-BraxtonConfig v1.0.0 successfully deployed to Git and Piniko!"
