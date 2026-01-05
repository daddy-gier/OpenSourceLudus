#!/bin/bash
# ============================================
# Smart Deploy: Nyghtshade-BraxtonConfig v1.0.0
# ============================================

PROJECT_NAME="Nyghtshade-BraxtonConfig"
VERSION="v1.0.0"
ZIP_FILE="${PROJECT_NAME}_${VERSION}.zip"
GIT_REMOTE_URL="<your-git-url>" # <-- Replace
PINIKO_API_KEY="<your-piniko-api-key>" # <-- Replace
PINIKO_PROJECT_ID="<your-piniko-project-id>" # <-- Replace

# === STEP 1: Unzip Project ===
echo "Unzipping project..."
unzip -o "$ZIP_FILE" -d "$PROJECT_NAME"
cd "$PROJECT_NAME" || { echo "❌ Failed to enter project directory"; exit 1; }

# === STEP 2: Replace 'revolt' with 'BRAXTONCONFIG' ===
echo "Replacing all occurrences of 'revolt' with 'BRAXTONCONFIG'..."
grep -rl "revolt" . | xargs sed -i '' 's/revolt/BRAXTONCONFIG/g'

# === STEP 3: Run Tests ===
if [ -d "Tests" ]; then
  echo "Running tests..."
  TEST_FAILED=0
  for test_file in Tests/*.sh; do
    if [ -f "$test_file" ]; then
      bash "$test_file"
      if [ $? -ne 0 ]; then
        echo "❌ Test failed: $test_file"
        TEST_FAILED=1
      else
        echo "✅ Test passed: $test_file"
      fi
    fi
  done
  if [ $TEST_FAILED -ne 0 ]; then
    echo "❌ One or more tests failed. Aborting deployment."
    exit 1
  fi
else
  echo "⚠️ No Tests directory found, skipping tests."
fi

# === STEP 4: Initialize Git Repo ===
echo "Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: ${PROJECT_NAME} ${VERSION}"
git branch -M main
git remote add origin "$GIT_REMOTE_URL"

# === STEP 5: Push to Git ===
echo "Pushing to Git repository..."
git push -u origin main
git tag -a "$VERSION" -m "Release ${VERSION} of ${PROJECT_NAME}"
git push origin "$VERSION"

# === STEP 6: Upload to Piniko ===
echo "Publishing to Piniko..."
curl -X POST "https://api.piniko.com/v1/projects/${PINIKO_PROJECT_ID}/uploads" \
  -H "Authorization: Bearer $PINIKO_API_KEY" \
  -F "file=@../$ZIP_FILE" \
  -F "version=$VERSION" \
  -F "notes=Release ${VERSION} of ${PROJECT_NAME}"

echo "✅ Smart deploy complete! All tests passed, Git & Piniko updated."
