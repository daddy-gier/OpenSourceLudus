#!/bin/bash
# ============================================
# Smart Deploy with Test Reporting + Auto-Rollback
# Nyghtshade-BraxtonConfig v1.0.0
# ============================================

PROJECT_NAME="Nyghtshade-BraxtonConfig"
VERSION="v1.0.0"
ZIP_FILE="${PROJECT_NAME}_${VERSION}.zip"
GIT_REMOTE_URL="<your-git-url>" # <-- Replace
PINIKO_API_KEY="<your-piniko-api-key>" # <-- Replace
PINIKO_PROJECT_ID="<your-piniko-project-id>" # <-- Replace

rollback_git() {
  echo "⏪ Rolling back Git..."
  git reset --hard HEAD~1
  git push origin main --force
  git tag -d "$VERSION"
  git push origin :refs/tags/"$VERSION"
}

rollback_piniko() {
  echo "⏪ Rolling back Piniko..."
  curl -X DELETE "https://api.piniko.com/v1/projects/${PINIKO_PROJECT_ID}/uploads/${VERSION}" \
    -H "Authorization: Bearer $PINIKO_API_KEY"
}

# === STEP 1: Unzip Project ===
echo "📂 Unzipping project..."
unzip -o "$ZIP_FILE" -d "$PROJECT_NAME"
cd "$PROJECT_NAME" || { echo "❌ Failed to enter project directory"; exit 1; }

# === STEP 2: Replace 'revolt' with 'BRAXTONCONFIG' ===
echo "🔄 Replacing all occurrences of 'revolt' with 'BRAXTONCONFIG'..."
grep -rl "revolt" . | xargs sed -i '' 's/revolt/BRAXTONCONFIG/g'

# === STEP 3: Run Tests with Detailed Reporting ===
if [ -d "Tests" ]; then
  echo "🧪 Running tests with detailed error reporting..."
  TEST_FAILED=0
  FAILURE_LOG="test_failures.log"
  > "$FAILURE_LOG"

  for test_file in Tests/*.sh; do
    if [ -f "$test_file" ]; then
      echo "⚡ Running $test_file..."
      bash -n "$test_file" # syntax check first
      bash "$test_file" 2>tmp_err.log
      EXIT_CODE=$?
      if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Test failed: $test_file (Exit code $EXIT_CODE)" | tee -a "$FAILURE_LOG"
        awk '{print " " $0}' tmp_err.log >> "$FAILURE_LOG"
        TEST_FAILED=1
      else
        echo "✅ Test passed: $test_file"
      fi
      rm -f tmp_err.log
    fi
  done

  if [ $TEST_FAILED -ne 0 ]; then
    echo "❌ One or more tests failed. Deployment aborted!"
    cat "$FAILURE_LOG"
    exit 1
  fi
else
  echo "⚠️ No Tests directory found, skipping tests."
fi

# === STEP 4: Initialize Git Repo ===
echo "🗂 Initializing Git repository..."
git init
git add .
git commit -m "Initial commit: ${PROJECT_NAME} ${VERSION}"
git branch -M main
git remote add origin "$GIT_REMOTE_URL"

# === STEP 5: Push to Git ===
echo "🚀 Pushing to Git repository..."
if ! git push -u origin main; then
  echo "❌ Git push failed! Aborting."
  exit 1
fi

git tag -a "$VERSION" -m "Release ${VERSION} of ${PROJECT_NAME}"
if ! git push origin "$VERSION"; then
  echo "❌ Git tag push failed! Rolling back..."
  rollback_git
  exit 1
fi

# === STEP 6: Upload to Piniko ===
echo "📤 Publishing to Piniko..."
PINIKO_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "https://api.piniko.com/v1/projects/${PINIKO_PROJECT_ID}/uploads" \
  -H "Authorization: Bearer $PINIKO_API_KEY" \
  -F "file=@../$ZIP_FILE" \
  -F "version=$VERSION" \
  -F "notes=Release ${VERSION} of ${PROJECT_NAME}")

if [ "$PINIKO_RESPONSE" -ne 200 ]; then
  echo "❌ Piniko upload failed! Rolling back Git..."
  rollback_git
  rollback_piniko
  exit 1
fi

echo "✅ Smart deploy complete! All tests passed, Git & Piniko updated safely."
