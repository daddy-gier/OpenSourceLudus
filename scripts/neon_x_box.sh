#!/bin/bash
# ============================================
# Neon X Box Message Display
# ============================================

# === COLORS ===
PURPLE='\033[0;35m'
NEON_GREEN='\033[1;32m'
NC='\033[0m'

# === FUNCTION TO PRINT IN NEON X BOX ===
print_neon_box() {
  local msg="$1"
  local padding=2
  local length=${#msg}
  local width=$((length + padding * 2))

  # Top border
  echo -e "${PURPLE}X$(printf '%.0sX' $(seq 1 $width))X${NC}"

  # Empty line
  echo -e "${PURPLE}X${NC}$(printf ' %.0s' $(seq 1 $width))${PURPLE}X${NC}"

  # Message line
  echo -e "${PURPLE}X${NC}$(printf ' %.0s' $padding)${NEON_GREEN}${msg}${NC}$(printf ' %.0s' $padding)${PURPLE}X${NC}"

  # Empty line
  echo -e "${PURPLE}X${NC}$(printf ' %.0s' $(seq 1 $width))${PURPLE}X${NC}"

  # Bottom border
  echo -e "${PURPLE}X$(printf '%.0sX' $(seq 1 $width))X${NC}"
}

# === USAGE ===
print_neon_box "BRAXTONCONFIG Deployment SUCCESS!"
print_neon_box "All tests passed ✅"
print_neon_box "Git push complete 🚀"
