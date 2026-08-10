#!/usr/bin/env bash
# Fetch the raw NASA CMAPSS turbofan degradation dataset (train/test/RUL for FD001-FD004).
set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/CMAPSSData"
FILES=(
  train_FD001.txt train_FD002.txt train_FD003.txt train_FD004.txt
  test_FD001.txt  test_FD002.txt  test_FD003.txt  test_FD004.txt
  RUL_FD001.txt   RUL_FD002.txt   RUL_FD003.txt   RUL_FD004.txt
  readme.txt
)

for f in "${FILES[@]}"; do
  curl -sf --max-time 30 -o "$f" "$BASE/$f" && echo "OK   $f" || { echo "FAIL $f"; exit 1; }
done

echo "Downloaded $(ls "$DEST"/*.txt | wc -l) files to $DEST"
