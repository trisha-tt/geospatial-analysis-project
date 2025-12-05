#!/usr/bin/env bash
# run_all.sh - run every Python file in the `code/` directory
# Usage: ./run_all.sh [python-executable]
# Example: ./run_all.sh        # uses python3
#          ./run_all.sh python  # uses python

PY=${1:-python3}
FAILED=0

echo "Using Python executable: $PY"

for f in ./code/*.py; do
  if [ ! -f "$f" ]; then
    continue
  fi
  # Skip `model.py` so it's not executed as a script
  if [ "$(basename "$f")" = "model.py" ]; then
    echo
    echo "================================================================"
    echo "Skipping: $f"
    echo "----------------------------------------------------------------"
    continue
  fi
  echo
  echo "================================================================"
  echo "Running: $f"
  echo "----------------------------------------------------------------"
  "$PY" "$f"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo ">>> ERROR: $f exited with code $rc"
    FAILED=1
  else
    echo "<<< Completed: $f (exit code 0)"
  fi
done

echo
if [ $FAILED -ne 0 ]; then
  echo "One or more scripts failed. Exit code 1 returned."
  exit 1
else
  echo "All scripts completed successfully."
  exit 0
fi