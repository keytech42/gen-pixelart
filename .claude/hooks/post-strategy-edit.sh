#!/bin/bash
# After editing strategy or model files, remind to run smoke test.

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.command // empty')

# Check if the edited file is in strategies/ or models/
if echo "$FILE_PATH" | grep -qE 'src/(strategies|models)/'; then
    jq -n '{
      "additionalContext": "REMINDER: Strategy or model code was modified. Run `uv run python scripts/smoke_test.py` to verify the strategy contract before moving on."
    }'
else
    echo '{}'
fi
