#!/bin/bash
# Inject lightweight active task reminder on each user prompt.
# Keeps Claude aware of what task is in progress without being verbose.

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ACTIVE_DIR="$PROJECT_DIR/.agents/dev_docs/active"

if [ -d "$ACTIVE_DIR" ] && [ "$(ls -A "$ACTIVE_DIR" 2>/dev/null)" ]; then
    tasks=""
    for task_dir in "$ACTIVE_DIR"/*/; do
        [ -d "$task_dir" ] || continue
        task_name=$(basename "$task_dir")

        # Get incomplete items count
        tasks_file="$task_dir"/*-tasks.md
        if ls $tasks_file 1>/dev/null 2>&1; then
            total=$(grep -c '^\- \[' $tasks_file 2>/dev/null || echo 0)
            done=$(grep -c '^\- \[x\]' $tasks_file 2>/dev/null || echo 0)
            tasks="$tasks $task_name ($done/$total done)"
        else
            tasks="$tasks $task_name"
        fi
    done
    jq -n --arg t "$tasks" '{
      "additionalContext": ("Active task:" + $t + ". Dev docs: .agents/dev_docs/active/. Update task checklist and context as you work. When task is complete: update decisions.md, then move task dir to archive/.")
    }'
else
    echo '{}'
fi
