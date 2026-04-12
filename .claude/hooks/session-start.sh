#!/bin/bash
# Inject active task context at session start so Claude picks up where we left off.

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DEV_DOCS="$PROJECT_DIR/.agents/dev_docs"
ACTIVE_DIR="$DEV_DOCS/active"

output=""

# Check for active tasks
if [ -d "$ACTIVE_DIR" ] && [ "$(ls -A "$ACTIVE_DIR" 2>/dev/null)" ]; then
    active_tasks=""
    for task_dir in "$ACTIVE_DIR"/*/; do
        [ -d "$task_dir" ] || continue
        task_name=$(basename "$task_dir")
        active_tasks="$active_tasks\n## Active: $task_name"

        # Include plan summary
        plan_file="$task_dir"/*-plan.md
        if ls $plan_file 1>/dev/null 2>&1; then
            plan_content=$(head -20 $plan_file)
            active_tasks="$active_tasks\n### Plan\n$plan_content"
        fi

        # Include task checklist
        tasks_file="$task_dir"/*-tasks.md
        if ls $tasks_file 1>/dev/null 2>&1; then
            tasks_content=$(cat $tasks_file)
            active_tasks="$active_tasks\n### Tasks\n$tasks_content"
        fi

        # Include context
        context_file="$task_dir"/*-context.md
        if ls $context_file 1>/dev/null 2>&1; then
            context_content=$(cat $context_file)
            active_tasks="$active_tasks\n### Context\n$context_content"
        fi
    done

    output=$(printf '%b' "$active_tasks")
else
    output="No active tasks. Check .agents/dev_docs/README.md for the task roadmap."
fi

# Also show recent decisions for continuity
if [ -f "$DEV_DOCS/decisions.md" ]; then
    recent_decisions=$(tail -20 "$DEV_DOCS/decisions.md")
    output="$output

## Recent Decisions
$recent_decisions"
fi

# Return as JSON with additionalContext
jq -n --arg ctx "$output" '{
  "additionalContext": ("DEV DOCS CONTEXT:\n" + $ctx)
}'
