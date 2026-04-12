# Dev Docs

Session-persistent context for this project. Carries state between Claude Code sessions so each one can pick up where the last left off.

## Structure

```
dev_docs/
├── README.md          ← you are here
├── decisions.md       ← cumulative cross-task design decisions
├── active/            ← in-progress tasks
│   └── <task>/
│       ├── <task>-plan.md
│       ├── <task>-context.md
│       └── <task>-tasks.md
└── archive/           ← completed tasks (same structure)
```

## Task Lifecycle

### Starting a task

Create a directory under `active/` with three files:

**`<task>-plan.md`** — The accepted implementation plan.
- What we're building and why
- Key design choices
- Dependencies / prerequisites
- What "done" looks like (runnable outcome)

**`<task>-context.md`** — Living reference doc updated during work.
- Key files being touched
- Decisions made during implementation (also add to `decisions.md` if non-obvious)
- Blockers, open questions, things tried that didn't work
- Relevant MLflow experiment/run IDs

**`<task>-tasks.md`** — Checklist of work items.
- Granular enough to track progress
- Each item should be independently verifiable
- Mark items `[x]` as completed

### Completing a task

1. Verify the runnable outcome works
2. Update `decisions.md` with any non-obvious decisions made
3. Move the task directory from `active/` to `archive/`

## decisions.md

The project's institutional memory. Log non-obvious decisions here — things a future session would waste time rediscovering. Format:

```markdown
## Task N: Name
- **Decision title**: What was decided and why. Include what was rejected if it's not obvious.
```

## Task Roadmap

| # | Task | Scope | Runnable outcome |
|---|------|-------|------------------|
| 1 | skeleton | Project structure, ABC, config, dummy strategy, Trainer, MLflow | `train.py` runs with dummy strategy, logs to MLflow |
| 2 | data | Sprite dataset loader, palette extraction, augmentation | Loads real sprites, logs sample batch to MLflow |
| 3 | vae | First real strategy end-to-end | Generates blurry but recognizable sprites |
| 4 | vqvae | Second strategy, codebook mechanics | Two-strategy comparison on MLflow |
| 5 | diffusion | Noise scheduler, U-Net, sampling loop | Three-way comparison on MLflow |
| 6 | conditioning | Class labels or palette conditioning | Conditional generation |

Each task targets a runnable state. Each maps to roughly one Claude Code session.
