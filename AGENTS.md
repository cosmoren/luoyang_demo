# Agent Conventions

This file documents conventions that any AI coding agent (Cursor, Claude, etc.) MUST follow when working in this repo.

## Throwaway files go in `playground/`

When you need to create a quick script, plot, debug output, smoke test, or any one-off file that isn't intended to live in the real codebase, put it under:

```
playground/{YYYY-MM-DD}_{short-task-slug}/
```

Examples:
- `playground/2026-06-11_folsom-debug/check_kt_distribution.py`
- `playground/2026-06-12_animation-test/sample_frames/`
- `playground/2026-06-15_inference-sanity/preds.npz`

### Rules

1. **`playground/` is fully gitignored.** It is personal scratch space. Nothing in it ever reaches GitHub.
2. **One subfolder per session.** Name it with today's date plus a short task slug (kebab-case).
3. **Do NOT place experimental / throwaway files anywhere else.** Specifically, do NOT drop them in:
   - `scripts/`, `inference/`, `dataloader/`, `training/`, `models/`, `config/`, `SPMF_preprocessing/`, `modules/` — these are real code directories.
   - `.cursor/` — that folder is for Cursor IDE configuration only (rules, skills, settings). No images, no logs, no scripts.
   - The repo root.
4. **Promote to the real codebase only with explicit user confirmation.** If something in the playground turns out to be worth keeping, the user (or you, with the user's go-ahead) explicitly copies it into the right real directory and commits it. Never promote silently.
5. **Old sessions can be moved into `playground/_archive/`** to keep the top of `playground/` tidy. Same naming convention.

## Where things actually live

| Kind of file | Location |
|---|---|
| Real production scripts | `scripts/`, `inference/`, `training/`, etc. |
| Real config | `config/datasets/`, `config/train/` |
| Cursor IDE config / rules / skills | `.cursor/` (and ONLY this kind of file) |
| Throwaway / debug / one-off | `playground/{date}_{slug}/` |
| Recurring report outputs | `report_{date}/` (already gitignored via `report_*/`) |
| Weekly handoff bundles | `handoff_{date}_{label}/` (already gitignored via `handoff_*/`) |
