# AGENTS.md — weightmask

## Remotes (AstroAI fork workflow)

| Remote | Points at | Use |
|--------|-----------|-----|
| `origin` | `sfabbro/weightmask` | Push `wip/*` only |
| `upstream` | `astroai/weightmask` | Sync `main`; PR target |

`main` tracks `upstream/main`. Never force-push `astroai` `main`.

```bash
git fetch upstream && git rebase upstream/main
git checkout -b wip/<topic>
git push -u origin HEAD
gh pr create -R astroai/weightmask --head sfabbro:$(git branch --show-current)
```

Workspace layout, CANFAR, and `/arc` install rules: parent workspace `AGENTS.md` (`~/src/AGENTS.md`).

## Environment

Pixi only. Do not use system Python or bare pip when the Pixi env is available.

```bash
pixi install
```

## Verification

- Fast: `pixi run lint`
- Before push: `pixi run test`
- Opt-in: `pre-commit run --all-files`

Read `.cursor/harness/config.json` and the README for task-specific checks.
