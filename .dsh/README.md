# weightmask × DeepSeek Harness (dsh)

Repo-local [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness)
configuration. dsh auto-discovers skills from `.dsh/skills/` at the project
root (top project priority).

> dsh is a **developer preview** — expect breaking changes. Requires a DeepSeek
> API key, or any OpenAI-compatible endpoint via `DEEPSEEK_BASE_URL`.

## Run

Web UI (needs localhost):

```bash
npx -y @deepseek-ai/dsh -- web --patch .dsh/cordis.patch.yml
```

Headless one-shot:

```bash
export DEEPSEEK_API_KEY=sk-...
npx -y @deepseek-ai/dsh -- --profile headless --patch .dsh/cordis.patch.yml \
  "Run the synthetic benchmark suite vs baselines and check defect-recovery statistics"
```

> The `--` after the package name passes dsh flags through npm's argument
> parser (without it, npm ≥ 10 swallows `--profile`/`--patch`).

Environment knobs:

- `DEEPSEEK_API_KEY` — required.
- `DEEPSEEK_BASE_URL` — OpenAI-compatible proxy.
- `DSH_PERMISSION_MODE=danger-full-access` — for one-shot headless runs on a
  disposable VM (default `workspace-write` + ask). Never on a shared machine.
- `DSH_TOOLS_MODE=code` — Code Mode.

## Skills

- `science-core` — shared correctness / math / statistics / data-driven / science-impact discipline
- `weightmask-dev` — YAML config surface, quality-bit contract, streak detector, benchmarks

## Notes

- Synthetic benchmarks (`--with-baselines`) are the statistics-principled gate
  for mask quality — injected defects must be recovered at stated rates.
- Sessions are append-only logs (provenance); reuse a session id to continue a
  durable conversation with persistent bash state.
- `cordis.patch.yml`: 10-minute bash timeout + durable full-text session search.
