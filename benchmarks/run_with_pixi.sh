#!/usr/bin/env bash
# Locate pixi and run a single pixi task, so git hooks work from any shell.
#
# pre-commit runs hook entries through a non-login, non-interactive `bash -lc`,
# where ~/.pixi/bin is usually not on PATH. Without this wrapper, a fresh clone
# that only ran `pre-commit install` gets a bare "pixi: command not found" and
# exit 127 on its first push, with nothing pointing at the cause.
#
# Usage: benchmarks/run_with_pixi.sh TASK [ARGS...]
set -uo pipefail

if command -v pixi > /dev/null 2>&1; then
  exec pixi run "$@"
fi

pixi_home_candidate="${HOME:-}"
if [ -n "$pixi_home_candidate" ] && [ -x "$pixi_home_candidate/.pixi/bin/pixi" ]; then
  exec "$pixi_home_candidate/.pixi/bin/pixi" run "$@"
fi

cat >&2 <<'EOF'
ERROR: pixi is not installed and ~/.pixi/bin/pixi does not exist.

The pre-push hooks and every pixi run task in this repo require pixi.
Install it with:

  curl -fsSL https://pixi.sh/install.sh | bash

then re-run, or push with --no-verify if you intend to skip the local
parity checks deliberately (GitHub CI will still run them).
EOF
exit 127
