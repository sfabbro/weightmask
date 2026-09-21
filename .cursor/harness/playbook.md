# Harness Playbook

## Bullets

- id: verify-tiers
  desc: verify_fast during edits; verify (ci-local) before push; verify_full only when human opts in.

- id: file-memory
  desc: Put durable notes in .cursor/harness/ — not long chat scrollback.

- id: minimal-diff
  desc: Smallest correct change; match existing repo style and tools.

- id: sky-mesh-sep
  desc: Sky mesh uses SEP n=(size-1)//box+1, nodes clip(rint((k+0.5)*box)) on encode and decode; reconstruct via CubicSpline (not map_coordinates).

- id: reconstruct-sky-cli
  desc: Rebuild mesh skies with weightmask-reconstruct-sky (module weightmask.reconstruct_sky); thin `weightmask reconstruct-sky` compat dispatch only.

- id: user-docs-v01
  desc: User path is README + docs/{installation,usage,algorithms,api}; YAML comments are the key list; research notes live in docs/research/ and are not advertised.

- id: elixir-megacam-only
  desc: Elixir is MegaCam-only. User docs, changelog, YAML comments, and CLI help must not brand the generic F² weight or keep-map as Elixir.

- id: pypi-oidc
  desc: "0.1.0 publishes via .github/workflows/publish.yml on GitHub release; PyPI pending publisher owner=astroai repo=weightmask workflow=publish.yml environment=pypi."

- id: science-gate
  desc: "pixi run science-gate is opt-in MegaCam evidence; a data skip is not a pass and must not be added to verify or verify_full."

- id: variance-frozen
  desc: "yaml variance.flat_fielded_poisson is on: a vignetted weighted aperture moved by more than the fixture read-noise floor. The in-code fallback without that key stays g²F²/(Sg+RN²)."

- id: streak-fp-prior
  desc: "Static columns need other exposures of the same CCD (persistent_axis_mask); a single-file run does not claim the 0.10 artefact-FP gate."

- id: mrt-bin
  desc: "mrt_rescue_params.bin stays 1. The 2026-09-21 science-gate on 1013719p HDU 1 was continuous recall 0.552 and recall5 0.600 at bin 1, so bin 4 was not enabled."

- id: prescreen-skip
  desc: "skip_when_prescreen_confirmed stays true. Disabling it on 1013719p HDU 1 added ~29 s and satdet returned 0 candidates; the missed trails stayed at recall 0. profile_accept cut fp5 4137 to 1359 with recall unchanged."
