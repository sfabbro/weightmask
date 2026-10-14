# Harness Playbook

## Bullets

- id: verify-tiers
  desc: verify_fast during edits; verify (pixi run test) before push; ci-local for a clean CI snapshot; verify_full only when human opts in.

- id: file-memory
  desc: Put durable notes in .cursor/harness/ — not long chat scrollback.

- id: minimal-diff
  desc: Smallest correct change; match existing repo style and tools.

- id: sky-mesh-sep
  desc: Sky mesh uses SEP n=(size-1)//box+1, nodes clip(rint((k+0.5)*box)) on encode and decode; reconstruct via CubicSpline (not map_coordinates).

- id: reconstruct-sky-cli
  desc: Rebuild mesh skies with the dedicated weightmask-reconstruct-sky command (module weightmask.reconstruct_sky).

- id: user-docs
  desc: User path is README + docs/{installation,usage,algorithms,api,releasing}; YAML comments are the key list; research notes live in docs/research/ and are not advertised.

- id: elixir-megacam-only
  desc: Elixir is MegaCam-only. User docs, changelog, YAML comments, and CLI help must not brand the generic F² weight or keep-map as Elixir.

- id: pypi-oidc
  desc: "Releases validate via pixi run release-check and publish via .github/workflows/release.yml (workflow_dispatch only, dry_run=true default); PyPI trusted publisher owner=astroai repo=weightmask workflow=release.yml environment=pypi."

- id: science-gate
  desc: "pixi run science-gate is opt-in MegaCam evidence; a data skip is not a pass and must not be added to verify or verify_full."

- id: variance-frozen
  desc: "Theoretical variance always uses g²F²/(SgF+RN²); a vignetted weighted aperture moved by more than the fixture read-noise floor with the old denominator."

- id: streak-fp-prior
  desc: "Static columns need other exposures of the same CCD (persistent_axis_mask); a single-file run does not claim the 0.10 artefact-FP gate."
