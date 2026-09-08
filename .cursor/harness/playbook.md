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

- id: pypi-oidc
  desc: "0.1.0 publishes via .github/workflows/publish.yml on GitHub release; PyPI pending publisher owner=astroai repo=weightmask workflow=publish.yml environment=pypi."
