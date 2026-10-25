# Contributing

Thanks for considering it. This project exists because embodied-AI data
processing is under-documented, and the most useful contribution is usually a
new backend or a sharper quality gate.

## Good first contributions

- **A new hand-pose backend** (MediaPipe, MANO, RTMPose) implementing the
  contract in `postprocess_pipeline/ADAPTER_CONTRACTS.md`.
- **A new robot adapter** (Franka, UR, xArm) — an IK script plus a start
  contract example.
- **A new packer** (ACT, Diffusion Policy) — anything that writes LeRobot v3 and
  keeps normalization statistics in metadata.
- **A stage that fails loudly** where the current one is too permissive.
- **Documentation of a failure mode** you hit. The troubleshooting page is more
  useful with real reports in it.

## Development setup

```bash
git clone https://github.com/fangdiyu/embodied-postprocess-pipeline.git
cd embodied-postprocess-pipeline
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
```

Run the checks:

```bash
python -m pytest
python -m ruff check .
python examples/make_demo_dataset.py
python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json --execute
```

The demo is the fastest way to confirm your change did not break the chain. It
runs in a few seconds and needs no external backend.

## Rules the code follows

These are not stylistic preferences; they are the reason the pipeline is
trustworthy. A PR that breaks one of them will be asked to change.

1. **No shell execution.** Stages are argv arrays. Do not add `shell=True`, do not
   join a command into a string, do not `os.system`.
2. **Never mutate source data.** No deleting, moving, or renumbering episodes.
   Selection goes through manifests.
3. **Quality gates are gates.** A failed check must produce a non-zero exit code
   or a `status: FAIL` artifact that a later stage consumes. A warning in a log is
   not a gate.
4. **Weak signals stay labelled.** If a value is interpolated, inferred from a
   prior, or scale-ambiguous, say so in the artifact.
5. **Physical values stay physical.** Normalization statistics belong in dataset
   metadata, not baked into stored state/action arrays.
6. **Deterministic where it matters.** No wall-clock or RNG dependence in a
   result that feeds a gate. Seeds and timestamps go in artifacts, not in
   decisions.
7. **No private data.** No absolute home paths, server addresses, serial numbers,
   credentials, or recordings without consent. `build_release.py` scans for these
   shapes; CI runs the same patterns.

## Adding an adapter

1. Read `ADAPTER_CONTRACTS.md` and pick the interface you are implementing.
2. Write a standalone script — an executable command, not a library import.
3. Put it in `adapters/`, name it after what it does, and keep its CLI flags
   self-documenting (`--help` must work without any optional dependency
   installed).
4. Resolve optional heavy imports lazily, inside `main()` or a loader, so `--help`
   never crashes.
5. Emit a `status: PASS|FAIL` field plus the numeric evidence for that decision.
6. Add a stage to an example config, and a test that runs it on synthetic data.
7. Update `docs/PIPELINE_STAGES.md` and `THIRD_PARTY.md` if you introduce a new
   dependency.

## Pull requests

- One logical change per PR. Refactors and behaviour changes in separate PRs.
- Describe the failure mode you are fixing, not just the diff.
- Include the output of the demo run if you touched the orchestration or a tool.
- New adapters: state the license of everything you link against.

## Reporting a bug

Use the issue template. The two things that make a report actionable are the
`provenance.json` from the failed run (it contains the exact argv, resolved
variables, and interpreter) and the tail of the stage log.

Redact paths and hostnames before pasting.

## License

By contributing you agree your contribution is licensed under LGPL-3.0, the same
terms as the project. Do not paste code from a project with an incompatible
license — in particular, do not copy from a noncommercial or proprietary
codebase into `adapters/` or `tools/`.
