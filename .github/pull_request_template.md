## What this changes

<!-- One paragraph. The failure mode, not the diff. -->

## Why

<!-- What is wrong today, or what cannot be done. -->

## How it was verified

- [ ] `python -m pytest` passes
- [ ] `python -m ruff check .` passes
- [ ] `python examples/make_demo_dataset.py && python postprocess_pipeline/pipeline.py --config examples/demo.pipeline.json --execute` passes
- [ ] The demo still rejects `episode_000003` at the fusion gate

## Contract and data-safety checklist

- [ ] No shell execution was introduced (`shell=True`, string commands, `os.system`)
- [ ] Source episodes are still never deleted, moved, or renumbered
- [ ] Any new gate fails with a non-zero exit code or a `status: FAIL` artifact
- [ ] Any interpolated or prior-derived value is labelled as such in the artifact
- [ ] No private paths, hostnames, credentials, or unconsented recordings
- [ ] `docs/PIPELINE_STAGES.md` updated if a stage changed

## Licensing

<!-- New dependencies: name and license. Noncommercial or proprietary ones must stay optional. -->
