# Security policy

## Reporting

Report a vulnerability privately through GitHub's
[security advisory](https://github.com/fangdiyu/embodied-postprocess-pipeline/security/advisories/new)
rather than in a public issue. Expect an acknowledgement within a week.

## Scope

The pipeline executes local commands and reads local datasets. The most relevant
risk classes are:

**Command injection through configuration.** Stages are argv arrays and are
executed without a shell, so a path cannot become a command. A config is
nevertheless executable content — it names the programs to run. Treat a config
from an untrusted source the way you would treat a shell script: read it, or run
it with `--list` and without `--execute` first. Dry run is the default for
exactly this reason.

**Path traversal in manifests.** Manifest and audit tools resolve episode names
against `--source-root`. A manifest is data, not code, but a crafted
`valid_episodes` entry containing `..` segments would cause files outside the
source root to be inspected. Only use manifests produced by this project, or by
a source you trust.

**Accidental disclosure in published artifacts.** Provenance files, logs, and
manifests contain absolute paths. `build_release.py` scans the source tree for
home directories, IP literals, SSH targets, key blocks, and token formats before
archiving, but it cannot inspect the datasets or logs you publish separately.
Review before you upload.

**Model weights and third-party code.** Not distributed here. See
[`NOTICE`](NOTICE) and
[`THIRD_PARTY.md`](postprocess_pipeline/THIRD_PARTY.md).

## Not in scope

- Malicious datasets that cause a decoder to crash or exhaust memory. The pipeline
  runs locally on data you collected; it is not a sandbox.
- Vulnerabilities in external hand-pose, IK, or LeRobot backends. Report those
  upstream.
- Resource exhaustion from very large recordings. Guard rails are the operator's
  responsibility.

## Supported versions

The `main` branch and the latest tagged release receive fixes. This is a research
pipeline at `0.x`; there are no long-term support branches.
