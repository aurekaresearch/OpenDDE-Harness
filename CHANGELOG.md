# Changelog

User-facing changes to OpenDDE Harness are documented here.

## [Unreleased]

No changes yet.

## [0.0.1] - 2026-09-09

**Introducing OpenDDE Harness - our first public preview!**

We're excited to share OpenDDE Harness, an open-source tool for agentic antibody
design. It brings LLM-guided sequence design and OpenDDE structure prediction into
one workflow, from preparing a target to reviewing candidate sequences and structures.

Describe your design task in natural language, review the plan, and follow the
results as the agent proposes, evaluates, and refines candidates. This first release
supports VHH, scFv, and paired VH/VL binders while preserving configured fixed residues.
A terminal UI guides setup and design, and a tracing dashboard lets you inspect
candidate structures, scores, and progress across iterations.

Alongside this release, we're sharing a
[technical report](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/assets/OpenDDE_harness_tech_report.pdf)
and [design examples](https://github.com/aurekaresearch/OpenDDE-Harness/tree/v0.0.1/docs/examples)
to introduce the approach and help you get started. Install from PyPI or source
using the [installation guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/README.md#installation),
then follow the [onboarding guide](https://github.com/aurekaresearch/OpenDDE-Harness/blob/v0.0.1/docs/onboarding.md)
to connect your LLM provider and compute service.

This is an early preview, and you may encounter bugs. Please
[open an issue](https://github.com/aurekaresearch/OpenDDE-Harness/issues) with your
feedback, reproduction steps, and `ddeharness doctor --json` output when relevant.
Thank you for trying OpenDDE Harness and helping us make it better!

[Unreleased]: https://github.com/aurekaresearch/OpenDDE-Harness/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/aurekaresearch/OpenDDE-Harness/releases/tag/v0.0.1
