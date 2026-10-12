# Scope of the sharing snapshot

Prepared on 12 October 2026 from `thesis_htru2_20261001`.

## Preservation

Existing algorithm modules, configurations, selected runners, tests, protocols and required data were copied byte for byte. `SOURCE_MANIFEST.json` records their SHA-256 values. The original project, stop marker, experimental state and fit ledger are retained separately and unchanged.

New files are the root README, this scope note, `METHOD_MAP.md`, `REPRODUCING.md`, data/evidence READMEs, `scripts/quickstart.py`, `scripts/verify_share.py`, `tests/__init__.py`, `.gitignore`, `.gitattributes`, `.python-version`, `STOP_ABLATION` and `SOURCE_MANIFEST.json`. The sharing stop marker is a new guard; the original marker was not removed. The selected evidence files are relocated byte-preserving copies with their original relative paths recorded in the manifest.

## Included historical inputs

- Baseline manifest and five-fold split plan, used by the frozen configurations.
- Joint fold-0 decisions, one saved selector and its four contribution games, used by regression tests.
- Original ablation implementation-gate record and frozen design.
- Dated ablation aggregate tables and the archived independent aggregate-check report.

The partial `runs/` directories are reference fixtures, not complete experimental runs. They are kept at their historical relative paths to preserve the source tests and frozen hashes. The manifest for an included run may refer to artifacts not included here.

## Excluded material

The virtual environment, caches, local account information, personal working notes, quota/checkpoint orchestration scripts, thesis drafts, generated Office/PDF reports, full model archives and full fit-budget database are not part of this repository. Historical COMPAS/Adult/WDBC projects are excluded from HTRU2 evidence.

The archived independent audit report records a check performed against the complete local experiment. Its presence does not mean this curated repository alone can repeat that entire audit. Full archival replay requires the complete matching artifacts and environment. No new thesis experiment was launched to prepare this repository.

## Changes after sharing

Use a new Git commit for each intentional revision. Preserve the dated evidence and frozen configurations. A changed algorithm/configuration requires a new run identity and appropriate validation; do not overwrite historical results or regenerate their hashes to make a changed file appear original.
