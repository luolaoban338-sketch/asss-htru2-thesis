# Sharing validation — 12 October 2026

- Clean Python 3.12.14 virtual environment installed successfully from the pinned requirements using official PyPI.
- All 88 existing tests passed in the clean environment (Windows).
- The integrity check verified 86 unchanged source files, 11 relocated evidence files and 25 frozen protocol references.
- The no-fitting synthetic quick-start demonstration passed.
- The documented bounded baseline command completed one new fit in the sharing copy's ignored `runs/demo-baseline/` directory. This was an installation smoke check, not a new thesis result; those outputs are not published.
- The original project source files, stop marker, experiment state and fit-budget database remained byte-identical.
- The selected text files were scanned for common token/private-key patterns, personal Windows paths and private email addresses; no matches were found. This is a focused packaging check, not a comprehensive security audit.

The original experiment environment remains separate. These checks establish the sharing snapshot's tested entry points, not cross-platform bitwise reproducibility or a fresh replay of the complete experiment archive.
