"""Verify the immutable sharing inputs; no model fitting or file mutation."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path.relative_to(ROOT))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")


def main():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text("utf-8"))
    for relative, expected in manifest["unchanged_source_files"].items():
        verify(ROOT / relative, expected)
    for relative, source in manifest["relocated_evidence_files"].items():
        verify(ROOT / relative, source["sha256"])
    raw = json.loads((ROOT / "data/raw/manifest.json").read_text("utf-8"))
    for name, expected in raw["files"].items():
        verify(ROOT / "data/raw" / name, expected)
    audit = json.loads((ROOT / "data/audit/audit_report.json").read_text("utf-8"))
    for name, expected in audit["processed_sha256"].items():
        verify(ROOT / "data/audit" / name, expected)
    frozen = json.loads((ROOT / "protocols/ablation-htru2-fixedbudget-v1-20261005/freeze_manifest.json").read_text("utf-8"))
    for section in ("design_sha256", "inherited_sha256"):
        for relative, expected in frozen[section].items():
            verify(ROOT / relative, expected)
    print(json.dumps({"status": "passed", "source_files": len(manifest["unchanged_source_files"]),
                      "evidence_files": len(manifest["relocated_evidence_files"]),
                      "frozen_references": sum(len(frozen[s]) for s in ("design_sha256", "inherited_sha256")),
                      "model_fits": 0}, indent=2))


if __name__ == "__main__":
    main()
