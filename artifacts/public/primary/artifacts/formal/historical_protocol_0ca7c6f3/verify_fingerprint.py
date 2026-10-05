#!/usr/bin/env python3
"""Verify the anonymous historical formal-protocol snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
MANIFEST = HERE / "fingerprint_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = {
        "formal_verify.py": HERE / "formal_verify.py",
        "formal_protocol.py": HERE / "formal_protocol.py",
        "docker/formal/toolchain.lock.json": ROOT / "docker/formal/toolchain.lock.json",
        "configs/vericodegen/protocol_amendment_2026-07-15.json": (
            ROOT / "configs/vericodegen/protocol_amendment_2026-07-15.json"
        ),
    }
    digest = hashlib.sha256()
    digest.update(manifest["protocol_version"].encode("utf-8"))
    for source in manifest["sources"]:
        path = paths[source["path"]]
        actual = sha256(path)
        if actual != source["sha256"]:
            raise RuntimeError(
                f"historical protocol component drift: {source['path']} "
                f"expected={source['sha256']} actual={actual}"
            )
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(manifest["container_image_id"].encode("utf-8"))
    actual_protocol = digest.hexdigest()
    expected_protocol = manifest["anonymized_snapshot_fingerprint_sha256"]
    if actual_protocol != expected_protocol:
        raise RuntimeError(
            "anonymous historical snapshot fingerprint mismatch: "
            f"expected={expected_protocol} actual={actual_protocol}"
        )
    print(
        json.dumps(
            {
                "anonymized_snapshot_fingerprint_sha256": actual_protocol,
                "recorded_formal_protocol_sha256": manifest[
                    "recorded_formal_protocol_sha256"
                ],
                "status": "anonymous_snapshot_verified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
