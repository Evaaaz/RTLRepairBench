# Historical formal protocol snapshot

This directory preserves an anonymous representation of the two verifier
source files that produced the frozen public formal ledgers. Project-namespace
identifiers were replaced with neutral equivalents for double-blind review;
that mechanical substitution does not alter verifier behavior, but it does
change the source bytes. This is a noncanonical historical snapshot, not a
second supported verifier. The canonical verifier for new runs is
`code/benchmarks/formal_verify.py` and intentionally has a different protocol
fingerprint after security hardening.

The frozen rows retain the original recorded protocol fingerprint
`0ca7c6f3b5bcbc07e4a52563a5aea7ddf6fa271ea9b0b9c3270ae20774e2b59e`.
Run
`python3 artifacts/public/primary/artifacts/formal/historical_protocol_0ca7c6f3/verify_fingerprint.py`
from the standalone repository root to verify the released anonymous source
bytes, the tracked historical toolchain lock and amendment, and the frozen
container image ID against the anonymous-snapshot digest in
`fingerprint_manifest.json`. Because the namespace substitution changes bytes,
the script deliberately does not claim to reconstruct the original recorded
digest during double-blind review. The original source and byte-exact check can
be restored in the archival release after review. This check does not rerun a
proof or validate the current hardened verifier against the external
candidate/proof trees.

Never edit the frozen ledgers to carry the current fingerprint. A hardened
recheck must write a new ledger and compare call IDs, source hashes, statuses,
replay results, simulation results, denominators, and both protocol hashes.
