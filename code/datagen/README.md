# Dataset pipeline

`datagen` is the canonical implementation of dataset construction. The old
copies at `code/` were removed.

```text
clean SystemVerilog seeds
        |
        +--> lint mutation --> Verilator fail/pass gate --------+
        |                                                       |
        +--> semantic mutation --> lint gate --> differential sim+
                                                                |
held-out VerilogEval/RTLLM fingerprints -------------------------+
                                                                v
                                                decontaminated repair JSONL
```

For user-provided clean seeds:

```bash
SEEDS=/absolute/path/to/seeds \
OUT=/absolute/path/to/new-run \
bash code/datagen/build_dataset.sh
```

For the historical-style source mix, also provide pinned local snapshots:

```bash
MG=/absolute/path/to/mg-verilog \
RAW=/absolute/path/to/verilogeval-and-rtllm \
OUT=/absolute/path/to/new-run \
bash code/datagen/build_dataset.sh
```

The core pipeline uses the Python standard library. `datasets` is required only
to import an MG-Verilog `load_from_disk` snapshot. System tools `verilator`,
`iverilog`, and `vvp` are mandatory for their respective gates.

Scientific runs fail closed. A missing tool, empty seed/certified output,
failed simulator, missing verdict, or missing/empty held-out reference stops
the run. Outputs are deterministically ordered. Diagnostic escape hatches are
for debugging only and must not be labeled as paper-valid.

The historical upstream snapshots and final 5,784-row training file are not in
this release. A fresh successful build is therefore a new certified dataset,
not a claim of byte identity with that historical corpus; see `claims.json`.
