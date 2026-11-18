# Provenance of the `qwen36-p2p-*` records

Every `qwen36-p2p-*` file and directory here is a **measurement record
imported with the P2P pair-service port** (zbrdc PR stack). Each record's
`tested_source_commit` / `tested_source_url` names the exact commit of the
author's fork the numbers were taken on. They were produced by the PR author
on the author's own rig (a two-GPU peer pair, one card an RTX 3080), not by
this repository's device lanes, and none of the figures were re-measured
here.

What this repository re-validated for these records: container compile of
every series commit (native-cuda, `cargo test --no-run`), the host gates and
the exact-tip battery on this tree, and — where a card window admitted it —
the single-card device arms. The two-GPU numbers (ring/chunk throughput,
256K recall, A/B comparisons) remain author-rig observations until this
fleet runs its own pair, and the pair path stays opt-in and default off.

Do not cite these files as this fleet's measurements; cite them as imported
upstream evidence with the per-file commit pins.
