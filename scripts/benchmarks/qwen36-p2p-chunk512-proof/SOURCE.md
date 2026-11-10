# Host-specific 512-row device parity tools

These are the exact guarded device runner and CPU tests used for the
512-row synthetic two-peer test. They contain this host's GPU UUIDs, service
names, paths and build receipts, so they are evidence rather than a portable
runner. The tested source commit was
`7142c14bf321238e9d5850a7afa640f7e26a6fb8`, whose only code changes from
`76dec8a61d040efa97beefe273bed8dc9e68ff57` were test-only.

The fixture has two model layers and a 2,048-token KV capacity. It compares
the final greedy token and top-two logits against the per-token path for a
512-row chunk and a two-chunk-plus-tail sequence; it does not compare every
logit, the full 40-layer model, a 128k/256k prompt, or throughput. The small
8-row smoke and 512-row run each held the shared lease and restored the
original service. The runner had 33 passing CPU safety checks.
