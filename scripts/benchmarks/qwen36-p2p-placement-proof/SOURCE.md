# Host-specific placement comparison tools

These are the exact guarded driver and CPU tests used for the placement
comparison. They record this host's GPU UUIDs, service names, paths, and build
receipts; they are evidence rather than a portable installer or a change to
production behavior. The related request and comparison drivers are archived
in `../qwen36-p2p-ab-proof/measurement/`.

The source candidate was the separately pushed integration commit
`76dec8a61d040efa97beefe273bed8dc9e68ff57`. The full comparison and smoke ran
with 43 passing local CPU guard/driver tests, a same-input smoke receipt, the
existing shared GPU lease, and a 1 GiB sampled free-VRAM floor. It is a single
control-then-candidate run, not production qualification.
