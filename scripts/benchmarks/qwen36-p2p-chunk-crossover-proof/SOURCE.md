# Host-specific 512-first P2P comparison tools

These are the exact guarded driver and CPU tests used for the reverse-order
served-model comparison. They contain this host's GPU UUIDs, service names,
paths and build receipts; they are evidence, not a portable installer or a
production change. The request and comparison drivers are archived in
`../qwen36-p2p-ab-proof/measurement/`.

Both arms used the same server from integration commit
`76dec8a61d040efa97beefe273bed8dc9e68ff57`. The 512-row arm ran first and
the 256-row arm second. The runner bound the earlier physical-peer parity,
256-row memory reference and 512-row served-model admission smoke, and
monitored a 1 GiB free-VRAM floor. It passed 46 local CPU safety checks.
Compare it with the control-first run; neither is a long-context or
production qualification.
