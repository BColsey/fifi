# Host-specific 256-versus-512 P2P comparison tools

These are the exact guarded driver and CPU tests used for the served-model
comparison. They record this host's GPU UUIDs, service names, paths and build
receipts; they are evidence, not a portable installer or production change.
The request and comparison drivers are archived in
`../qwen36-p2p-ab-proof/measurement/`.

Both arms used the same server built from integration commit
`76dec8a61d040efa97beefe273bed8dc9e68ff57`, with only `Q36_P2P_CHUNK`
changed. The runner bound a 512-row physical-peer parity receipt and an
existing 256-row memory reference, required a short served-model 512-row
admission smoke, and monitored a 1 GiB free-VRAM floor. It passed 45 local
CPU safety checks. The A/B was control then candidate and is not a
long-context or production qualification.
