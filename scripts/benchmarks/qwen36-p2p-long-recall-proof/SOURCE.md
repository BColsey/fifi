# Guarded P2P long-recall evidence source

This is an archival snapshot of the host-specific runner used for the
server-counted recall experiment. It contains hard-coded paths and device
identifiers from the measured host; it is not a portable device runner or a
recommendation to stop an active service. Use the current host's coordinator
and device admission process before adapting it.

The private server was built from `76dec8a61d040efa97beefe273bed8dc9e68ff57`;
its SHA-256 was `efac9000d75f7fddddfa5c50882837f60915dbcfdfa6afa51433fe56da9fd69d`.
The recall driver came from `74f2b561e3cb3bf1eb36c62908f08515e2c94cb2`.
The source hashes recorded by the live preflight match this snapshot. The
runner pins model, driver, configuration, build and previous smoke evidence,
checks cold usage against server request logs, monitors the pair's VRAM and
other work, and restores the original engine after a handoff.

The short smoke passed with 636 server-counted tokens. In the staged run,
8,311 and 32,884 tokens completed successfully with three markers checked.
Another build appeared during the 131,195-token request, so the guard stopped
the campaign and restored the server. That partial request has no recall or
throughput result. The completed smaller requests occurred during clear guard
samples; the final sample records the conflicting build.

The campaign checks synthetic cold recall with prefix reuse disabled. It does
not establish long coding quality, warm-cache behavior, 128k completion, or
256k capability. `test_long_runner.py` and the copied runner tests exercised
51 CPU checks before the smoke. Refer to the accompanying measured evidence
for exact rates, artifact digests and additional limitations.
