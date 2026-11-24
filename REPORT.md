# Resident token chain

Branch `task/inkling-resident-token-chain`, base `54389798b1dd594531de10ad9032bfca382c74b3` (`evidence/setup.log`). Setup complete; prior history searched (`evidence/history.log`, `evidence/prior-notes.log`). No new measurements yet.

S4a plan: admit a pair of banked local visits together, stage both inputs first, hand off the complete depth state device-to-device on one stream, defer all downloads, validate both receipts before publishing either cursor. Gate on decode only, decline taps and draft windows. No widening to global until real-weight parity. Main risk: incomplete resident projections on the second visit. The old path remains the oracle. The bounded pair implementation now passes native-CUDA cargo check with tests and formatting (`evidence/check-latest.log`, `check-latest.status`, `fmt.log`). Both visits are pre-admitted, share a stream and one transfer fence, and hand off depth device-to-device. Warm convolution histories are reused only when immutable K/V row identities match the last committed pair. Real checkpoint test `real_consecutive_banked_rows` passed on the leased device, slots 18→19. End row: 104/2304 exact, max absolute 0.0000152587890625, relative L2 0.000000401303 (limits 0.005 and 0.0005); one passed and none skipped. Candidate `7356cb8d2f73d35c94a2ccae50a5583cfbf5c553`. Release build `evidence/release-latest.log`, result `evidence/pair.log`. Next command: `bash run-visit.sh artifact evidence/trace trace` (queued; inspect `evidence/trace-second.status`). The initial trace preflight declined to run during a concurrent build. Timing under concurrent CPU compilation will not be interpreted as serving performance.

Measured local pair, implementation `7356cb8d2f73d35c94a2ccae50a5583cfbf5c553`: see `evidence/trace-table.log` (from `python trace-table.py`), `evidence/trace/comparison.json` (CPU-side analysis of the raw Nsight SQLite traces), raw `inkling.nsys-rep` in each arm. Both arms are the same executable and flags except `FIFI_INKLING_CUDA_RESIDENT_TOKEN`. No prefix reuse. Counts are per steady decode token. Timings are traced and the trace ran during CPU compilation; not a serving A/B.

| per token | off | local pair |
|---|---:|---:|
| kernel launches |1054|1090|
| explicit host syncs |109|106|
| kernel time µs |15417|15752|
| observed GPU busy µs |18218|18567|
| no observed GPU work µs |18955|15774|
| H2D bytes |5165856|4861296|
| D2H bytes |2699368|2717872|
| H2D µs / effective GB/s |1919 / 2.69|1880 / 2.59|
| D2H µs / effective GB/s |881 / 3.06|887 / 3.06|

The trace verifier found 64 pair transactions in the full candidate request and none off: no blocking host call between upload and final download; both routers and convolutions finish before the first D2H. End-of-pair host shadow still needs rows, hence more D2H, not less. Text, usage and finish reason matched for warmup, context_512 and natural_decode_64. No kernel-specific DRAM bandwidth or whole-token extrapolation. Sixteen existing glue tests passed under lease (evidence/glue.log); no skip messages. Remaining bottleneck: most visits still host orchestrated; launch overhead within the admitted pair actually increased.

S4b banked local→global pair passes real-weight row parity. New FiFi effective-tap kernel folds the dynamic per-head coefficients and residual scaling into the existing attention prep layout. Separate Q/K/V depth mixes and gammas; controllers, eight projections, dynamic prep, ring attention, output projection and expert residuals stay on the pair stream. Slots 22→23 continuation row: 89/2304 exact, max absolute 0.00003814697265625, relative L2 0.000000524687 (same limits). Local rerun unchanged. `evidence/global-pair.log`: 2 passed, none failed or skipped. Native-CUDA check, fmt, release build complete. Next command: `bash run-visit.sh artifact-global evidence/trace-global trace`.
