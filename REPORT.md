# Resident token chain

Branch `task/inkling-resident-token-chain`, base `54389798b1dd594531de10ad9032bfca382c74b3` (`evidence/setup.log`). Setup complete; prior history searched (`evidence/history.log`, `evidence/prior-notes.log`). No new measurements yet.

S4a plan: admit a pair of banked local visits together, stage both inputs first, hand off the complete depth state device-to-device on one stream, defer all downloads, validate both receipts before publishing either cursor. Gate on decode only, decline taps and draft windows. No widening to global until real-weight parity. Main risk: incomplete resident projections on the second visit. The old path remains the oracle. Next command: `cd wt/engine && cargo fmt --check` after the pair hook, then the native-CUDA check and release tests in the lane build environment.
