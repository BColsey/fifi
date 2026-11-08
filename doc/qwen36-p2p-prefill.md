# Experimental prompt-only head skipping

The multi-device Qwen3.6 path normally computes the final vocabulary
projection for every prompt token, even when the session only needs to
advance model state. `Q36_P2P_SKIP_PREFILL_HEAD=1` skips the final norm,
vocabulary projection and argmax for those prompt-only feeds. The last
prompt token still takes the normal logits-producing path before sampling.
The option is off by default; remove the variable or set it to `0` to roll
back.

The next input token is supplied by the host on both cards. Layer and P2P
status checks still run for each feed, and state advancement is unchanged.
Because the skipped logits are never computed, non-finite output-head values
are checked when logits are next required, rather than for every prompt-only
token. This option is experimental until qualified on the target system.

`python3 scripts/bench_p2p_prefill.py --self-test` checks the measurement driver
without a GPU. Run it against an otherwise idle server with an output file in
scratch storage, once with the option off and once on. The reported rate is
prompt tokens divided by time to the complete reply, so it includes request
and generation overhead. Save individual samples and compare the same prompt,
model, context and host configuration.

For example, with `FIFI_RUN` pointing to a writable scratch directory:

```sh
python3 scripts/bench_p2p_prefill.py --samples 3 --repeat 100 \
  --output "$FIFI_RUN/baseline.json"
# Restart the engine with Q36_P2P_SKIP_PREFILL_HEAD=1, then:
python3 scripts/bench_p2p_prefill.py --samples 3 --repeat 100 \
  --output "$FIFI_RUN/candidate.json"
```

The driver defaults to model `fifi-q36`, `http://127.0.0.1:18792`, and a
131,072-token context. Override these with `--model`, `--url`, and
`--expected-context` when benchmarking a different configuration. A result
file is never overwritten; use a new path for every run.

On an RTX 3060 12 GB and RTX 3080 10 GB, using the Q4_K_M GGUF model,
q8 KV, mapped host experts and a 131,072-token context, the median of three
524-token requests rose from 31.99 to 32.73 prompt tokens/s with the same
binary and the option off versus on. One 4,274-token request fell from
133.35 s to 130.52 s when compared with the previous binary. These are
end-to-end times through a two-token reply, not pure kernel timings. The
[individual samples, first-token telemetry, hardware, and test results](qwen36-p2p-prefill-results.json)
record the conditions and limitations; a full 131,072-token prompt was not
run.
