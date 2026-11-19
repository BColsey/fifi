# Coalescing contiguous Flint expert reads

Flint stages sorted expert IDs from file-backed safetensors. When multiple selected experts are consecutive, both their source bytes and their destination slots are contiguous. `FIFI_FLINT_MLX_EXPERT_READ_RUNS=1` (with `FIFI_FLINT_MLX_EXPERT_PIPELINE=1`) combines each such run into one checked read of the exact requested bytes, using the same page layout, output remapping and Metal dispatch as the shared-page pipeline. Missing tensors and invalid extents or destination lengths still fail before a read. The `prefill` value coalesces only multi-row forwards: single-row stages — the final prompt token and every decode step — keep the per-slice reader.

## Mini measurements

Contributor record (zbrdc/fifi#62 commits f0c705d4e and ffc0180cc, 18 November 2026, the verified published checkpoint on his 16 GB M4 Mac mini; carried as a record, not reproduced here at port time).

From the prior six-request trace, the number of read calls per tensor role would fall from 28,164 to 14,657 for multi-row stages (48.0%). For single-row stages it would fall from 36,096 to 35,396 (1.9%). The logical bytes are identical. The published-checkpoint A/B used four fresh benchmark processes in control, coalesced, coalesced, control order; all eight responses matched across both modes.

For the prefill-only mode, both arms used shared pages, 512-row admission and the same prompts with 12 completion tokens; servers ran control, candidate, candidate, control:

| Prompt tokens | Prefill read calls, control → coalesced | Time to first token, control → coalesced | Full request, control → coalesced |
| ---: | ---: | ---: | ---: |
| 16 | 26,244 → 20,979 | 5.96 → 5.50 s | 11.68 → 11.22 s |
| 45 | 52,821 → 29,727 | 13.47 → 9.58 s | 19.53 → 15.69 s |
| 143 | 82,575 → 23,472 | 17.01 → 12.07 s | 22.16 → 17.27 s |
| 557 | 142,722 → 40,392 | 28.65 → 21.23 s | 34.61 → 27.19 s |

The table shows medians from two fresh servers per mode. The 45-, 143- and 557-token prompts reached their first token 26–29% sooner. The server requested the same logical bytes on every match, and all 16 completion digests matched their controls. Every run issued 27,918 single-row decode reads; median decode rates were within 1% and the median peak process footprint rose by at most 120 MB.

The mode remains opt-in and default-off. Broader performance qualification requires production-serving workloads beyond these runs per prompt. The raw sample CSVs live in the PR branch (`scripts/benchmarks/mac-m4-flint-read-runs.csv`, `mac-m4-flint-prefill-read-runs.csv`), not in this tree.
