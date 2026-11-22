# Gemma 4 test data

These fixtures use two Hugging Face models: `jialinyyzz/humanizer` at revision
`2e57b6528a4070c33f3e09ca24e3325a84272f68` (a Gemma 4 fine-tune, Apache-2.0) and
`nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` at revision
`49843eed8db1f1c616c47af75cfd36e07aec0667` (OpenMDW-1.1). See `THIRD_PARTY_NOTICES.md`
at the repository root for the licence texts.
No file here contains model weights.

- `humanizer-config.json`: the unmodified `config.json` of the model:
  https://huggingface.co/jialinyyzz/humanizer/blob/2e57b6528a4070c33f3e09ca24e3325a84272f68/config.json
  SHA-256 `a9f6a5679ff252dce0845866c58992520171113e1e3489631ca8e2e7f47f7bb4` (4,298 bytes).
  The configuration tests in `engine/src/family/gemma/gemma4/tests.rs` read it.
- `bolt-config.json`: the unmodified `config.json` of `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16`
  at revision `49843eed8db1f1c616c47af75cfd36e07aec0667`:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/blob/49843eed8db1f1c616c47af75cfd36e07aec0667/config.json
  SHA-256 `6f7a13f1d5fc3295e682616f937a4a573756f59cb9a1700cd4801fef08ad6546` (5,144 bytes).
  The Bolt configuration tests in `engine/src/family/gemma/gemma4/tests.rs` and the serve
  dispatch tests in `engine/src/family/gemma/gemma4/serve.rs` read it.
- `prompt-cases.json`: five prompt cases for the model's completion prompt.
  The instruction and separator come from the model README.
  Each case records the draft text, the full prompt, its SHA-256, and the token ids from the model's tokenizer with a literal `<bos>` prefix.
  `engine/src/family/gemma/gemma4/prompt.rs` checks its prompt builder against these cases.
- `parity-prompt.json`: the prompt `The capital of France is` as token ids (BOS id `2` first), with the SHA-256 of the model's `tokenizer.json`.
  The ignored test `traces_first_position_top_five` in `engine/src/family/gemma/gemma4/resident/tests.rs` feeds these ids to a real checkpoint named by `GEMMA4_Q8_FILE`.
- `kernels/`: CUDA kernel reference fixtures. See `kernels/README.md`.

These files support configuration, prompt, and kernel checks only. They do not load weights or enable generation by themselves.
