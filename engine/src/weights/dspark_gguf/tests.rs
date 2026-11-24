use std::{io::Write, sync::Arc};

use half::bf16;

use super::*;

#[test]
fn loads_dense_sidecar_and_runs_local_attention() {
    let path = tiny_sidecar(Options::default());
    let draft = DsparkGguf::open(path.path()).unwrap();
    let config = draft.config();
    assert_eq!(config.target_layers, [5, 9]);
    assert_eq!(config.vocab, 3);
    assert_eq!(config.padded_vocab, 4);
    assert_eq!(config.final_logit_softcap, 30.0);
    assert_eq!(draft.markov().previous.row(2), [2.0]);
    assert_eq!(draft.markov().output.project(&[2.0]), [0.0, 2.0]);
    assert_eq!(draft.markov().joint.project(&[0.0; 6]), [0.0; 3]);
    assert_eq!(draft.markov().joint_bias, [0.0; 3]);
    assert_eq!(draft.markov().output_bias, [0.0; 2]);

    let features: [Arc<[f32]>; 2] = [
        Arc::new([2.0, 0.0, 2.0, 0.0]),
        Arc::new([0.0, 2.0, 0.0, 2.0]),
    ];
    let anchor = [1.0, 0.0];
    let block = draft.forward(&anchor, &features, 2).unwrap();
    assert_eq!(block.hidden.len(), 1);
    assert_eq!(block.hidden[0], [1.28125, 0.6015625]);
    assert_eq!(block.recurrent_context, [0.0, 1.4140625]);

    let changed_old_row: [Arc<[f32]>; 2] = [Arc::new([9.0; 4]), Arc::clone(&features[1])];
    assert_eq!(
        draft.forward(&anchor, &changed_old_row, 2).unwrap().hidden,
        block.hidden
    );
    assert!(draft.forward(&anchor, &features, 1).is_err());
    assert!(draft.forward(&anchor, &[Arc::new([1.0, 0.0])], 1).is_err());
}

#[test]
fn draft_workspace_grows_with_context_until_the_last_full_block() {
    let fixture = tiny_sidecar(Options::default());
    let model = DsparkGguf::open(fixture.path()).unwrap();
    let config = model.config();
    let within = config.workspace_bytes(config.context_length / 2).unwrap();
    let at_limit = config.workspace_bytes(config.context_length).unwrap();
    assert!(within < at_limit);
    assert_eq!(at_limit, config.workspace_bytes(usize::MAX).unwrap());
}

#[test]
fn rejects_wrong_draft_convention_and_tensor_geometry() {
    let cases = [
        (
            Options {
                causal: true,
                ..Options::default()
            },
            "dflash.attention.causal",
        ),
        (
            Options {
                attention_width: 3,
                ..Options::default()
            },
            "blk.0.attn_q.weight",
        ),
        (
            Options {
                visits: [9, 5],
                ..Options::default()
            },
            "visit order",
        ),
        (
            Options {
                visits: [-1, 5],
                ..Options::default()
            },
            "nonnegative",
        ),
    ];
    for (options, expected) in cases {
        let path = tiny_sidecar(options);
        let error = DsparkGguf::open(path.path()).unwrap_err();
        assert!(matches!(error, EngError::Checkpoint(message) if message.contains(expected)));
    }
}

#[test]
#[ignore = "requires the published Unsloth Inkling-2-Mini DSpark GGUF"]
fn published_inkling_sidecar_executes_a_draft_block() {
    let path = std::env::var_os("FIFI_TEST_INKLING_DSPARK_GGUF")
        .expect("set FIFI_TEST_INKLING_DSPARK_GGUF to the published Q8_0 or BF16 sidecar");
    let draft = DsparkGguf::open(Path::new(&path)).unwrap();
    let config = draft.config();
    assert_eq!(config.target_layers, [5, 23, 41, 48]);
    assert_eq!(config.width, 2304);
    assert_eq!(config.markov_rank, 256);
    assert_eq!(config.padded_vocab, 201024);
    assert_eq!(config.vocab, 200058);
    assert_eq!(config.block_size, 8);
    let probe = |index: usize| ((index % 19) as f32 - 9.0) / 19.0;
    let anchor: Vec<_> = (0..config.width).map(probe).collect();
    let taps: Arc<[f32]> = (0..config.width * config.target_layers.len())
        .map(probe)
        .collect();
    let block = draft.forward(&anchor, &[taps], 1).unwrap();
    assert_eq!(block.hidden.len(), 7);
    assert!(
        block
            .hidden
            .iter()
            .all(|row| row.len() == 2304 && row.iter().all(|x| x.is_finite()))
    );
    assert_ne!(block.hidden[0], block.hidden[6]);
    assert_eq!(block.recurrent_context.len(), config.width);
    assert!(block.recurrent_context.iter().all(|x| x.is_finite()));
    assert!(block.recurrent_context.iter().any(|x| *x != 0.0));

    if draft.layers[0].query.ty == GgufType::Q80 {
        // Independent NumPy projection/attention reference on this Q8_0
        // checkpoint. The F32 reductions can round at different BF16 bins.
        assert_reference(
            &block.hidden[0],
            &[
                -1.703125, 0.46484375, 1.953125, 1.03125, -3.125, 3.890625, -4.75, 3.703125,
            ],
            0.0625,
        );
        assert_reference(
            &block.hidden[6],
            &[
                -1.8984375, 0.515625, 1.734375, 1.421875, -2.765625, 3.421875, -4.25, 3.90625,
            ],
            0.0625,
        );
        assert_reference(
            &block.recurrent_context,
            &[
                -0.029_907_227,
                0.022_705_078,
                -0.185_546_88,
                -0.28125,
                0.7109375,
                0.6484375,
                -1.0078125,
                1.4296875,
            ],
            0.0078125,
        );
    }
}

fn assert_reference(row: &[f32], expected: &[f32], tolerance: f32) {
    let actual = &row[..expected.len()];
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max);
    eprintln!("DSpark reference max absolute error: {max_error}");
    assert!(
        max_error <= tolerance,
        "error {max_error} exceeds {tolerance} on {actual:?} vs {expected:?}"
    );
}

struct Options {
    causal: bool,
    attention_width: u64,
    visits: [i32; 2],
}

impl Default for Options {
    fn default() -> Self {
        Self {
            causal: false,
            attention_width: 2,
            visits: [5, 9],
        }
    }
}

fn tiny_sidecar(options: Options) -> tempfile::NamedTempFile {
    let metadata = metadata(&options);
    let tensors = tensors(options.attention_width);
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((tensors.len() as u64).to_le_bytes());
    header.extend((metadata.len() as u64).to_le_bytes());
    header.extend(metadata.into_iter().flatten());
    let mut payload = Vec::new();
    for (name, shape, ty, values) in tensors {
        pad(&mut payload);
        header.extend(string(name));
        header.extend((shape.len() as u32).to_le_bytes());
        header.extend(shape.iter().flat_map(|value| value.to_le_bytes()));
        header.extend(ty.id().to_le_bytes());
        header.extend((payload.len() as u64).to_le_bytes());
        if ty == GgufType::F32 {
            payload.extend(values.into_iter().flat_map(f32::to_le_bytes));
        } else {
            payload.extend(
                values
                    .into_iter()
                    .flat_map(|x| bf16::from_f32(x).to_le_bytes()),
            );
        }
    }
    pad(&mut header);
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&payload).unwrap();
    file
}

fn metadata(options: &Options) -> Vec<Vec<u8>> {
    let mut metadata = vec![kv("general.architecture", 8, string("dflash"))];
    for (key, value) in [
        ("block_count", 1u32),
        ("embedding_length", 2),
        ("dspark.markov_rank", 1),
        ("vocab_size_unpadded", 3),
        ("vocab_size", 4),
        ("block_size", 2),
        ("context_length", 16),
        ("feed_forward_length", 2),
        ("attention.head_count", 1),
        ("attention.head_count_kv", 1),
        ("attention.key_length", 2),
        ("attention.value_length", 2),
        ("rope.dimension_count", 2),
        ("attention.sliding_window", 1),
    ] {
        metadata.push(kv(&format!("dflash.{key}"), 4, value.to_le_bytes()));
    }
    for (key, value) in [
        ("rope.freq_base", 10000.0f32),
        ("attention.layer_norm_rms_epsilon", 0.000001),
        ("final_logit_softcapping", 30.0),
    ] {
        metadata.push(kv(&format!("dflash.{key}"), 6, value.to_le_bytes()));
    }
    for (key, value) in [
        ("sample_from_anchor", false),
        ("has_confidence_head", false),
        ("attention.causal", options.causal),
    ] {
        metadata.push(kv(&format!("dflash.{key}"), 7, [u8::from(value)]));
    }
    let array = |key: &str, elem: u32, len: u64, data: &[u8]| {
        let mut body = elem.to_le_bytes().to_vec();
        body.extend(len.to_le_bytes());
        body.extend(data);
        kv(key, 9, body)
    };
    metadata.push(array("dflash.attention.sliding_window_pattern", 7, 1, &[1]));
    let target_layers: Vec<_> = options
        .visits
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect();
    metadata.push(array("dflash.target_layers", 5, 2, &target_layers));
    metadata
}

type Tensor = (&'static str, Vec<u64>, GgufType, Vec<f32>);

fn tensors(attention_width: u64) -> Vec<Tensor> {
    let f32_tensor = |name, dims, values| (name, dims, GgufType::F32, values);
    let matrix = |name, dims, values| (name, dims, GgufType::Bf16, values);
    let norm = |name| f32_tensor(name, vec![2], vec![1.0; 2]);
    let zero = |name| matrix(name, vec![2, 2], vec![0.0; 4]);
    let identity = |name| matrix(name, vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    vec![
        f32_tensor("dspark_fuse.weight", vec![2, 2], vec![0.0; 4]),
        f32_tensor("mask_embd.weight", vec![2], vec![1.0, 0.0]),
        norm("output_norm.weight"),
        norm("dspark_norm.weight"),
        matrix("markov_w1.weight", vec![1, 4], vec![0.0, 1.0, 2.0, 3.0]),
        matrix("joint_proj.weight", vec![6, 3], vec![0.0; 18]),
        f32_tensor("joint_proj.bias", vec![3], vec![0.0; 3]),
        matrix("dspark_hidden_out.weight", vec![1, 2], vec![0.0, 1.0]),
        f32_tensor("dspark_hidden_out.bias", vec![2], vec![0.0; 2]),
        matrix(
            "blk.0.attn_q.weight",
            vec![2, attention_width],
            vec![0.0; 2 * attention_width as usize],
        ),
        zero("blk.0.attn_k.weight"),
        zero("blk.0.attn_v.weight"),
        identity("blk.0.attn_output.weight"),
        norm("blk.0.attn_q_norm.weight"),
        norm("blk.0.attn_k_norm.weight"),
        norm("blk.0.attn_norm.weight"),
        zero("blk.0.attn_k_ctx.weight"),
        identity("blk.0.attn_v_ctx.weight"),
        norm("blk.0.attn_k_ctx_norm.weight"),
        norm("blk.0.dflare_norm.weight"),
        zero("blk.0.ffn_gate.weight"),
        zero("blk.0.ffn_up.weight"),
        zero("blk.0.ffn_down.weight"),
        norm("blk.0.ffn_norm.weight"),
    ]
}

fn pad(bytes: &mut Vec<u8>) {
    bytes.resize(bytes.len().next_multiple_of(32), 0);
}

fn string(text: &str) -> Vec<u8> {
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend(text.as_bytes());
    bytes
}

fn kv(key: &str, ty: u32, contents: impl AsRef<[u8]>) -> Vec<u8> {
    let mut bytes = string(key);
    bytes.extend(ty.to_le_bytes());
    bytes.extend(contents.as_ref());
    bytes
}
