//! Test-only factorial replay: device/host current input crossed with device-generated histories.
use super::*;
use sha2::{Digest, Sha256};

pub(super) struct History {
    layer: usize,
    conv: Vec<u8>,
    rec: Vec<u8>,
}

impl History {
    pub(super) fn read(step: &FlashStep<'_>, layer: usize) -> Self {
        Self {
            layer,
            conv: step.read_state_bytes(StateSlot::DeltaConv(layer)).unwrap(),
            rec: step.read_state_bytes(StateSlot::DeltaRec(layer)).unwrap(),
        }
    }

    fn restore(&self, step: &mut FlashStep<'_>) {
        step.write_state_bytes(StateSlot::DeltaConv(self.layer), &self.conv)
            .unwrap();
        step.write_state_bytes(StateSlot::DeltaRec(self.layer), &self.rec)
            .unwrap();
        let restored = Self::read(step, self.layer);
        assert_eq!(restored.conv, self.conv, "conv history failed roundtrip");
        assert_eq!(restored.rec, self.rec, "recurrent history failed roundtrip");
    }

    fn receipt(&self) -> serde_json::Value {
        serde_json::json!({"conv_bytes":self.conv.len(),"rec_bytes":self.rec.len(),
            "conv_sha256":format!("{:x}",Sha256::digest(&self.conv)),
            "rec_sha256":format!("{:x}",Sha256::digest(&self.rec))})
    }
}

pub(super) struct MixerTrace {
    pub(super) input: Vec<f32>,
    pub(super) core: Vec<f32>,
    pub(super) output: Vec<f32>,
}

fn run_mixer(
    step: &mut FlashStep<'_>,
    layer: usize,
    at: (usize, u32),
    input: &[f32],
) -> MixerTrace {
    let (position, token) = at;
    step.begin_token().unwrap();
    let (words, offset) = step.words_buffer().unwrap();
    words
        .write_at(
            &step.stream,
            offset,
            &StepParams::new(token as i32, position as i32)
                .unwrap()
                .to_bytes(),
        )
        .unwrap();
    let mixed = step.slot(Slot::Mixed(Site::Attn)).unwrap();
    assert_eq!(input.len(), step.geometry.hidden);
    assert!(input.iter().all(|v| v.is_finite()));
    let bytes: Vec<_> = input.iter().flat_map(|v| v.to_ne_bytes()).collect();
    mixed
        .buffer()
        .write_at(&step.stream, mixed.offset(), &bytes)
        .unwrap();
    assert!(
        same_bits(&step.read_f32(mixed, input.len()).unwrap(), input),
        "input upload changed bits"
    );
    step.mixer(layer, position, 1).unwrap();
    let core = step
        .read_f32(
            step.slot(Slot::GgufDeltaTile).unwrap(),
            step.geometry.delta.value_width(),
        )
        .unwrap();
    let output = step
        .read_f32(step.slot(Slot::BlockOut(Site::Attn)).unwrap(), input.len())
        .unwrap();
    assert!(core.iter().chain(&output).all(|v| v.is_finite()));
    step.finish_token().unwrap();
    MixerTrace {
        input: input.to_vec(),
        core,
        output,
    }
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

fn scores(
    name: &str,
    result: &MixerTrace,
    host: &delta_diagnostic::Reference,
) -> serde_json::Value {
    let core_error = relative_l2(&result.core, host.core());
    let output_error = relative_l2(&result.output, host.output());
    assert!(core_error.is_finite() && output_error.is_finite());
    eprintln!(
        "Delta factorial: {name} core_relative_l2={core_error:.9} output_relative_l2={output_error:.9}"
    );
    serde_json::json!({"core":result.core,"output":result.output,"core_relative_l2":core_error,"output_relative_l2":output_error})
}

pub(super) struct Inputs<'a, 'b> {
    pub(super) model: &'a GgufModel,
    pub(super) live_step: &'a FlashStep<'b>,
    pub(super) live: &'a MixerTrace,
    pub(super) gpu_history: &'a History,
    pub(super) layer: usize,
    pub(super) host0: &'a delta_diagnostic::Reference,
    pub(super) host1: &'a delta_diagnostic::Reference,
}

pub(super) fn run(inputs: Inputs<'_, '_>) {
    let Inputs {
        model,
        live_step,
        live,
        gpu_history,
        layer,
        host0,
        host1,
    } = inputs;
    assert_eq!(gpu_history.layer, layer);
    let out =
        std::env::var_os("FIFI_FLASH_GGUF_DELTA_HISTORY_OUT").expect("new diagnostic output path");
    assert!(!Path::new(&out).exists(), "diagnostic output must be new");
    let name = layer_name(layer, "linear_attn.out_proj");
    let Some(CudaWeight::Native { weight, .. }) = live_step.weights.get(&name) else {
        panic!("native layer-{layer} output missing")
    };
    assert!(
        !live_step
            .geometry
            .delta
            .value_dim
            .is_multiple_of(weight.quant().block_size()),
        "fixture requires tiled Delta core"
    );
    let mut isolated = FlashStep::new_gguf_decode(
        live_step.ctx.clone(),
        live_step.stream.clone(),
        live_step.weights.clone(),
        model.config().model_shape().unwrap(),
        model.config().kinds(),
        3,
        model.map().max_quantized_input_width().unwrap(),
        layer..layer + 1,
    )
    .unwrap();
    let zero_history = History::read(&isolated, layer);
    assert!(
        zero_history
            .conv
            .iter()
            .chain(&zero_history.rec)
            .all(|v| *v == 0)
    );
    let host_first = run_mixer(&mut isolated, layer, (0, 9707), host0.input());
    let first_core_error = relative_l2(&host_first.core, host0.core());
    let first_output_error = relative_l2(&host_first.output, host0.output());
    eprintln!(
        "Delta factorial: host-input position-0 core_relative_l2={first_core_error:.9} output_relative_l2={first_output_error:.9}"
    );
    assert!(
        first_core_error < 0.0001 && first_output_error < 0.0001,
        "position-0 host-input replay differs from host: core={first_core_error}, output={first_output_error}"
    );
    let host_history = History::read(&isolated, layer);
    assert_ne!(
        gpu_history.rec, zero_history.rec,
        "GPU recurrent history must be nonzero"
    );
    assert_ne!(
        gpu_history.conv, zero_history.conv,
        "GPU convolution history must be nonzero"
    );
    assert_ne!(
        host_history.rec, zero_history.rec,
        "host-input recurrent history must be nonzero"
    );
    assert_ne!(
        host_history.conv, zero_history.conv,
        "host-input convolution history must be nonzero"
    );
    assert_ne!(
        host_history.rec, gpu_history.rec,
        "host-input and actual-input recurrent histories should differ"
    );
    assert_ne!(
        host_history.conv, gpu_history.conv,
        "host-input and actual-input convolution histories should differ"
    );
    let gpu_conv_host_rec = History {
        layer,
        conv: gpu_history.conv.clone(),
        rec: host_history.rec.clone(),
    };
    let host_conv_gpu_rec = History {
        layer,
        conv: host_history.conv.clone(),
        rec: gpu_history.rec.clone(),
    };
    let zero_conv_host_rec = History {
        layer,
        conv: zero_history.conv.clone(),
        rec: host_history.rec.clone(),
    };
    let host_conv_zero_rec = History {
        layer,
        conv: host_history.conv.clone(),
        rec: zero_history.rec.clone(),
    };
    let mut results = serde_json::Map::new();
    for (name, history, input) in [
        ("gpu_history_gpu_input", gpu_history, live.input.as_slice()),
        ("gpu_history_host_input", gpu_history, host1.input()),
        (
            "host_input_history_gpu_input",
            &host_history,
            live.input.as_slice(),
        ),
        (
            "host_input_history_host_input",
            &host_history,
            host1.input(),
        ),
        (
            "gpu_conv_host_rec_host_input",
            &gpu_conv_host_rec,
            host1.input(),
        ),
        (
            "host_conv_gpu_rec_host_input",
            &host_conv_gpu_rec,
            host1.input(),
        ),
        (
            "zero_conv_host_rec_host_input",
            &zero_conv_host_rec,
            host1.input(),
        ),
        (
            "host_conv_zero_rec_host_input",
            &host_conv_zero_rec,
            host1.input(),
        ),
        ("zero_history_host_input", &zero_history, host1.input()),
    ] {
        history.restore(&mut isolated);
        let result = run_mixer(&mut isolated, layer, (1, 11), input);
        if name == "gpu_history_gpu_input" {
            assert!(
                same_bits(&result.core, &live.core) && same_bits(&result.output, &live.output),
                "replay failed to reproduce propagated mixer bitwise"
            );
        }
        results.insert(name.to_string(), scores(name, &result, host1));
    }
    let host_out: Vec<f32> =
        serde_json::from_value(results["host_input_history_host_input"]["output"].clone()).unwrap();
    let zero_out: Vec<f32> =
        serde_json::from_value(results["zero_history_host_input"]["output"].clone()).unwrap();
    let history_effect = relative_l2(&zero_out, &host_out);
    assert!(
        history_effect.is_finite() && history_effect > 1e-5,
        "zero-history negative control failed"
    );
    eprintln!(
        "Delta factorial: zero_history_effect_relative_l2={history_effect:.9}; replay is isolated, not host-state or model parity"
    );
    let receipt = serde_json::json!({"layer":layer,"position":1,"token":11,
        "gpu_history":gpu_history.receipt(),"host_input_generated_history":host_history.receipt(),"zero_history":zero_history.receipt(),
        "host_first_core":host_first.core,"host_first_output":host_first.output,"host_first_reference_core":host0.core(),"host_first_reference_output":host0.output(),
        "actual_input":live.input,"host_input":host1.input(),"host_core":host1.core(),"host_output":host1.output(),
        "zero_history_effect_relative_l2":history_effect,"results":results});
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .unwrap();
    serde_json::to_writer(file, &receipt).unwrap();
    eprintln!("Delta factorial diagnostic completed; not propagated model parity");
}
