//! Pinned QSA Q/K/V matched-input and actual dispatch diagnostic.
use super::*;
use crate::{
    family::qwen::flash_next::load::gguf::GgufModel,
    kernels::cuda::{CudaContext, resident::matvec_q::QuantType},
};

#[test]
#[ignore = "requires verified pinned GGUF, position-1 layer-3 host capture and admitted CUDA"]
fn gguf_qsa_input_matched_and_dispatch() {
    let path = std::env::var_os("FIFI_FLASH_GGUF").expect("set verified GGUF first shard");
    let capture = std::env::var_os("FIFI_FLASH_GGUF_QSA_INPUT_CAPTURE").expect("set host capture");
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(capture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut input: Option<Vec<f32>> = None;
    let mut host = Vec::new();
    for part in ["q", "k", "v"] {
        let name = format!("blk.3.attn_{part}.weight");
        let mut found = rows
            .iter()
            .filter(|row| row["name"] == name && row["expert"].is_null());
        let row = found.next().expect("missing QSA capture");
        assert!(found.next().is_none(), "duplicate {name}");
        assert_eq!(row["type"], "Q6K");
        let x: Vec<f32> = serde_json::from_value(row["input"].clone()).unwrap();
        if let Some(ref first) = input {
            assert_eq!(&x, first, "QSA inputs differ");
        } else {
            input = Some(x);
        }
        host.push(serde_json::from_value::<Vec<f32>>(row["output"].clone()).unwrap());
    }
    let input = input.unwrap();
    let model = GgufModel::open(Path::new(&path)).unwrap().unwrap();
    let ctx = CudaContext::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = CudaWeights::from_native(model.upload_qsa_projections(3, &ctx).unwrap()).unwrap();
    let mut results = serde_json::Map::new();
    for (part, expected) in ["q", "k", "v"].into_iter().zip(&host) {
        let name = layer_name(3, &format!("self_attn.{part}_proj"));
        let Some(CudaWeight::Native { weight, .. }) = weights.get(&name) else {
            panic!("missing {name}")
        };
        assert_eq!(weight.quant(), QuantType::Q6K);
        let result =
            super::shared_projection_probe::projection(&ctx, &stream, weight, &input, expected);
        eprintln!(
            "QSA matched {part}: native={} f32={} native_ms={} f32_ms={}",
            result["native_error"],
            relative_l2(
                &serde_json::from_value::<Vec<f32>>(result["f32_output"].clone()).unwrap(),
                expected
            ),
            result["native_ms"],
            result["f32_ms"]
        );
        results.insert(part.into(), result);
    }

    // Verify the actual Q project_cols plus grouped K/V dispatch against the isolated outputs,
    // with sentinel columns on both ends and a bitwise input check. Each invocation uses one
    // process-global mode; the GPU runner executes this test alone.
    let shape = model.config().model_shape().unwrap();
    let max_k = model.map().max_quantized_input_width().unwrap();
    let step = FlashStep::new_gguf_decode(
        &ctx,
        &stream,
        &weights,
        shape,
        model.config().kinds(),
        2,
        max_k,
        3..4,
    )
    .unwrap();
    let x = ctx.allocate(input.len() * 4).unwrap();
    x.write_at(
        &stream,
        0,
        &input
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let widths: Vec<_> = host.iter().map(Vec::len).collect();
    assert_eq!(widths[1], widths[2]);
    let total = 4 + widths.iter().sum::<usize>();
    let buffer = ctx.allocate(total * 4).unwrap();
    let row = buffer.range(0, total * 4).unwrap();
    let sentinel = -1234.5f32;
    buffer
        .write_at(&stream, 0, &sentinel.to_ne_bytes().repeat(total))
        .unwrap();
    let q = layer_name(3, "self_attn.q_proj");
    let k = layer_name(3, "self_attn.k_proj");
    let v = layer_name(3, "self_attn.v_proj");
    let qoff = 2;
    let koff = qoff + widths[0];
    let voff = koff + widths[1];
    let xr = x.range(0, x.bytes()).unwrap();
    step.project_cols(&q, xr, row, qoff, widths[0], total, 1)
        .unwrap();
    step.project_group(&[(&k, koff, widths[1]), (&v, voff, widths[2])], xr, &row, 1)
        .unwrap();
    let actual = step.read_f32(row, total).unwrap();
    let precise = std::env::var_os("FIFI_FLASH_GGUF_TEST_F32_QSA_IN").is_some();
    let field = if precise {
        "f32_output"
    } else {
        "native_output"
    };
    for (part, offset, width) in [
        ("q", qoff, widths[0]),
        ("k", koff, widths[1]),
        ("v", voff, widths[2]),
    ] {
        let expected: Vec<f32> = serde_json::from_value(results[part][field].clone()).unwrap();
        assert_eq!(expected.len(), width);
        assert!(
            actual[offset..offset + width]
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{part} dispatch differs from isolated {field}"
        );
    }
    assert!(
        actual[..2]
            .iter()
            .chain(&actual[total - 2..])
            .all(|x| x.to_bits() == sentinel.to_bits())
    );
    assert!(
        step.read_f32(xr, input.len())
            .unwrap()
            .iter()
            .zip(&input)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(step.health().unwrap(), (0, 0));
    let path = std::env::var_os("FIFI_FLASH_GGUF_QSA_INPUT_OUT").expect("set fresh receipt path");
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer(
        file,
        &serde_json::json!({"layer":3,"position":1,"token":11,
        "precise_dispatch":precise,"dispatch_bitwise_matches_isolated":true,"results":results}),
    )
    .unwrap();
    eprintln!(
        "QSA Q/K/V dispatch mode={field} matches isolated bitwise; sentinel and input preserved"
    );
}
