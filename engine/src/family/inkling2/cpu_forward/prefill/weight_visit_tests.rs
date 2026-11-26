//! Model-level probe. Run each selector in a fresh process with fixed token IDs.
use super::CpuForward;
use std::{env, fs, path::Path, time::Instant};

#[test]
#[ignore = "requires the Inkling checkpoint and a CUDA device from SM75"]
fn weight_visit_request() {
    let checkpoint = env::var_os("FIFI_INKLING_MODEL_DIR").expect("model path");
    let ids: Vec<u32> = env::var("FIFI_INKLING_TOKEN_IDS")
        .expect("fixed 512-token request")
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(ids.len(), 512);
    let mut model = CpuForward::open(Path::new(&checkpoint), ids.len()).unwrap();
    model.enable_cuda(0).unwrap();
    assert!(model.supports_grouped_prefill());
    let start = Instant::now();
    model.prefill_grouped(&ids[..511]).unwrap();
    eprintln!(
        "weight_visit_request prefill_us={}",
        start.elapsed().as_micros()
    );
    if let Some(out) = env::var_os("FIFI_INKLING_LOGITS") {
        let logits = model.token(ids[511], true).unwrap().unwrap();
        assert!(logits.iter().all(|x| x.is_finite()));
        let data: Vec<_> = logits.into_iter().flat_map(f32::to_le_bytes).collect();
        fs::write(out, data).unwrap();
    }
}
