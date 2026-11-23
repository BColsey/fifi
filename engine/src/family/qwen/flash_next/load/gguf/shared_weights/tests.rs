use super::*;
use crate::{
    family::qwen::flash_next::load::gguf::testkit::{Spec, write_flash},
    weights::GgufType,
};

#[test]
fn shared_projection_plan_checks_dims_type_and_bytes() {
    let file = write_flash(&Spec {
        retype: vec![("blk.0.ffn_gate_shexp.weight".into(), GgufType::Q80)],
        ..Spec::default()
    });
    let model = GgufModel::open(&file.first).unwrap().unwrap();
    let d = &model.config.dims;
    let dims = [u64::from(d.hidden), u64::from(d.shared_inter)];
    let entry = model.map.get(Role::SharedGate, Some(0)).unwrap();
    assert!(validate(entry, &dims).is_ok());
    assert!(validate(entry, &[dims[1], dims[0]]).is_err());
    let mut bad = entry.clone();
    bad.bytes -= 1;
    assert!(validate(&bad, &dims).is_err());
    bad = entry.clone();
    bad.ty = GgufType::F32;
    assert!(validate(&bad, &dims).is_err());
    bad = entry.clone();
    bad.convention = Convention::GammaOnePlusW;
    assert!(validate(&bad, &dims).is_err());
    bad = entry.clone();
    bad.dims = vec![dims[0] - 1, dims[1]];
    assert!(validate(&bad, &bad.dims).is_err());
    bad.dims = vec![u64::MAX, u64::MAX];
    assert!(validate(&bad, &bad.dims).is_err());
}
