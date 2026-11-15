//! The shared served-kernel applier's own order: defaults first, then the switch check.

/// `CudaContext::register_served_kernels` applies the single-row default, the id-indexed
/// default, the odd-rate and fused-input switches, and only then runs the launch-switch
/// check - in the source, in that order, so a family start cannot register a subset or check
/// before its defaults are in.
#[test]
fn the_shared_applier_sets_every_default_before_the_switch_check() {
    let source = include_str!("../exl3.rs");
    let applier = &source[source
        .find("pub(crate) fn register_served_kernels(&self, profile: ServedKernels)")
        .expect("register_served_kernels")..];
    let single = applier
        .find("self.set_single_row_default(")
        .expect("the single-row default");
    let indexed = applier
        .find("self.set_indexed_default(")
        .expect("the indexed default");
    let odd = applier
        .find("self.set_odd_rate_kernel(")
        .expect("the odd-rate switch");
    let fused = applier
        .find("self.set_fused_in_kernel(")
        .expect("the fused-input switch");
    let check = applier
        .find("self.check_single_gather()")
        .expect("the switch check");
    assert!(
        single < indexed && indexed < odd && odd < fused && fused < check,
        "every default must be registered before check_single_gather()"
    );
}
