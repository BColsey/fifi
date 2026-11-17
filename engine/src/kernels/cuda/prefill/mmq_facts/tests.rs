//! The device test of the MMQ facts binding: the first glue FFI child proves the glue archive
//! links and resolves by reading the device's own facts through the C entry. Every value is
//! checked against its valid domain and against the device probe; the exact 3090 values are
//! asserted only on a 3090.

use crate::kernels::cuda::testkit;

#[test]
fn mmq_facts_read_the_device() {
    if !testkit::device_present("mmq_facts_read_the_device") {
        return;
    }
    let phase = || -> Result<(i32, usize, usize), crate::EngError> {
        // The fixed API: a context method with the usable check and the dispatch guard, like
        // `device_is_integrated`.
        let ctx = crate::kernels::cuda::CudaContext::new()?;
        let facts = ctx.mmq_device_facts()?;
        Ok((facts.cc, facts.nsm, facts.smpbo))
    };
    let (cc, nsm, smpbo) = phase().expect("the device phase");
    // The hard gate first: every value is in its valid domain (a sentinel or a wrapped
    // negative would fail here) before the equality pins.
    assert!((700..=1000).contains(&cc), "cc {cc} is not an architecture");
    assert!((1..=1024).contains(&nsm), "nsm {nsm} is out of range");
    assert!(
        (32_768..=(1 << 20)).contains(&smpbo),
        "smpbo {smpbo} is out of range"
    );
    assert!(
        [720, 860].contains(&cc),
        "cc {cc} is not a built architecture (720 or 860)"
    );
    let facts = crate::kernels::cuda::peer::probe_device_set(&[0]).expect("probe device 0");
    let probe = &facts[0];
    assert_eq!(
        nsm, probe.multiprocessors as usize,
        "the facts and the probe agree on the SM count"
    );
    if probe.name.contains("3090") {
        assert_eq!(cc, 860, "the 3090 runs the sm_86 cubins");
        assert_eq!(nsm, 82, "the 3090 has 82 SMs");
        assert_eq!(smpbo, 101_376, "the 3090 offers 99 KiB opt-in per block");
    } else {
        println!("not a 3090 ({}): the 3090 pins are skipped", probe.name);
    }
    println!("MMQ facts: cc {cc}, nsm {nsm}, smpbo {smpbo}");
}

#[test]
fn mmq_facts_are_queried_once_per_context() {
    // The facts are read from the device once per context and every later call returns the
    // cached value, so a launch does not pay five attribute calls. The cache is per context,
    // not per process: a second context queries again.
    if !testkit::device_present("mmq_facts_are_queried_once_per_context") {
        return;
    }
    let phase = || -> Result<[(super::MmqFacts, usize); 4], crate::EngError> {
        let ctx = crate::kernels::cuda::CudaContext::new()?;
        let first = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        let second = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        let third = (ctx.mmq_device_facts()?, ctx.mmq_facts.queries.get());
        drop(ctx);
        let next = crate::kernels::cuda::CudaContext::new()?;
        let fresh = (next.mmq_device_facts()?, next.mmq_facts.queries.get());
        drop(next);
        Ok([first, second, third, fresh])
    };
    let [first, second, third, fresh] = phase().expect("the device phase");
    assert_eq!(first.1, 1, "the first call queries the device");
    assert_eq!(
        (second.0, second.1),
        (first.0, 1),
        "the second call is cached"
    );
    assert_eq!((third.0, third.1), (first.0, 1), "the third call is cached");
    assert_eq!(
        (fresh.0, fresh.1),
        (first.0, 1),
        "a new context queries once itself"
    );
    println!("MMQ facts: three calls, one query; a fresh context one query");
}

#[test]
#[ignore = "requires two physical CUDA devices"]
fn mmq_facts_follow_the_context_when_another_device_was_selected() {
    const NAME: &str = "mmq_facts_follow_the_context_when_another_device_was_selected";
    assert!(
        testkit::device_present(NAME),
        "{NAME}: CUDA devices are required"
    );
    let admission = crate::family::project::cuda::served_admission();
    let contexts = [
        crate::kernels::cuda::CudaContext::with_admission_on(0, &admission).expect("device 0"),
        crate::kernels::cuda::CudaContext::with_admission_on(1, &admission).expect("device 1"),
    ];
    let live = |device| {
        let _dispatch = crate::kernels::cuda::context::enter_on(device).expect("select device");
        crate::kernels::cuda::native::prefill::mmq_facts::mmq_facts().expect("live facts")
    };
    let expected = [live(0), live(1)];
    // Device 1 is current here. Querying context 0 must still select device 0.
    let observed = [
        contexts[0].mmq_device_facts().expect("context 0 facts"),
        contexts[1].mmq_device_facts().expect("context 1 facts"),
    ];
    for index in 0..2 {
        assert_eq!(observed[index].cc, expected[index].cc, "device {index}");
        assert_eq!(
            observed[index].nsm, expected[index].nsm as usize,
            "device {index}"
        );
        assert_eq!(
            observed[index].smpbo, expected[index].smpbo,
            "device {index}"
        );
    }
}
