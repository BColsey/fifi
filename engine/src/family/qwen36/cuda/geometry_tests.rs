//! Unit tests of the mixer geometry and the weight shapes. CPU only.
use super::{
    common_tests::{dims, plan},
    mixer_geometry::{DenseSpec, MixerGeometry, check_dense},
    shard::ShardTable,
    step_plan::Slot,
};
use crate::family::project::cuda::WeightLayout;
use crate::family::{AttentionKind, gguf_model::Role};

#[test]
fn projection_rows_tile_exactly_and_match_the_plan() {
    let d = dims();
    let g = MixerGeometry::new(&d).expect("geometry");
    let x = &g.delta;
    assert_eq!(
        (x.z_off, x.beta_off, x.alpha_off, x.ld),
        (8192, 12288, 12320, 12352)
    );
    assert_eq!(x.alpha_off + x.alpha_rows, x.ld);
    assert_eq!(x.z_off + x.z_rows, x.beta_off);
    assert_eq!(x.beta_off + x.beta_rows, x.alpha_off);
    assert_eq!(
        (x.channels, x.value_width, x.key_heads, x.value_heads),
        (8192, 4096, 16, 32)
    );
    let a = &g.attn;
    assert_eq!(
        (a.q_rows, a.k_off, a.v_off, a.ld, a.q_width),
        (8192, 8192, 8704, 9216, 4096)
    );
    assert_eq!(a.v_off + a.kv_rows, a.ld);
    assert_eq!(a.scale, 0.0625);
    assert_eq!(a.n_rot, 64);
    // The plan sizes DeltaProj and AttnProj from the same widths, one row of f32.
    let p = plan(&d);
    let bytes = |slot| p.tensors()[p.index(slot).expect("slot")].bytes;
    assert_eq!(bytes(Slot::DeltaProj), x.ld * 4);
    assert_eq!(bytes(Slot::AttnProj), a.ld * 4);
}

#[test]
fn geometry_refuses_projection_widths_that_differ_from_the_plan() {
    let mut d = dims();
    d.delta_proj += 8;
    let error = MixerGeometry::new(&d).expect_err("plan width");
    assert!(
        format!("{error:?}").contains("differ from the plan"),
        "{error:?}"
    );
}

#[test]
fn a_conv_kernel_other_than_four_is_refused() {
    let mut d = dims();
    d.shape.delta.conv_kernel = 5;
    let error = MixerGeometry::new(&d).expect_err("the causal conv kernel takes four taps");
    assert!(
        format!("{error:?}").contains("the causal conv kernel takes"),
        "{error:?}"
    );
}

#[test]
fn weight_specs_name_every_role_a_block_needs() {
    let g = MixerGeometry::new(&dims()).expect("geometry");
    let dense = g
        .dense_specs(WeightLayout::Tiled, AttentionKind::DeltaNet)
        .expect("delta");
    let roles: Vec<Role> = dense.iter().map(|s| s.role).collect();
    assert_eq!(
        roles,
        [
            Role::AttnQkv,
            Role::AttnGate,
            Role::SsmBeta,
            Role::SsmAlpha,
            Role::SsmOut
        ]
    );
    assert_eq!(
        dense[4],
        DenseSpec {
            role: Role::SsmOut,
            rows: 2048,
            k: 4096
        }
    );
    let full = g
        .dense_specs(WeightLayout::Tiled, AttentionKind::Full)
        .expect("full");
    assert_eq!(
        (full[0].rows, full[1].rows, full[2].rows, full[3].k),
        (8192, 512, 512, 4096)
    );
    let raw = g.raw_specs(AttentionKind::DeltaNet).expect("raw");
    let conv = raw
        .iter()
        .find(|s| s.role == Role::SsmConv1d)
        .expect("conv");
    assert_eq!(conv.values, 8192 * 4);
    let norm = raw.iter().find(|s| s.role == Role::SsmNorm).expect("norm");
    assert_eq!(norm.values, 128);
    let raw_full = g.raw_specs(AttentionKind::Full).expect("raw full");
    assert!(
        raw_full
            .iter()
            .any(|s| s.role == Role::AttnQNorm && s.values == 256)
    );
    assert!(raw_full.iter().all(|s| s.role != Role::SsmA));
    assert!(
        g.dense_specs(WeightLayout::Tiled, AttentionKind::Qsa)
            .is_err()
    );
}

#[test]
fn dense_weights_must_match_rows_k_and_have_no_stack() {
    let want = DenseSpec {
        role: Role::SsmOut,
        rows: 2048,
        k: 2048,
    };
    check_dense("w", (2048, 2048, None), &want).expect("exact");
    assert!(check_dense("w", (2047, 2048, None), &want).is_err());
    assert!(check_dense("w", (2048, 4096, None), &want).is_err());
    assert!(check_dense("w", (2048, 2048, Some(256)), &want).is_err());
}

#[test]
fn a_one_card_table_yields_the_exact_single_device_geometry() {
    let d = dims();
    let table = ShardTable::new(&d.shape, 1, 32).expect("table");
    let card = MixerGeometry::for_card(&d, &table, 0).expect("card geometry");
    let full = MixerGeometry::new(&d).expect("geometry");
    assert_eq!(card, full);
}

#[test]
fn card_geometries_tile_the_full_geometry() {
    let d = dims();
    let full = MixerGeometry::new(&d).expect("geometry");
    for cards in [2, 3] {
        let table = ShardTable::new(&d.shape, cards, 32).expect("table");
        let sum = |get: &dyn Fn(&MixerGeometry) -> usize| -> usize {
            (0..cards)
                .map(|card| get(&MixerGeometry::for_card(&d, &table, card).expect("card geometry")))
                .sum()
        };
        let g = &full;
        assert_eq!(
            sum(&|c| c.attn.q_rows),
            g.attn.q_rows,
            "{cards} cards: q rows"
        );
        assert_eq!(
            sum(&|c| c.attn.kv_rows),
            g.attn.kv_rows,
            "{cards} cards: kv rows"
        );
        assert_eq!(
            sum(&|c| c.attn.q_width),
            g.attn.q_width,
            "{cards} cards: o columns"
        );
        assert_eq!(
            sum(&|c| c.delta.channels),
            g.delta.channels,
            "{cards} cards: conv channels"
        );
        assert_eq!(
            sum(&|c| c.delta.z_rows),
            g.delta.z_rows,
            "{cards} cards: z rows"
        );
        assert_eq!(
            sum(&|c| c.delta.key_heads),
            g.delta.key_heads,
            "{cards} cards: key heads"
        );
        // The replicated widths stay full on every card.
        for card in 0..cards {
            let c = MixerGeometry::for_card(&d, &table, card).expect("card geometry");
            assert_eq!(c.hidden, g.hidden, "card {card}: residual row");
            assert_eq!(c.attn.heads_kv, g.attn.heads_kv, "card {card}: KV cache");
            assert_eq!(c.attn.head_dim, g.attn.head_dim, "card {card}: head dim");
            assert_eq!(c.attn.n_rot, g.attn.n_rot, "card {card}: rope");
            assert_eq!(
                c.delta.value_dim, g.delta.value_dim,
                "card {card}: value dim"
            );
            // The card's fused rows stay internally tiled, exactly like the
            // single-device row: the offsets close on `ld`.
            assert_eq!(c.delta.alpha_off + c.delta.alpha_rows, c.delta.ld);
            assert_eq!(c.attn.v_off + c.attn.kv_rows, c.attn.ld);
        }
    }
}
