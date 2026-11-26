//! The two fixed-loop Inkling 2 Mini GGUF layouts and where each stores the
//! loop's metadata, controllers and message queries.

use std::borrow::Cow;

use crate::weights::{GgufHeader, MetaValue};

const MAINLINE: &str = "inkling2_loop";

/// The fork advertises logical visits as its block count. Both formats store
/// only the physical layers, but their loop message queries use different names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    Mainline,
    Fork,
}

impl Layout {
    pub(crate) fn from_header(header: &GgufHeader) -> Option<Self> {
        match header.meta("general.architecture") {
            Some(MetaValue::String(arch)) if arch == MAINLINE => Some(Self::Mainline),
            Some(MetaValue::String(arch)) if arch == "inkling2" => Some(Self::Fork),
            _ => None,
        }
    }

    pub(crate) fn architecture(self) -> &'static str {
        match self {
            Self::Mainline => MAINLINE,
            Self::Fork => "inkling2",
        }
    }

    /// Metadata names, under the architecture, of the first core layer, the
    /// layer after the core and the pass count.
    pub(crate) fn loop_keys(self) -> [&'static str; 3] {
        match self {
            Self::Mainline => ["loop.start", "loop.end", "loop.count"],
            Self::Fork => ["loop.core_start", "loop.core_end", "num_loops"],
        }
    }

    /// The block count the header advertises: stored layers or layer visits.
    pub(crate) fn block_count(self, stored: usize, visits: usize) -> usize {
        match self {
            Self::Mainline => stored,
            Self::Fork => visits,
        }
    }

    /// Tensor-name stem of a core layer's controller for one stream, in the
    /// order attention query, key, value, MLP.
    pub(crate) fn controller_stream(self, stream: usize) -> &'static str {
        match self {
            Self::Mainline => ["q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"][stream],
            Self::Fork => ["attn", "k", "v", "mlp"][stream],
        }
    }

    /// The tensor and the row that hold a core layer's loop message query for
    /// one stream, in the order attention query, key, value, MLP.
    pub(crate) fn loop_query(
        self,
        layer: usize,
        core_start: usize,
        stream: usize,
    ) -> (Cow<'static, str>, usize) {
        match self {
            Self::Mainline => (format!("blk.{layer}.loop_msg_query.weight").into(), stream),
            Self::Fork => {
                let (name, offset) = [
                    ("loop_res_query.weight", 0),
                    ("loop_kv_res_query.weight", 0),
                    ("loop_kv_res_query.weight", 1),
                    ("loop_res_query.weight", 1),
                ][stream];
                (name.into(), 2 * (layer - core_start) + offset)
            }
        }
    }

    /// Rows in each tensor that holds loop message queries.
    pub(crate) fn loop_query_rows(self, core_layers: usize) -> usize {
        match self {
            Self::Mainline => 4,
            Self::Fork => 2 * core_layers,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::Layout;

    const CORE: std::ops::Range<usize> = 6..18;

    #[test]
    fn each_layout_places_every_loop_query_in_its_own_row() {
        for layout in [Layout::Mainline, Layout::Fork] {
            let rows = layout.loop_query_rows(CORE.len());
            let mut seen = BTreeSet::new();
            for layer in CORE {
                for stream in 0..4 {
                    let (name, row) = layout.loop_query(layer, CORE.start, stream);
                    assert!(row < rows, "{layout:?} layer {layer} stream {stream}");
                    assert!(
                        seen.insert((name.into_owned(), row)),
                        "{layout:?} reuses a row"
                    );
                }
            }
            let tensors: BTreeSet<_> = seen.iter().map(|(name, _)| name.as_str()).collect();
            assert_eq!(
                seen.len(),
                tensors.len() * rows,
                "{layout:?} leaves a row unused"
            );
        }
    }

    #[test]
    fn the_fork_pairs_query_with_mlp_and_key_with_value() {
        let at = |stream| Layout::Fork.loop_query(9, CORE.start, stream);
        assert_eq!(at(0), ("loop_res_query.weight".into(), 6));
        assert_eq!(at(3), ("loop_res_query.weight".into(), 7));
        assert_eq!(at(1), ("loop_kv_res_query.weight".into(), 6));
        assert_eq!(at(2), ("loop_kv_res_query.weight".into(), 7));
        let mainline = Layout::Mainline.loop_query(9, CORE.start, 2);
        assert_eq!(mainline, ("blk.9.loop_msg_query.weight".into(), 2));
    }
}
