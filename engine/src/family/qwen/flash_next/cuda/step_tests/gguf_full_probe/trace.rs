//! First-token diagnostics: observe the live routed IDs before the arena can reuse their slot.
use super::*;
use crate::family::exl3::{launch::MoeDims, moe::Exl3MoeBufs};

pub(super) struct RouteSnapshot {
    ids: Vec<i32>,
    input: Vec<f32>,
}

pub(super) struct RouteTap<'a, M> {
    inner: M,
    snapshot: &'a mut Option<RouteSnapshot>,
}

impl<'a, M> RouteTap<'a, M> {
    pub(super) fn new(inner: M, snapshot: &'a mut Option<RouteSnapshot>) -> Self {
        Self { inner, snapshot }
    }
}

impl<M: MoeBlock> MoeBlock for RouteTap<'_, M> {
    fn run(
        &mut self,
        launch: &Launch<'_>,
        layer: usize,
        pos: usize,
        bufs: &Exl3MoeBufs<'_>,
        dims: &MoeDims,
    ) -> Result<(), EngError> {
        // Mixed(Mlp) is live here; capture it before the MoE can reuse the arena.
        let input_len = dims
            .hidden
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| EngError::BadRequest("GGUF router input length overflow".into()))?;
        if bufs.route.norm_row.bytes() < input_len {
            return Err(EngError::BadRequest(
                "GGUF router input buffer too short".into(),
            ));
        }
        let mut input_bytes = vec![0; input_len];
        bufs.route.norm_row.buffer().read_at(
            launch.stream,
            bufs.route.norm_row.offset(),
            &mut input_bytes,
        )?;
        let input = input_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_ne_bytes(*x))
            .collect();
        self.inner.run(launch, layer, pos, bufs, dims)?;
        let len = dims
            .top_k
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| EngError::BadRequest("GGUF route trace length overflow".into()))?;
        if bufs.route.ids.bytes() < len {
            return Err(EngError::BadRequest(
                "GGUF route trace buffer too short".into(),
            ));
        }
        let mut bytes = vec![0; len];
        bufs.route
            .ids
            .buffer()
            .read_at(launch.stream, bufs.route.ids.offset(), &mut bytes)?;
        let ids = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| i32::from_ne_bytes(*x))
            .collect();
        *self.snapshot = Some(RouteSnapshot { ids, input });
        Ok(())
    }
}

#[derive(serde::Serialize)]
struct Row {
    layer: usize,
    residual_rel_l2: f64,
    residual_max_abs: f32,
    ordered_match: bool,
    set_match: bool,
    host_ids: Vec<i32>,
    gpu_ids: Vec<i32>,
    gpu_router_input: Vec<f32>,
    gpu_stream: Vec<f32>,
}

pub(super) struct Trace {
    host: Vec<Vec<f32>>,
    routes: Vec<Vec<i32>>,
    rows: Vec<Row>,
}

fn parse_routes(
    text: &str,
    layers: usize,
    top_k: usize,
    experts: usize,
) -> Result<Vec<Vec<i32>>, String> {
    let mut routes = vec![Vec::new(); layers];
    let mut count = 0usize;
    for line in text.lines() {
        let Some((_, entry)) = line.split_once("expert route: name=blk.") else {
            continue;
        };
        let (layer, expert) = entry
            .split_once(".ffn_gate_exps.weight expert=")
            .ok_or("invalid route line")?;
        let layer: usize = layer.parse().map_err(|_| "invalid route layer")?;
        let expert: i32 = expert.trim().parse().map_err(|_| "invalid route expert")?;
        if top_k == 0
            || layer != count / top_k
            || layer >= layers
            || expert < 0
            || expert as usize >= experts
            || routes[layer].contains(&expert)
        {
            return Err("out-of-order, duplicate, or invalid route".into());
        }
        routes[layer].push(expert);
        count += 1;
    }
    if layers.checked_mul(top_k) != Some(count) {
        return Err("missing host routes".into());
    }
    Ok(routes)
}

impl Trace {
    pub(super) fn open(layers: usize, width: usize, top_k: usize, experts: usize) -> Self {
        let path = std::env::var_os("FIFI_FLASH_GGUF_HOST_FIRST_LAYERS")
            .expect("first-token host residuals");
        let route_path =
            std::env::var_os("FIFI_FLASH_GGUF_HOST_FIRST_ROUTES").expect("first-token host routes");
        assert_pinned_file(
            &path,
            "62fdd25ecb4f3e14f2f4417de026421e6e6557c972eadee3e779e9cc6656bace",
        );
        assert_pinned_file(
            &route_path,
            "ac61a12015eee3cbc85bcb71d0acf3f6e5c6a53072381f314853c4a17c2fea45",
        );
        let captured: Vec<Vec<Vec<f32>>> =
            serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
        assert_eq!(captured.len(), 1, "one host token");
        let host = captured.into_iter().next().unwrap();
        assert_eq!(host.len(), layers);
        assert!(
            host.iter()
                .all(|row| row.len() == width && row.iter().all(|x| x.is_finite()))
        );
        let routes = parse_routes(
            &std::fs::read_to_string(route_path).unwrap(),
            layers,
            top_k,
            experts,
        )
        .unwrap();
        Self {
            host,
            routes,
            rows: Vec::with_capacity(layers),
        }
    }

    pub(super) fn record(&mut self, step: &FlashStep<'_>, layer: usize, snapshot: RouteSnapshot) {
        assert_eq!(layer, self.rows.len());
        let RouteSnapshot {
            ids: gpu_ids,
            input: gpu_router_input,
        } = snapshot;
        assert!(gpu_router_input.iter().all(|x| x.is_finite()));
        let host = &self.host[layer];
        let gpu_stream = step
            .read_f32(step.slot(Slot::Stream).unwrap(), host.len())
            .unwrap();
        assert!(gpu_stream.iter().all(|x| x.is_finite()));
        let host_ids = self.routes[layer].clone();
        assert_eq!(gpu_ids.len(), host_ids.len());
        let mut sorted_host = host_ids.clone();
        let mut sorted_gpu = gpu_ids.clone();
        sorted_host.sort_unstable();
        sorted_gpu.sort_unstable();
        let set_match = sorted_host == sorted_gpu;
        let ordered_match = host_ids == gpu_ids;
        let residual_rel_l2 = relative_l2(&gpu_stream, host);
        let residual_max_abs = gpu_stream
            .iter()
            .zip(host)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(residual_rel_l2.is_finite() && residual_max_abs.is_finite());
        eprintln!(
            "full first-token layer={layer} residual_l2={residual_rel_l2:.9} max_abs={residual_max_abs:.9} route_order_match={ordered_match} route_set_match={set_match} gpu_ids={gpu_ids:?} host_ids={host_ids:?}"
        );
        self.rows.push(Row {
            layer,
            residual_rel_l2,
            residual_max_abs,
            ordered_match,
            set_match,
            host_ids,
            gpu_ids,
            gpu_router_input,
            gpu_stream,
        });
    }

    pub(super) fn finish(self, expected_layers: usize) {
        assert_eq!(self.rows.len(), expected_layers);
        assert!(expected_layers <= self.host.len());
        let path = std::env::var_os("FIFI_FLASH_GGUF_FIRST_TRACE_OUT").expect("new trace path");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(file, &self.rows).unwrap();
        eprintln!(
            "first-token trace completed layers={expected_layers}; residual and route comparison is diagnostic, not parity qualification"
        );
    }
}

#[derive(serde::Serialize)]
struct StatefulRow {
    position: usize,
    token: u32,
    layer: usize,
    gpu_ids: Vec<i32>,
    gpu_router_input: Vec<f32>,
    gpu_stream: Vec<f32>,
}

/// Captures the live stream without rerunning layers or changing recurrent state.
pub(super) struct StatefulTrace {
    layers: usize,
    width: usize,
    top_k: usize,
    experts: usize,
    rows: Vec<StatefulRow>,
}

impl StatefulTrace {
    pub(super) fn new(layers: usize, width: usize, top_k: usize, experts: usize) -> Self {
        assert!(layers > 0 && width > 0 && top_k > 0 && top_k <= experts);
        Self {
            layers,
            width,
            top_k,
            experts,
            rows: Vec::new(),
        }
    }

    pub(super) fn record(
        &mut self,
        step: &FlashStep<'_>,
        position: usize,
        token: u32,
        layer: usize,
        snapshot: RouteSnapshot,
    ) {
        assert_eq!(self.rows.len(), position * self.layers + layer);
        let RouteSnapshot { ids, input } = snapshot;
        assert_eq!(ids.len(), self.top_k);
        assert_eq!(input.len(), step.geometry.hidden);
        assert!(input.iter().all(|v| v.is_finite()));
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert!(
            sorted
                .iter()
                .all(|&id| id >= 0 && (id as usize) < self.experts)
        );
        assert!(sorted.windows(2).all(|pair| pair[0] != pair[1]));
        let stream = step
            .read_f32(step.slot(Slot::Stream).unwrap(), self.width)
            .unwrap();
        assert!(stream.iter().all(|v| v.is_finite()));
        self.rows.push(StatefulRow {
            position,
            token,
            layer,
            gpu_ids: ids,
            gpu_router_input: input,
            gpu_stream: stream,
        });
    }

    pub(super) fn finish(self, tokens: usize) {
        assert_eq!(self.rows.len(), tokens * self.layers);
        let path = std::env::var_os("FIFI_FLASH_GGUF_STATEFUL_GPU_OUT").expect("new trace path");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(file, &self.rows).unwrap();
        eprintln!(
            "stateful GPU trace completed positions={tokens} layers={}; diagnostic collection only",
            self.layers
        );
    }
}

#[cfg(test)]
mod tests {
    use super::parse_routes;
    #[test]
    fn route_parser_requires_contiguous_unique_valid_groups() {
        let ok = "test ... expert route: name=blk.0.ffn_gate_exps.weight expert=2\nexpert route: name=blk.0.ffn_gate_exps.weight expert=1\nexpert route: name=blk.1.ffn_gate_exps.weight expert=0\nexpert route: name=blk.1.ffn_gate_exps.weight expert=3\n";
        assert_eq!(
            parse_routes(ok, 2, 2, 4).unwrap(),
            vec![vec![2, 1], vec![0, 3]]
        );
        assert!(parse_routes(ok, 2, 3, 4).is_err());
        assert!(parse_routes(&ok.replace("expert=1", "expert=2"), 2, 2, 4).is_err());
        assert!(parse_routes(&ok.replace("blk.1", "blk.0"), 2, 2, 4).is_err());
        assert!(parse_routes(ok, 2, 2, 3).is_err());
    }
}
