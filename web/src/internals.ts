/**
 * The Internals view: what the model actually did while it ran. First
 * section is the router — which experts each layer picked for each
 * token, and with what gate weight. The packet is family-neutral (any
 * MoE family routes top-k over N experts; dense families simply have
 * no routing packet). Phase 0 renders a recorded fixture
 * (internals.fixture.ts); the Live toggle switches to the engine's
 * routing ring via GET /v1/route-trace (Q36_ROUTE_STAGE), transposed
 * into the same shape — decode steps are the tokens.
 */

import { getRouteTrace, ApiError, type RouteTracePacket } from "./api.js";

/** One layer of the trace: per token, the picked expert ids in choice
 * order and their renormalized gates, same order. */
export interface RoutingLayerRow {
  layer: number;
  ids: number[][];
  gates: number[][];
}

export interface RoutingSummary {
  tokens: number;
  layers: number;
  top_k: number;
  experts: number;
  experts_seen: number;
  /** picks per expert id over the whole trace */
  expert_histogram: Record<string, number>;
}

export interface RoutingTracePacket {
  packet: string;
  family: string;
  checkpoint: string;
  activation_dtype: string;
  kv_dtype: string;
  routing: string;
  step_mode: string;
  path_note: string;
  layer_rows: RoutingLayerRow[];
  summary: RoutingSummary;
}

export interface ExpertUsage {
  id: number;
  picks: number;
}

/** The most-selected experts, descending; ties break by expert id so
 * the order is stable. */
export function topExperts(histogram: Record<string, number>, limit: number): ExpertUsage[] {
  return Object.entries(histogram)
    .map(([id, picks]) => ({ id: Number(id), picks }))
    .sort((a, b) => b.picks - a.picks || a.id - b.id)
    .slice(0, limit);
}

/** All picks summed — the denominator for share-of-tokens numbers. */
export function totalPicks(histogram: Record<string, number>): number {
  let total = 0;
  for (const picks of Object.values(histogram)) total += picks;
  return total;
}

/** Adapts the live /v1/route-trace ring into the packet shape this view
 * renders: decode steps become tokens, per-step layer rows are
 * transposed into per-layer token rows, and the histogram is rebuilt.
 * Null when the view cannot render it — no facts (stage never
 * allocated), no steps yet, or a step whose layer rows do not match the
 * facts. */
export function liveRoutingPacket(live: RouteTracePacket): RoutingTracePacket | null {
  if (live.layers === null || live.top_k === null || live.experts === null) return null;
  if (live.steps.length === 0) return null;
  const histogram: Record<string, number> = {};
  const layer_rows: RoutingLayerRow[] = [];
  for (let l = 0; l < live.layers; l++) {
    const ids: number[][] = [];
    const gates: number[][] = [];
    for (const step of live.steps) {
      const row = step.layers[l];
      // A step must carry exactly the facts' layer count: fewer would
      // render holes, more would silently truncate (two shapes in one
      // process) \u2014 both refuse.
      if (row === undefined || step.layers.length !== live.layers) return null;
      ids.push(row.ids);
      gates.push(row.weights);
      for (const id of row.ids) histogram[id] = (histogram[id] ?? 0) + 1;
    }
    layer_rows.push({ layer: l, ids, gates });
  }
  return {
    packet: "route-trace/1",
    family: "live",
    checkpoint: "engine routing ring",
    activation_dtype: "-",
    kv_dtype: "-",
    routing: "top-" + live.top_k,
    step_mode: "decode",
    path_note:
      "Live routing ring (Q36_ROUTE_STAGE): the last " +
      live.steps.length +
      " decode steps, engine-wide. The recorded fixture shows one full reply instead.",
    layer_rows,
    summary: {
      tokens: live.steps.length,
      layers: live.layers,
      top_k: live.top_k,
      experts: live.experts,
      experts_seen: Object.keys(histogram).length,
      expert_histogram: histogram,
    },
  };
}

/** One heatmap cell: the layer's first-choice expert for a token and
 * its gate. Ids are in choice order, so index 0 is the winner. */
export interface HeatCell {
  topId: number;
  topGate: number;
}

export function heatRow(row: RoutingLayerRow): HeatCell[] {
  return row.ids.map((tokenIds, t) => ({
    topId: tokenIds[0] ?? 0,
    topGate: row.gates[t]?.[0] ?? 0,
  }));
}

/** Deterministic cell color: hue by expert id, darker for a stronger
 * gate. Pure so the map renders identically everywhere. */
export function cellColor(topId: number, topGate: number, experts: number): string {
  const hue = (Math.min(Math.max(topId, 0), Math.max(experts - 1, 0)) / Math.max(experts, 1)) * 360;
  const gate = Math.min(Math.max(topGate, 0), 1);
  const light = 78 - gate * 40;
  return "hsl(" + hue.toFixed(1) + " 62% " + light.toFixed(1) + "%)";
}

export interface TokenPick {
  id: number;
  gate: number;
}

export interface TokenLayer {
  layer: number;
  picks: TokenPick[];
}

/** The drill-down column for one token: every layer's full pick list in
 * choice order. Out-of-range token indexes yield empty pick lists, so a
 * short trace never throws from the view. */
export function tokenColumn(packet: RoutingTracePacket, t: number): TokenLayer[] {
  return packet.layer_rows.map((row) => ({
    layer: row.layer,
    picks: (row.ids[t] ?? []).map((id, i) => ({ id, gate: row.gates[t]?.[i] ?? 0 })),
  }));
}

function div(cls: string): HTMLElement {
  const node = document.createElement("div");
  node.className = cls;
  return node;
}

export class InternalsView {
  private selectedToken = 0;
  private detail: HTMLElement | null = null;
  private packet: RoutingTracePacket;
  private readonly fixture: RoutingTracePacket;
  private badge: HTMLElement | null = null;
  private toggle: HTMLButtonElement | null = null;
  private live = false;
  private polling: number | null = null;
  private fetching = false;

  constructor(
    private readonly section: HTMLElement,
    fixture: RoutingTracePacket
  ) {
    this.fixture = fixture;
    this.packet = fixture;
    this.render();
  }

  /** Switches to the live routing ring and polls it every 2 s. Errors
   * keep the last good packet and show the reason in the badge (the
   * stage-off 404 carries the enabling hint). */
  async goLive(): Promise<void> {
    this.live = true;
    this.startPolling();
    await this.refreshLive();
  }

  /** Back to the recorded fixture; stops the poll. */
  useFixture(): void {
    this.live = false;
    this.stopPolling();
    this.packet = this.fixture;
    this.selectedToken = 0;
    this.render();
  }

  private startPolling(): void {
    if (this.polling === null)
      this.polling = window.setInterval(() => void this.refreshLive(), 2000);
  }

  private stopPolling(): void {
    if (this.polling !== null) {
      window.clearInterval(this.polling);
      this.polling = null;
    }
  }

  private async refreshLive(): Promise<void> {
    if (!this.live || this.fetching) return;
    this.fetching = true;
    try {
      try {
        const live = await getRouteTrace();
        const packet = liveRoutingPacket(live);
        if (packet === null) {
          // The two null causes say different things: no facts means no
          // routing stage exists (flag off on a fresh process, or a
          // non-qwen36-native engine); facts with no steps means the
          // stage exists but nothing decoded yet.
          const noStage = live.layers === null;
          this.setBadge(
            noStage
              ? "Live \u2014 no routing stage (qwen36 native layout, Q36_ROUTE_STAGE=1)"
              : "Live \u2014 ring empty (no decode steps yet)"
          );
        } else {
          this.packet = packet;
          this.selectedToken = Math.min(this.selectedToken, packet.summary.tokens - 1);
          this.render();
          this.setBadge("Live \u2014 last " + packet.summary.tokens + " decode steps");
        }
      } catch (err) {
        const note = err instanceof ApiError ? err.message : "server unreachable";
        this.setBadge("Live unavailable \u2014 " + note);
      }
    } finally {
      this.fetching = false;
    }
  }

  private setBadge(text: string): void {
    if (this.badge !== null) this.badge.textContent = text;
  }

  private card(title: string): { card: HTMLElement; body: HTMLElement } {
    const card = div("card");
    const h3 = document.createElement("h3");
    h3.textContent = title;
    const body = div("");
    card.append(h3, body);
    return { card, body };
  }

  private fact(body: HTMLElement, name: string, value: string): void {
    const dt = document.createElement("dt");
    dt.textContent = name;
    const dd = document.createElement("dd");
    dd.textContent = value;
    const grid = body.querySelector("dl") ?? document.createElement("dl");
    if (!grid.isConnected) {
      grid.className = "fact-grid";
      body.append(grid);
    }
    grid.append(dt, dd);
  }

  private render(): void {
    this.section.textContent = "";
    const scroll = div("panel-scroll");
    const column = div("panel-column");
    const head = div("panel-head");
    const h2 = document.createElement("h2");
    h2.textContent = "Internals";
    const badge = document.createElement("span");
    badge.className = "fixture-badge";
    badge.textContent = "Recorded fixture \u2014 not live";
    this.badge = badge;
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "live-toggle";
    toggle.textContent = "Go live";
    toggle.addEventListener("click", () => {
      if (this.live) this.useFixture();
      else void this.goLive();
    });
    this.toggle = toggle;
    head.append(h2, badge, toggle);
    column.append(head);

    const { summary } = this.packet;
    const name = this.packet.checkpoint.split("/").pop() ?? this.packet.checkpoint;
    const { card: traceCard, body: traceBody } = this.card("Trace");
    this.fact(traceBody, "Family", this.packet.family);
    this.fact(traceBody, "Checkpoint", name);
    this.fact(traceBody, "Dtype", this.packet.activation_dtype + " activations, " + this.packet.kv_dtype + " KV");
    this.fact(
      traceBody,
      "Depth",
      summary.tokens + " tokens × " + summary.layers + " layers × top-" + summary.top_k
    );
    this.fact(traceBody, "Experts seen", summary.experts_seen + " of " + summary.experts);
    this.fact(traceBody, "Mode", this.packet.step_mode + " step, " + this.packet.routing + " routing");
    column.append(traceCard);

    column.append(this.histogramCard());
    column.append(this.routingCard());

    const note = document.createElement("p");
    note.className = "panel-hint";
    note.textContent = this.packet.path_note;
    column.append(note);

    scroll.append(column);
    this.section.append(scroll);
  }

  private histogramCard(): HTMLElement {
    const { summary } = this.packet;
    const { card, body } = this.card("Most-selected experts");
    const total = totalPicks(summary.expert_histogram);
    const bars = div("expert-bars");
    for (const expert of topExperts(summary.expert_histogram, 24)) {
      const line = div("expert-bar");
      const lab = div("lab");
      lab.textContent = String(expert.id);
      const bar = div("bar");
      bar.style.width = (expert.picks / (total / 100)).toFixed(2) + "%";
      const n = div("n");
      n.textContent = String(expert.picks);
      line.append(lab, bar, n);
      bars.append(line);
    }
    body.append(bars);
    return card;
  }

  private routingCard(): HTMLElement {
    const { summary } = this.packet;
    const { card, body } = this.card("Routing map — one row per layer, one column per token; click a cell");
    const wrap = div("heat");
    const grid = div("heat-grid");
    grid.style.gridTemplateColumns = "auto repeat(" + summary.tokens + ", 9px)";
    const corner = div("lab");
    corner.textContent = "L";
    grid.append(corner);
    for (const row of this.packet.layer_rows) {
      const label = div("lab");
      label.textContent = String(row.layer);
      grid.append(label);
      const cells = heatRow(row);
      for (let t = 0; t < cells.length; t++) {
        const cell = cells[t];
        if (cell === undefined) continue;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.t = String(t);
        btn.className = "hc" + (t === this.selectedToken ? " sel" : "");
        btn.style.setProperty("--c", cellColor(cell.topId, cell.topGate, summary.experts));
        btn.title = "token " + t + " · layer " + row.layer + " · expert " + cell.topId + " · gate " + cell.topGate.toFixed(3);
        btn.setAttribute("aria-label", btn.title);
        btn.addEventListener("click", () => {
          this.selectedToken = t;
          this.renderDetail();
        });
        grid.append(btn);
      }
    }
    wrap.append(grid);
    this.detail = div("heat-detail");
    body.append(wrap, this.detail);
    this.renderDetail();
    return card;
  }

  private renderDetail(): void {
    this.renderToggleState();
    const node = this.detail;
    if (node === null) return;
    node.replaceChildren();
    const heading = document.createElement("h4");
    heading.textContent = "Token " + this.selectedToken + " — picks per layer (expert · gate)";
    node.append(heading);
    const layers = div("pick-layers");
    for (const layer of tokenColumn(this.packet, this.selectedToken)) {
      const line = div("pick-line");
      const lab = div("lab");
      lab.textContent = "L" + layer.layer;
      const chips = document.createElement("span");
      chips.className = "pick-chips";
      for (const pick of layer.picks) {
        const chip = document.createElement("span");
        chip.className = "pick-chip";
        chip.textContent = pick.id + " · " + pick.gate.toFixed(3);
        chips.append(chip);
      }
      line.append(lab, chips);
      layers.append(line);
    }
    node.append(layers);
    for (const btn of Array.from(this.section.querySelectorAll<HTMLButtonElement>(".hc.sel"))) {
      btn.classList.remove("sel");
    }
    for (const btn of Array.from(this.section.querySelectorAll<HTMLButtonElement>(".hc"))) {
      if (btn.dataset.t === String(this.selectedToken)) btn.classList.add("sel");
    }
  }

  /** The toggle text follows the mode; called after renders and clicks. */
  private renderToggleState(): void {
    if (this.toggle !== null) {
      this.toggle.textContent = this.live ? "Back to fixture" : "Go live";
    }
    if (!this.live && this.badge !== null) {
      this.badge.textContent = "Recorded fixture \u2014 not live";
    }
  }
}
