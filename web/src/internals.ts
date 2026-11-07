/**
 * The Internals view: what the model actually did while it ran. First
 * section is the router — which experts each layer picked for each
 * token, and with what gate weight. The packet is family-neutral (any
 * MoE family routes top-k over N experts; dense families simply have
 * no routing packet), and its shape is the shape a future
 * /v1/telemetry endpoint would return. Phase 0 renders a recorded
 * fixture (internals.fixture.ts); nothing here talks to the server.
 */

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

  constructor(
    private readonly section: HTMLElement,
    private readonly packet: RoutingTracePacket
  ) {
    this.render();
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
    head.append(h2);
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
}
