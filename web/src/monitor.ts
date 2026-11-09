import {
  ApiError,
  getHealth,
  getReady,
  getSystem,
  getTelemetry,
  listModels,
  type Model,
  type RecentRequest,
  type SystemInfo,
  type TelemetryInfo,
} from "./api";
import { formatBytes, formatClock, formatDate, formatTokens } from "./format";
import {
  applyProbe,
  engineErrorLines,
  engineStageLine,
  prefixHitRate,
  refreshRecentRows,
  requestTotals,
  settingsLines,
  sparkArea,
  sparkPath,
  systemEngineFor,
  type PanelData,
} from "./panel";
import { groupSpeed, speedSummary } from "./speed";
import { appendPoint, HISTORY_KEY, parseHistory, serialize, values, type SparkPoint } from "./sparkstore";
import { downloadText } from "./copy";

export type MonitorData = PanelData;

const HISTORY_CAP = 24;

export class MonitorView {
  private readonly section: HTMLElement;
  /** Set while any refresh is in flight; a second call returns the last
   * data instead of stacking fetches (N3). */
  private refreshing = false;
  private data: MonitorData = {
    healthOk: false,
    readyOk: false,
    notReadyCode: null,
    models: null,
    modelsError: null,
    system: null,
    systemError: null,
    recent: null,
    recentError: null,
    checkedAt: 0,
  };
  /** Client-side sparkline history: medians from each sweep, oldest
   * first. The server keeps no history series, so the trend is built
   * here, one point per poll, exactly like Strata's history ring. */
  private decodeHistory: (number | null)[] = [];
  private ttftHistory: (number | null)[] = [];
  private decodePoints: SparkPoint[] = [];
  private ttftPoints: SparkPoint[] = [];
  /** The request shown in the detail drawer, when one is selected. */
  private selectedRequest: RecentRequest | null = null;
  /** GPU telemetry from /v1/telemetry; null when unprobed or down. */
  private telemetry: TelemetryInfo | null = null;
  private telemetryError: string | null = null;

  constructor(section: HTMLElement) {
    this.section = section;
    try {
      const stored = parseHistory(window.localStorage.getItem(HISTORY_KEY), Date.now());
      this.decodePoints = stored.decode;
      this.ttftPoints = stored.ttft;
      this.decodeHistory = values(stored.decode);
      this.ttftHistory = values(stored.ttft);
    } catch {
      // Storage unavailable (private mode, quota): history stays in memory.
    }
    this.render();
  }

  /** One probe of the public routes plus the models and recent-requests
   * routes. Every fetch carries the app's 4 s timeout, so a hung server
   * cannot stall the poll or the view. The poll calls this on every
   * branch — server up, starting, or down — so health, readiness, and
   * the ring share one truth and one timestamp (a recent-only
   * poll leg kept last-good health next to a current time). */
  async refresh(): Promise<MonitorData> {
    if (this.refreshing) return this.data;
    this.refreshing = true;
    try {
      return await this.sweep();
    } finally {
      this.refreshing = false;
    }
  }

  private async sweep(): Promise<MonitorData> {
    let healthOk = false;
    try {
      await getHealth(AbortSignal.timeout(4000));
      healthOk = true;
    } catch {
      healthOk = false;
    }
    let readyOk = false;
    let notReadyCode: string | null = null;
    try {
      await getReady(AbortSignal.timeout(4000));
      readyOk = true;
    } catch (err) {
      // The contract gives /ready exactly two answers: 200 ready and 503
      // not_ready. Anything else, including an unreachable server, is a
      // non-answer and must not display a failure code.
      notReadyCode = err instanceof ApiError && err.code === "not_ready" ? "not_ready" : null;
    }
    let models: Model[] | null = null;
    let modelsError: string | null = null;
    if (healthOk) {
      try {
        models = (await listModels(AbortSignal.timeout(4000))).data;
      } catch (err) {
        modelsError = err instanceof ApiError ? err.code : "network";
      }
    }
    // The system leg carries the retained settings rows; it degrades
    // the same way the models leg does — null data plus a cause, never
    // last-good rows beside a fresh timestamp.
    let system: SystemInfo | null = null;
    let systemError: string | null = null;
    if (healthOk) {
      try {
        system = await getSystem(AbortSignal.timeout(4000));
      } catch (err) {
        systemError = err instanceof ApiError ? err.code : "network";
      }
    }
    // The recent-rows fetch-and-reduce is the panel module's one
    // implementation; the sweep calls it only from a reachable server —
    // a dead one gets null rows, and applyProbe's full replacement keeps
    // no last-good field.
    const panel = healthOk
      ? await refreshRecentRows()
      : { recent: null, recentError: null };
    if (healthOk) {
      try {
        this.telemetry = await getTelemetry(AbortSignal.timeout(4000));
        this.telemetryError = null;
      } catch (err) {
        this.telemetry = null;
        this.telemetryError = err instanceof ApiError ? err.code : "network";
      }
    } else {
      this.telemetry = null;
      this.telemetryError = null;
    }
    this.data = applyProbe({
      healthOk,
      readyOk,
      notReadyCode,
      models,
      modelsError,
      system,
      systemError,
      recent: panel.recent,
      recentError: panel.recentError,
    });
    if (this.data.recent !== null) {
      const speed = speedSummary(this.data.recent);
      this.decodeHistory = [...this.decodeHistory, speed.decodePerSecond].slice(-HISTORY_CAP);
      this.ttftHistory = [...this.ttftHistory, speed.firstTokenMs].slice(-HISTORY_CAP);
      this.decodePoints = appendPoint(this.decodePoints, this.data.checkedAt, speed.decodePerSecond);
      this.ttftPoints = appendPoint(this.ttftPoints, this.data.checkedAt, speed.firstTokenMs);
      this.decodeHistory = values(this.decodePoints);
      this.ttftHistory = values(this.ttftPoints);
      try {
        window.localStorage.setItem(
          HISTORY_KEY,
          serialize({ decode: this.decodePoints, ttft: this.ttftPoints })
        );
      } catch {
        // Persistence is best-effort; the in-memory trend still works.
      }
    }
    this.render();
    return this.data;
  }

  private card(title: string, sub?: string): { card: HTMLElement; body: HTMLElement } {
    const card = document.createElement("div");
    card.className = "card";
    const h3 = document.createElement("h3");
    h3.textContent = title;
    card.append(h3);
    if (sub !== undefined) {
      const p = document.createElement("p");
      p.className = "sub";
      p.textContent = sub;
      card.append(p);
    }
    const body = document.createElement("div");
    card.append(body);
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

  private hint(body: HTMLElement, text: string): void {
    const hint = document.createElement("p");
    hint.className = "panel-hint";
    hint.textContent = text;
    body.append(hint);
  }

  /** A labelled progress bar: the Strata monitor's core visual unit. */
  private bar(body: HTMLElement, label: string, valueText: string, fraction: number | null, tone?: string): void {
    const row = document.createElement("div");
    row.className = "bar-row";
    const name = document.createElement("span");
    name.textContent = label;
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = valueText;
    row.append(name, value);
    const track = document.createElement("div");
    track.className = "bar-track";
    if (tone !== undefined) track.dataset.tone = tone;
    track.setAttribute("role", "progressbar");
    track.setAttribute("aria-label", label);
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = fraction === null ? "0%" : Math.min(100, Math.max(0, fraction * 100)).toFixed(1) + "%";
    track.append(fill);
    body.append(row, track);
  }

  /** An inline-SVG sparkline in a 100x32 viewBox. */
  private spark(body: HTMLElement, values: readonly (number | null)[], label: string): void {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 100 32");
    svg.setAttribute("preserveAspectRatio", "none");
    svg.setAttribute("class", "spark");
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", label);
    const area = document.createElementNS("http://www.w3.org/2000/svg", "path");
    area.setAttribute("d", sparkArea(values));
    area.setAttribute("class", "spark-area");
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", sparkPath(values));
    line.setAttribute("class", "spark-line");
    svg.append(area, line);
    body.append(svg);
  }

  private statCard(parent: HTMLElement, label: string, value: string, sub: string, state?: string): void {
    const card = document.createElement("div");
    card.className = "stat-card";
    const labelEl = document.createElement("div");
    labelEl.className = "stat-label";
    if (state !== undefined) {
      const dot = document.createElement("span");
      dot.className = "stat-dot s-" + state;
      dot.setAttribute("aria-hidden", "true");
      labelEl.append(dot);
    }
    labelEl.append(document.createTextNode(label));
    const valueEl = document.createElement("div");
    valueEl.className = "stat-value";
    valueEl.textContent = value;
    const subEl = document.createElement("div");
    subEl.className = "stat-sub";
    subEl.textContent = sub;
    card.append(labelEl, valueEl, subEl);
    parent.append(card);
  }

  private render(): void {
    this.section.textContent = "";
    const scroll = document.createElement("div");
    scroll.className = "panel-scroll";
    const column = document.createElement("div");
    column.className = "panel-column panel-wide";
    const head = document.createElement("div");
    head.className = "panel-head";
    const h2 = document.createElement("h2");
    h2.textContent = "Monitor";
    const live = document.createElement("span");
    live.className = "live-badge" + (this.data.healthOk ? " live-on" : "");
    live.textContent =
      this.data.checkedAt === 0
        ? "Not checked yet"
        : (this.data.healthOk ? "Live" : "No answer") + " · " + formatClock(this.data.checkedAt) + " · refreshes every 5 s";
    const refresh = document.createElement("button");
    refresh.className = "btn";
    refresh.type = "button";
    refresh.textContent = "Refresh";
    refresh.addEventListener("click", () => void this.refresh());
    head.append(h2, live, refresh);
    column.append(head);

    if (!this.data.healthOk && this.data.checkedAt !== 0) {
      const alert = document.createElement("div");
      alert.className = "monitor-alert";
      alert.setAttribute("role", "status");
      alert.textContent = "Server not answering. Check that enginesrv is running, then Refresh.";
      column.append(alert);
    }

    const engineErrors = (this.data.models ?? [])
      .flatMap((m) => m.engines ?? [])
      .filter((e) => e.state === "error");
    const queuedTotal = (this.data.system?.engines ?? []).reduce((sum, e) => sum + (e.queued ?? 0), 0);
    if (engineErrors.length > 0 || queuedTotal >= 4) {
      const alert = document.createElement("div");
      alert.className = "monitor-alert";
      alert.setAttribute("role", "alert");
      const parts: string[] = [];
      if (engineErrors.length > 0) {
        parts.push("Engine error: " + engineErrors.map((e) => e.id).join(", ") + ". See Models & engines.");
      }
      if (queuedTotal >= 4) parts.push("Queue building: " + queuedTotal + " requests waiting.");
      alert.textContent = parts.join(" ");
      column.append(alert);
    }

    column.append(this.renderOverview());
    const row = document.createElement("div");
    row.className = "monitor-grid";
    row.append(this.renderSpeedCard(), this.renderSystemCard());
    column.append(row);
    column.append(this.renderRatesCard());
    column.append(this.renderGpuCard());
    column.append(this.renderEnginesCard());
    column.append(this.renderRequestsCard());

    scroll.append(column);
    this.section.append(scroll);
  }

  private renderOverview(): HTMLElement {
    const grid = document.createElement("div");
    grid.className = "stat-grid";
    const engines = (this.data.models ?? []).flatMap((m) => m.engines ?? []);
    const readyCount = engines.filter((e) => e.state === "ready" || e.state === "busy").length;
    const busyCount = engines.filter((e) => e.state === "busy").length;
    const totals = this.data.recent === null ? null : requestTotals(this.data.recent);
    this.statCard(
      grid,
      "Server",
      this.data.healthOk ? "Running" : "No answer",
      this.data.checkedAt === 0 ? "Not checked yet" : "Checked " + formatClock(this.data.checkedAt),
      this.data.healthOk ? "ready" : "offline"
    );
    this.statCard(
      grid,
      "Readiness",
      this.data.readyOk ? "Ready" : this.data.notReadyCode ?? "Not ready",
      this.data.readyOk ? "An engine accepts requests" : "No engine accepts requests yet",
      this.data.readyOk ? "ready" : "busy"
    );
    this.statCard(
      grid,
      "Engines",
      engines.length === 0 ? "—" : readyCount + " / " + engines.length + " ready",
      busyCount > 0 ? busyCount + " busy now" : engines.length === 0 ? "No model list yet" : "None busy",
      busyCount > 0 ? "busy" : "ready"
    );
    this.statCard(
      grid,
      "Requests kept",
      totals === null ? "—" : String(totals.total),
      totals === null
        ? "No answer from the server yet"
        : totals.ok + " ok · " + totals.errors + " errors · " + totals.cancelled + " stopped",
      totals !== null && totals.errors > 0 ? "error" : "ready"
    );
    return grid;
  }

  private renderSpeedCard(): HTMLElement {
    const { card, body } = this.card("Live speed", "Medians over finished requests in the retained ring.");
    if (this.data.recentError !== null) {
      this.hint(
        body,
        this.data.recentError === "unauthorized"
          ? "Set the API key in About to see the speed panel."
          : "The recent-requests route failed: " + this.data.recentError
      );
      return card;
    }
    if (this.data.recent === null) {
      this.hint(body, "No answer from the server yet.");
      return card;
    }
    const speed = speedSummary(this.data.recent);
    const totals = requestTotals(this.data.recent);
    const big = document.createElement("div");
    big.className = "speed-big";
    const decode = document.createElement("div");
    const decodeValue = document.createElement("div");
    decodeValue.className = "speed-value";
    decodeValue.textContent = speed.decodePerSecond === null ? "—" : speed.decodePerSecond.toFixed(1);
    const decodeLabel = document.createElement("div");
    decodeLabel.className = "stat-sub";
    decodeLabel.textContent = "tok/s decode (median)";
    decode.append(decodeValue, decodeLabel);
    const ttft = document.createElement("div");
    const ttftValue = document.createElement("div");
    ttftValue.className = "speed-value";
    ttftValue.textContent = speed.firstTokenMs === null ? "—" : Math.round(speed.firstTokenMs) + " ms";
    const ttftLabel = document.createElement("div");
    ttftLabel.className = "stat-sub";
    ttftLabel.textContent = "First token (median)";
    ttft.append(ttftValue, ttftLabel);
    big.append(decode, ttft);
    body.append(big);
    this.spark(body, this.decodeHistory, "Decode speed trend");
    const sparkLabel = document.createElement("div");
    sparkLabel.className = "stat-sub";
    sparkLabel.textContent = "Decode trend · one point per refresh · TTFT median now " +
      (speed.firstTokenMs === null ? "—" : Math.round(speed.firstTokenMs) + " ms");
    body.append(sparkLabel);
    this.fact(body, "Finished ok", String(totals.ok) + " of " + totals.total);
    this.fact(body, "Prompt tokens", formatTokens(totals.promptTokens) +
      (totals.reuseShare === null ? "" : " · " + Math.round(totals.reuseShare * 100) + "% reused"));
    this.fact(body, "Output tokens", formatTokens(totals.completionTokens));
    const queued = (this.data.system?.engines ?? []).reduce((sum, e) => sum + (e.queued ?? 0), 0);
    const busy = (this.data.models ?? []).flatMap((m) => m.engines ?? []).filter((e) => e.state === "busy").length;
    this.fact(body, "In flight now", String(busy + queued) + " (" + busy + " running · " + queued + " queued)");
    return card;
  }

  /** Prefill and decode medians per model and per engine, measured from
   * the ring rows' own prefill_ms and decode rate. Groups with no
   * finished rows show counts only. */
  private renderRatesCard(): HTMLElement {
    const { card, body } = this.card("Rates by model, engine, and session", "Medians over finished requests. Prefill uses the engine's measured prefill time.");
    if (this.data.recent === null) {
      this.hint(body, this.data.recentError !== null ? "The recent-requests route failed: " + this.data.recentError : "No answer from the server yet.");
      return card;
    }
    if (this.data.recent.length === 0) {
      this.hint(body, "No requests yet.");
      return card;
    }
    const byModel = groupSpeed(this.data.recent, (row) => row.model ?? "unknown model");
    const byEngine = groupSpeed(this.data.recent, (row) => row.engine ?? "unknown engine");
    const bySession = groupSpeed(this.data.recent, (row) => row.session ?? "no session");
    const wrap = document.createElement("div");
    wrap.className = "req-table-wrap";
    const table = document.createElement("table");
    table.className = "req-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const [label, numeric] of [
      ["Group", false], ["Requests", true], ["Prefill tok/s", true],
      ["Decode tok/s", true], ["Queue", true], ["First token", true],
    ] as Array<[string, boolean]>) {
      const th = document.createElement("th");
      th.textContent = label;
      if (numeric) th.className = "num";
      headRow.append(th);
    }
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const [kind, groups] of [["Model", byModel], ["Engine", byEngine], ["Session", bySession]] as const) {
      for (const g of groups) {
        const tr = document.createElement("tr");
        const name = document.createElement("td");
        name.className = "mono req-model";
        name.textContent = kind + " · " + g.key;
        tr.append(name);
        const cells = [
          String(g.total),
          g.prefillPerSecond === null ? "—" : Math.round(g.prefillPerSecond).toLocaleString() + " tok/s",
          g.decodePerSecond === null ? "—" : g.decodePerSecond.toFixed(1) + " tok/s",
          g.queueWaitMs === null ? "—" : Math.round(g.queueWaitMs) + " ms",
          g.firstTokenMs === null ? "—" : Math.round(g.firstTokenMs) + " ms",
        ];
        for (const text of cells) {
          const td = document.createElement("td");
          td.className = "num";
          td.textContent = text;
          tr.append(td);
        }
        tbody.append(tr);
      }
    }
    table.append(thead, tbody);
    wrap.append(table);
    body.append(wrap);
    return card;
  }

  /** One block per GPU from /v1/telemetry: utilisation and VRAM bars,
   * temperature and power as facts. The probe is a subprocess on the
   * server; its failure renders as a stated reason, not empty bars. */
  private renderGpuCard(): HTMLElement {
    const { card, body } = this.card("GPUs", "Per-card utilisation, VRAM, temperature, and power.");
    if (this.telemetryError !== null) {
      this.hint(
        body,
        this.telemetryError === "unauthorized"
          ? "Set the API key in About to see GPU telemetry."
          : "The telemetry route failed: " + this.telemetryError
      );
      return card;
    }
    if (this.telemetry === null) {
      this.hint(body, "No answer from the server yet.");
      return card;
    }
    if (this.telemetry.gpus.length === 0) {
      this.hint(body, "No GPU data. " + (this.telemetry.error ?? "The probe returned no rows."));
      return card;
    }
    for (const gpu of this.telemetry.gpus) {
      const block = document.createElement("div");
      block.className = "model-block";
      const name = document.createElement("div");
      name.className = "mono";
      name.textContent = "GPU " + (gpu.index ?? "?") + " · " + (gpu.name ?? "unknown");
      block.append(name);
      if (gpu.util_percent !== null) {
        this.bar(block, "Utilisation", Math.round(gpu.util_percent) + "%", gpu.util_percent / 100);
      }
      if (gpu.mem_used_bytes !== null && gpu.mem_total_bytes !== null && gpu.mem_total_bytes > 0) {
        this.bar(block, "VRAM", formatBytes(gpu.mem_used_bytes) + " / " + formatBytes(gpu.mem_total_bytes),
          gpu.mem_used_bytes / gpu.mem_total_bytes,
          gpu.mem_used_bytes / gpu.mem_total_bytes > 0.92 ? "danger" : undefined);
      }
      const facts = document.createElement("div");
      facts.className = "cache-facts";
      facts.textContent =
        (gpu.temp_c === null ? "temp —" : "temp " + Math.round(gpu.temp_c) + " °C") +
        " · " + (gpu.power_w === null ? "power —" : "power " + Math.round(gpu.power_w) + " W");
      block.append(facts);
      body.append(block);
    }
    const tie = document.createElement("p");
    tie.className = "panel-hint";
    tie.textContent = "Which engine runs on which card is not reported by the server.";
    body.append(tie);
    return card;
  }

  private renderSystemCard(): HTMLElement {
    const { card, body } = this.card("System & memory");
    if (this.data.systemError !== null) {
      this.hint(
        body,
        this.data.systemError === "unauthorized"
          ? "Set the API key in About to see system facts."
          : "The system route failed: " + this.data.systemError
      );
      return card;
    }
    const system: SystemInfo | null = this.data.system;
    if (system === null) {
      this.hint(body, "No answer from the server yet.");
      return card;
    }
    this.fact(body, "Version", system.version);
    this.fact(body, "Backends", system.backends_compiled.length === 0 ? "none compiled" : system.backends_compiled.join(", "));
    this.fact(body, "CPU threads", String(system.host.cpu_threads));
    const total = system.host.mem_total_bytes;
    const avail = system.host.mem_available_bytes;
    if (typeof total === "number" && typeof avail === "number" && total > 0) {
      const used = total - avail;
      this.bar(body, "System RAM", formatBytes(used) + " / " + formatBytes(total), used / total,
        used / total > 0.92 ? "danger" : undefined);
      this.fact(body, "Available", formatBytes(avail));
    } else {
      const reason = (v: number | { unavailable: string }): string =>
        typeof v === "number" ? formatBytes(v) : "unavailable (" + v.unavailable + ")";
      this.fact(body, "RAM total", reason(total));
      this.fact(body, "RAM available", reason(avail));
    }
    for (const engine of system.engines) {
      this.fact(body, "Context · " + engine.id, engine.context_limit > 0 ? formatTokens(engine.context_limit) + " tokens" : "—");
    }
    return card;
  }

  private renderEnginesCard(): HTMLElement {
    const { card, body } = this.card("Models & engines", "State, load stage, cache, and the settings each engine started with.");
    if (this.data.modelsError !== null) {
      this.hint(
        body,
        this.data.modelsError === "unauthorized"
          ? "Set the API key in About to see the engines."
          : "The models route failed: " + this.data.modelsError
      );
      return card;
    }
    if (this.data.models === null) {
      this.hint(body, "No answer from the server yet.");
      return card;
    }
    if (this.data.models.length === 0) {
      this.hint(body, "The server lists no models.");
      return card;
    }
    for (const model of this.data.models) this.renderModel(body, model);
    return card;
  }

  private renderRequestsCard(): HTMLElement {
    const { card, body } = this.card("Recent requests", "Newest first. The server keeps the last 64; prompt and reply text are never shown.");
    if (this.data.recentError !== null) {
      this.hint(
        body,
        this.data.recentError === "unauthorized"
          ? "Set the API key in About to see requests."
          : "The recent-requests route failed: " + this.data.recentError
      );
      return card;
    }
    if (this.data.recent === null) {
      this.hint(body, "No answer from the server yet.");
      return card;
    }
    if (this.data.recent.length === 0) {
      this.hint(body, "No requests yet. Chat replies and API calls will appear here.");
      return card;
    }
    const wrap = document.createElement("div");
    wrap.className = "req-table-wrap";
    const table = document.createElement("table");
    table.className = "req-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const [label, numeric] of [
      ["Time", false],
      ["Model", false],
      ["Result", false],
      ["Prompt", true],
      ["Reused", true],
      ["Output", true],
      ["Queue", true],
      ["Prefill", true],
      ["First token", true],
      ["Speed", true],
    ] as Array<[string, boolean]>) {
      const th = document.createElement("th");
      th.textContent = label;
      if (numeric) th.className = "num";
      headRow.append(th);
    }
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const row of this.data.recent.slice(0, 12)) {
      tbody.append(this.requestRow(row));
    }
    table.append(thead, tbody);
    wrap.append(table);
    body.append(wrap);
    const exportBtn = document.createElement("button");
    exportBtn.type = "button";
    exportBtn.className = "btn";
    exportBtn.textContent = "Download requests as JSONL";
    const rows = this.data.recent;
    exportBtn.addEventListener("click", () => {
      downloadText("recent-requests.jsonl", rows.map((r) => JSON.stringify(r)).join("\n") + "\n", "application/json");
    });
    body.append(exportBtn);
    if (this.selectedRequest !== null) {
      body.append(this.requestDetail(this.selectedRequest));
    }
    const totals = requestTotals(this.data.recent);
    const foot = document.createElement("p");
    foot.className = "req-totals";
    foot.textContent =
      "Kept: " + totals.total + " · " + totals.ok + " ok · " + totals.errors + " errors · " +
      totals.cancelled + " stopped · " + formatTokens(totals.promptTokens) + " prompt (" +
      formatTokens(totals.reusedTokens) + " reused) · " + formatTokens(totals.completionTokens) + " out";
    body.append(foot);
    return card;
  }

  private requestDetail(row: RecentRequest): HTMLElement {
    const detail = document.createElement("div");
    detail.className = "request-detail";
    const head = document.createElement("div");
    head.className = "code-head";
    const title = document.createElement("span");
    title.className = "code-lang";
    title.textContent = "Request detail";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "code-copy";
    close.textContent = "Close";
    close.addEventListener("click", () => {
      this.selectedRequest = null;
      this.render();
    });
    const copyJson = document.createElement("button");
    copyJson.type = "button";
    copyJson.className = "code-copy";
    copyJson.textContent = "Copy JSON";
    copyJson.addEventListener("click", () => {
      void navigator.clipboard?.writeText(JSON.stringify(row, null, 2)).catch(() => undefined);
      copyJson.textContent = "Copied";
      window.setTimeout(() => (copyJson.textContent = "Copy JSON"), 1500);
    });
    head.append(title, copyJson, close);
    detail.append(head);
    const dl = document.createElement("dl");
    dl.className = "fact-grid";
    detail.append(dl);
    const host = document.createElement("div");
    host.append(dl);
    const facts: Array<[string, string]> = [
      ["Time", row.started_at_ms === undefined ? "—" : new Date(row.started_at_ms).toLocaleString()],
      ["Model", row.model],
      ["Engine", row.engine ?? "—"],
      ["Session", row.session ?? "—"],
      ["Result", row.outcome + (row.error_code !== null ? " · " + row.error_code : "") + (row.finish_reason ? " · " + row.finish_reason : "")],
      ["HTTP status", row.status === null ? "—" : String(row.status)],
      ["Stream", row.stream ? "yes" : "no"],
      ["Prompt tokens", String(row.prompt_tokens)],
      ["Reused tokens", String(row.reused_tokens)],
      ["Output tokens", String(row.completion_tokens)],
      ["Queue wait", row.queue_wait_ms == null ? "—" : Math.round(row.queue_wait_ms) + " ms"],
      ["Prefill", row.prefill_ms == null ? "—" : Math.round(row.prefill_ms) + " ms"],
      ["First token", row.time_to_first_token_ms === null ? "—" : Math.round(row.time_to_first_token_ms) + " ms"],
      ["Decode", row.decode_tokens_per_second === null ? "—" : row.decode_tokens_per_second.toFixed(1) + " tok/s"],
    ];
    for (const [name, value] of facts) {
      const dt = document.createElement("dt");
      dt.textContent = name;
      const dd = document.createElement("dd");
      dd.textContent = value;
      dl.append(dt, dd);
    }
    detail.append(host);
    return detail;
  }

  private requestRow(row: RecentRequest): HTMLTableRowElement {
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    tr.className = "req-row-clickable";
    tr.setAttribute("aria-label", "Show request detail");
    const open = (): void => {
      this.selectedRequest = row;
      this.render();
    };
    tr.addEventListener("click", open);
    tr.addEventListener("keydown", (e) => {
      if (e.key === "Enter") open();
    });
    const time = document.createElement("td");
    time.className = "num";
    time.textContent = row.started_at_ms === undefined ? "—" : formatClock(row.started_at_ms);
    tr.append(time);
    const model = document.createElement("td");
    model.className = "mono req-model";
    model.textContent = row.model + (row.engine ? " · " + row.engine : "") + (row.stream ? " · stream" : "");
    const result = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "outcome-badge o-" + row.outcome;
    badge.textContent =
      row.outcome === "ok"
        ? row.finish_reason === "length"
          ? "Max tokens"
          : "Done"
        : row.outcome === "error"
          ? "Error" + (row.error_code !== null ? " · " + row.error_code : "")
          : "Stopped";
    result.append(badge);
    if (row.status !== null && row.outcome === "error") {
      const status = document.createElement("span");
      status.className = "req-status";
      status.textContent = " HTTP " + row.status;
      result.append(status);
    }
    tr.append(model, result);
    const cells: Array<[string, string]> = [
      [formatTokens(row.prompt_tokens), "num"],
      [formatTokens(row.reused_tokens), "num"],
      [formatTokens(row.completion_tokens), "num"],
      [row.queue_wait_ms === null || row.queue_wait_ms === undefined ? "—" : Math.round(row.queue_wait_ms) + " ms", "num"],
      [row.prefill_ms === null || row.prefill_ms === undefined ? "—" : Math.round(row.prefill_ms) + " ms", "num"],
      [row.time_to_first_token_ms === null ? "—" : Math.round(row.time_to_first_token_ms) + " ms", "num"],
      [row.decode_tokens_per_second === null ? "—" : row.decode_tokens_per_second.toFixed(1) + " tok/s", "num"],
    ];
    for (const [text, cls] of cells) {
      const td = document.createElement("td");
      td.className = cls;
      td.textContent = text;
      tr.append(td);
    }
    return tr;
  }

  private renderModel(body: HTMLElement, model: Model): void {
    const block = document.createElement("div");
    block.className = "model-block";
    const name = document.createElement("div");
    name.className = "mono";
    name.textContent = model.id;
    const sub = document.createElement("div");
    sub.className = "cache-facts";
    sub.textContent =
      "served since " + formatDate(model.created * 1000) +
      (model.owned_by !== "" ? " · " + model.owned_by : "");
    block.append(name, sub);

    const engines = model.engines ?? [];
    if (engines.length === 0) {
      const none = document.createElement("div");
      none.className = "cache-facts";
      none.textContent = "no engines hold this model";
      block.append(none);
    }
    for (const engine of engines) {
      const row = document.createElement("div");
      row.className = "engine-row";
      const id = document.createElement("span");
      id.className = "engine-id";
      id.textContent = engine.id;
      const chip = document.createElement("span");
      chip.className = "state-chip s-" + engine.state;
      chip.textContent = engine.state;
      row.append(id, chip);
      block.append(row);
      const stage = engineStageLine(engine);
      if (stage !== null) {
        const stageLine = document.createElement("div");
        stageLine.className = "cache-facts";
        stageLine.textContent = stage;
        block.append(stageLine);
      }
      for (const line of engineErrorLines(engine)) {
        const failure = document.createElement("div");
        failure.className = "cache-facts engine-error";
        failure.textContent = line;
        block.append(failure);
      }
      // The retained settings rows from /v1/system, joined by engine
      // id: what this engine actually started with. Collapsed by
      // default — it is reference material, not status.
      const systemEngine = systemEngineFor(this.data.system, engine.id);
      if (systemEngine !== null) {
        const ctx = document.createElement("div");
        ctx.className = "cache-facts";
        ctx.textContent = "context limit " + formatTokens(systemEngine.context_limit) + " tokens" +
          (systemEngine.queued !== undefined ? " · " + systemEngine.queued + " queued" : "");
        block.append(ctx);
      }
      if (engine.cache !== undefined) {
        const facts = document.createElement("div");
        facts.className = "cache-facts";
        facts.textContent =
          "KV " + formatBytes(engine.cache.kv_bytes) +
          " · prefix hits " + formatTokens(engine.cache.prefix_hits) +
          " · misses " + formatTokens(engine.cache.prefix_misses) +
          (engine.cache.snapshot_bytes !== undefined
            ? " · snapshots " + formatBytes(engine.cache.snapshot_bytes)
            : "");
        block.append(facts);
        const rate = prefixHitRate(engine.cache);
        if (rate !== null) {
          const barHost = document.createElement("div");
          this.bar(barHost, "Prefix hit rate", Math.round(rate * 100) + "%", rate);
          block.append(barHost);
        }
      }
      if (systemEngine !== null && systemEngine.settings.length > 0) {
        const details = document.createElement("details");
        details.className = "cache-facts engine-settings";
        const summary = document.createElement("summary");
        summary.textContent = "settings";
        details.append(summary);
        for (const line of settingsLines(systemEngine)) {
          const setting = document.createElement("div");
          setting.textContent = line;
          details.append(setting);
        }
        block.append(details);
      }
    }
    body.append(block);
  }
}
