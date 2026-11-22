import {
  ApiError,
  getHealth,
  getReady,
  getSystem,
  getTelemetry,
  listModels,
  type ActiveRequest,
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
  liveSpeeds,
  prefixHitRate,
  refreshRecentRows,
  requestTotals,
  settingsLines,
  sparkArea,
  sparkPath,
  systemEngineFor,
  type ActiveSnapshot,
  type PanelData,
} from "./panel";
import { groupSpeed, speedSummary } from "./speed";
import { appendPoint, HISTORY_KEY, parseHistory, serialize, values, type SparkPoint } from "./sparkstore";
import { copyText, downloadText, flashCopied } from "./copy";

export type MonitorData = PanelData;

const HISTORY_CAP = 24;

export class MonitorView {
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
    active: null,
    checkedAt: 0,
  };
  /** Client-side sparkline history: medians from each sweep, oldest
   * first. The server keeps no history series, so the trend is built
   * here, one point per poll, exactly like Strata's history ring. */
  private decodeHistory: (number | null)[] = [];
  private ttftHistory: (number | null)[] = [];
  private prefillHistory: (number | null)[] = [];
  private decodePoints: SparkPoint[] = [];
  private ttftPoints: SparkPoint[] = [];
  private prefillPoints: SparkPoint[] = [];
  /** The last active-board snapshot and the live speeds derived from it:
   * per-request decode rates between consecutive polls, plus their sum. */
  private activeSnapshot: ActiveSnapshot = {};
  private activeSpeeds: Record<string, number> = {};
  private activeTotalPerSecond = 0;
  private readonly scroll = document.createElement("div");
  private readonly requestDrawer = document.createElement("dialog");
  private requestKey: string | null = null;
  /** GPU telemetry from /v1/telemetry; null when unprobed or down. */
  private telemetry: TelemetryInfo | null = null;
  private telemetryError: string | null = null;

  constructor(section: HTMLElement) {
    this.scroll.className = "panel-scroll";
    this.requestDrawer.className = "side-drawer monitor-request-drawer";
    this.requestDrawer.id = "monitor-request-detail";
    this.requestDrawer.setAttribute("aria-labelledby", "monitor-request-title");
    this.requestDrawer.addEventListener("close", () => {
      const key = this.requestKey;
      this.requestKey = null;
      if (section.hidden) return;
      const trigger = Array.from(this.scroll.querySelectorAll<HTMLButtonElement>(".req-inspect"))
        .find((button) => button.dataset.request === key);
      (trigger ?? this.scroll.querySelector<HTMLButtonElement>(".monitor-refresh"))?.focus();
    });
    section.replaceChildren(this.scroll, this.requestDrawer);
    try {
      const stored = parseHistory(window.localStorage.getItem(HISTORY_KEY), Date.now());
      this.decodePoints = stored.decode;
      this.ttftPoints = stored.ttft;
      this.prefillPoints = stored.prefill ?? [];
      this.decodeHistory = values(stored.decode);
      this.ttftHistory = values(stored.ttft);
      this.prefillHistory = values(this.prefillPoints);
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
      : { recent: null, recentError: null, active: null };
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
      active: panel.active,
    });
    if (this.data.active !== null) {
      const live = liveSpeeds(this.activeSnapshot, this.data.active, this.data.checkedAt);
      this.activeSnapshot = live.next;
      this.activeSpeeds = live.speeds;
      this.activeTotalPerSecond = live.totalPerSecond;
    } else {
      this.activeSnapshot = {};
      this.activeSpeeds = {};
      this.activeTotalPerSecond = 0;
    }
    if (this.data.recent !== null) {
      const speed = speedSummary(this.data.recent);
      this.decodeHistory = [...this.decodeHistory, speed.decodePerSecond].slice(-HISTORY_CAP);
      this.ttftHistory = [...this.ttftHistory, speed.firstTokenMs].slice(-HISTORY_CAP);
      this.decodePoints = appendPoint(this.decodePoints, this.data.checkedAt, speed.decodePerSecond);
      this.ttftPoints = appendPoint(this.ttftPoints, this.data.checkedAt, speed.firstTokenMs);
      this.prefillPoints = appendPoint(this.prefillPoints, this.data.checkedAt, speed.prefillPerSecond);
      this.decodeHistory = values(this.decodePoints);
      this.ttftHistory = values(this.ttftPoints);
      this.prefillHistory = values(this.prefillPoints);
      try {
        window.localStorage.setItem(
          HISTORY_KEY,
          serialize({ decode: this.decodePoints, ttft: this.ttftPoints, prefill: this.prefillPoints })
        );
      } catch {
        // Persistence is best-effort; the in-memory trend still works.
      }
    }
    this.render();
    return this.data;
  }

  closeRequestDetail(): void {
    if (this.requestDrawer.open) this.requestDrawer.close();
  }

  private card(title: string, sub?: string): { card: HTMLElement; body: HTMLElement } {
    const card = document.createElement("section");
    card.className = "monitor-section";
    const h3 = document.createElement("h2");
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

  /** The hero band: one line of engine-wide figures, big numerals with
   * small labels, hairline-separated — in flight, waiting, live total
   * throughput, the finished-request median, and the hardware load. */
  private renderOverview(): HTMLElement {
    const band = document.createElement("div");
    band.className = "monitor-hero";
    const engines = this.data.system?.engines;
    const queue = engines?.some((engine) => engine.queued === undefined)
      ? null
      : engines?.reduce((sum, engine) => sum + (engine.queued ?? 0), 0) ?? null;
    const decode = this.data.recent === null ? null : speedSummary(this.data.recent).decodePerSecond;
    const gpus = this.telemetry?.gpus ?? [];
    const memory = gpus.filter((gpu) => gpu.mem_used_bytes !== null && (gpu.mem_total_bytes ?? 0) > 0);
    const used = memory.reduce((sum, gpu) => sum + (gpu.mem_used_bytes ?? 0), 0);
    const total = memory.reduce((sum, gpu) => sum + (gpu.mem_total_bytes ?? 0), 0);
    const reporting = gpus.filter((gpu) => gpu.util_percent !== null);
    const load = reporting.length === 0
      ? null
      : reporting.reduce((sum, gpu) => sum + (gpu.util_percent ?? 0), 0) / reporting.length;
    let serverStatus = "Checking…";
    let serverState = "offline";
    if (this.data.checkedAt !== 0) {
      serverStatus = "No answer";
      if (this.data.healthOk) {
        serverStatus = this.data.readyOk ? "Ready" : "Not ready";
        serverState = this.data.readyOk ? "ready" : "busy";
      }
    }
    const status = document.createElement("span");
    status.className = "hero-status";
    const dot = document.createElement("span");
    dot.className = "stat-dot s-" + serverState;
    dot.setAttribute("aria-hidden", "true");
    status.append(dot, document.createTextNode(serverStatus + (this.data.notReadyCode ? " · " + this.data.notReadyCode : "")));
    band.append(status);
    const inFlight = this.data.active;
    const figs: Array<[string, string]> = [
      [inFlight === null ? "—" : String(inFlight.length), "in flight"],
      [queue === null ? "—" : String(queue), "waiting"],
      [inFlight === null || inFlight.length === 0 ? "—" : this.activeTotalPerSecond.toFixed(1), "tok/s now · all requests"],
      [decode === null ? "—" : decode.toFixed(1), "tok/s median · finished"],
      [load === null ? "—" : Math.round(load) + "%", "GPU load · " + (total === 0 ? "no data" : formatBytes(used) + " / " + formatBytes(total))],
    ];
    const host = this.data.system?.host;
    const ramUsed = typeof host?.mem_total_bytes === "number" && typeof host?.mem_available_bytes === "number"
      ? host.mem_total_bytes - host.mem_available_bytes
      : null;
    if (ramUsed !== null && typeof host?.mem_total_bytes === "number") {
      figs.push([formatBytes(ramUsed), "RAM · of " + formatBytes(host.mem_total_bytes)]);
    }
    for (const [value, label] of figs) {
      const fig = document.createElement("div");
      fig.className = "hero-fig";
      const num = document.createElement("span");
      num.className = "hero-num";
      num.textContent = value;
      const text = document.createElement("span");
      text.className = "hero-label";
      text.textContent = label;
      fig.append(num, text);
      band.append(fig);
    }
    return band;
  }

  private render(): void {
    const column = document.createElement("div");
    column.className = "panel-column monitor-column";
    const head = document.createElement("div");
    head.className = "panel-head monitor-head";
    const h2 = document.createElement("h2");
    h2.textContent = "Server overview";
    const live = document.createElement("span");
    live.className = "live-badge";
    live.textContent = this.data.checkedAt ? `Checked ${formatClock(this.data.checkedAt)}` : "Not checked yet";
    const refresh = document.createElement("button");
    refresh.className = "btn monitor-refresh";
    refresh.type = "button";
    refresh.textContent = "Refresh";
    refresh.addEventListener("click", () => void this.refresh());
    head.append(h2, live, refresh);
    column.append(head);

    if (!this.data.healthOk && this.data.checkedAt !== 0) {
      const alert = document.createElement("div");
      alert.className = "monitor-alert";
      alert.setAttribute("role", "status");
      alert.textContent = "Server not answering. Check that enginesrv is running, then select Refresh.";
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

    column.append(this.renderOverview(), this.renderActiveCard(), this.renderRequestsCard());
    const metrics = document.createElement("div");
    metrics.className = "monitor-grid";
    metrics.append(this.renderSpeedCard(), this.renderGpuCard());
    column.append(metrics, this.renderRatesCard());
    const system = document.createElement("div");
    system.className = "monitor-grid";
    system.append(this.renderSystemCard(), this.renderEnginesCard());
    column.append(system);

    const focused = document.activeElement;
    const requestKey = focused instanceof HTMLButtonElement ? focused.dataset.request : undefined;
    const keepRefresh = this.scroll.contains(focused) && focused?.classList.contains("monitor-refresh");
    this.scroll.replaceChildren(column);
    if (requestKey !== undefined) {
      const request = Array.from(column.querySelectorAll<HTMLButtonElement>(".req-inspect"))
        .find((button) => button.dataset.request === requestKey);
      (request ?? refresh).focus({ preventScroll: true });
    } else if (keepRefresh) {
      refresh.focus({ preventScroll: true });
    }
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
    for (const figure of [
      [speed.decodePerSecond === null ? "—" : speed.decodePerSecond.toFixed(1), "tok/s decode (median)"],
      [speed.prefillPerSecond === null ? "—" : Math.round(speed.prefillPerSecond).toLocaleString(), "tok/s prefill (median)"],
      [speed.firstTokenMs === null ? "—" : Math.round(speed.firstTokenMs) + " ms", "First token (median)"],
    ] as Array<[string, string]>) {
      const cell = document.createElement("div");
      const value = document.createElement("div");
      value.className = "speed-value";
      value.textContent = figure[0];
      const label = document.createElement("div");
      label.className = "stat-sub";
      label.textContent = figure[1];
      cell.append(value, label);
      big.append(cell);
    }
    body.append(big);
    this.spark(body, this.decodeHistory, "Decode speed trend");
    this.spark(body, this.prefillHistory, "Prefill speed trend");
    const sparkLabel = document.createElement("div");
    sparkLabel.className = "stat-sub";
    sparkLabel.textContent = "Trends · one point per refresh · TTFT median now " +
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

  /** The serving engine's context limit for an active row, matched by
   * engine id first, then by model — the system route carries both. */
  private contextLimitOf(row: ActiveRequest): number | null {
    const engines = this.data.system?.engines ?? [];
    const engine = engines.find((e) => e.id === row.engine) ?? engines.find((e) => e.model === row.model);
    return engine?.context_limit ?? null;
  }

  /** One in-flight row: the phase chip, the numbers the board carries,
   * and the live decode speed derived from the previous poll. Elapsed
   * comes from the row's own start time, so it ticks with the poll. */
  private activeRow(row: ActiveRequest): HTMLTableRowElement {
    const tr = document.createElement("tr");
    const phase = document.createElement("td");
    const chip = document.createElement("span");
    chip.className = "state-chip s-" + row.phase;
    chip.textContent = row.phase;
    phase.append(chip);
    const model = document.createElement("td");
    const name = document.createElement("div");
    name.textContent = row.model;
    const engineSub = document.createElement("div");
    engineSub.className = "stat-sub";
    engineSub.textContent = row.engine ?? "—";
    model.append(name, engineSub);
    const session = document.createElement("td");
    session.textContent = row.session ?? "—";
    const elapsedMs = Math.max(0, Date.now() - row.started_at_ms);
    const limit = this.contextLimitOf(row);
    const used = row.prompt_tokens + row.completion_tokens;
    const context = document.createElement("td");
    if (limit !== null && limit > 0) {
      context.className = "num";
      context.title = used.toLocaleString() + " of " + limit.toLocaleString() + " tokens · prompt " + row.prompt_tokens.toLocaleString();
      context.textContent = Math.round((used / limit) * 100) + "%";
    } else {
      context.textContent = "—";
    }
    tr.append(phase, model, session, context);
    const speed = this.activeSpeeds[row.request_id];
    const cells: Array<[string, boolean]> = [
      [String(row.completion_tokens), true],
      [speed === undefined ? "—" : speed.toFixed(1), true],
      [elapsedMs >= 60_000
        ? Math.floor(elapsedMs / 60_000) + " m " + String(Math.floor((elapsedMs % 60_000) / 1000)).padStart(2, "0") + " s"
        : (elapsedMs / 1000).toFixed(1) + " s", true],
    ];
    for (const [text, numeric] of cells) {
      const td = document.createElement("td");
      td.textContent = text;
      if (numeric) td.className = "num";
      tr.append(td);
    }
    return tr;
  }

  private renderActiveCard(): HTMLElement {
    const { card, body } = this.card("Active now", "In-flight requests from queue to finish · tok/s is measured between refreshes · prompt tokens in the context tooltip.");
    if (this.data.recentError !== null && this.data.active === null) {
      this.hint(
        body,
        this.data.recentError === "unauthorized"
          ? "Set the API key in About to see requests."
          : "The recent-requests route failed: " + this.data.recentError
      );
      return card;
    }
    const active = this.data.active ?? [];
    if (active.length === 0) {
      this.hint(body, "No requests running. The next one appears here from queue to finish.");
      return card;
    }
    const wrap = document.createElement("div");
    wrap.className = "req-table-wrap";
    const table = document.createElement("table");
    table.className = "req-table";
    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    for (const [label, numeric] of [
      ["Phase", false],
      ["Model", false],
      ["Session", false],
      ["Context", true],
      ["Out", true],
      ["tok/s", true],
      ["Elapsed", true],
    ] as Array<[string, boolean]>) {
      const th = document.createElement("th");
      th.textContent = label;
      if (numeric) th.className = "num";
      headRow.append(th);
    }
    thead.append(headRow);
    const tbody = document.createElement("tbody");
    for (const row of active) tbody.append(this.activeRow(row));
    table.append(thead, tbody);
    wrap.append(table);
    body.append(wrap);
    return card;
  }

  private renderRequestsCard(): HTMLElement {
    const { card, body } = this.card("Recent requests", "Newest first · Up to 12 shown · Inspect any row for timings and token counts.");
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
      ["Queue", true],
      ["First token", true],
      ["Decode", true],
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

  private openRequestDetail(row: RecentRequest, key: string): void {
    this.requestKey = key;
    const head = document.createElement("header");
    head.className = "side-drawer-head";
    const title = document.createElement("h2");
    title.id = "monitor-request-title";
    title.textContent = "Request detail";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "btn";
    close.textContent = "Close";
    close.autofocus = true;
    close.addEventListener("click", () => this.closeRequestDetail());
    head.append(title, close);
    const host = document.createElement("div");
    host.className = "monitor-detail-body";
    const dl = document.createElement("dl");
    dl.className = "fact-grid";
    const copyJson = document.createElement("button");
    copyJson.type = "button";
    copyJson.className = "btn";
    copyJson.textContent = "Copy JSON";
    copyJson.addEventListener("click", () => {
      void copyText(JSON.stringify(row, null, 2), this.requestDrawer).then((ok) => flashCopied(copyJson, ok));
    });
    host.append(dl, copyJson);
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
    this.requestDrawer.replaceChildren(head, host);
    this.requestDrawer.showModal();
  }

  private requestRow(row: RecentRequest): HTMLTableRowElement {
    const tr = document.createElement("tr");
    tr.className = "req-row-clickable";
    const key = JSON.stringify(row);
    tr.addEventListener("click", () => this.openRequestDetail(row, key));
    const time = document.createElement("td");
    time.className = "num";
    time.textContent = row.started_at_ms === undefined ? "—" : formatClock(row.started_at_ms);
    tr.append(time);
    const model = document.createElement("td");
    model.className = "req-model";
    const inspect = document.createElement("button");
    inspect.type = "button";
    inspect.className = "req-inspect";
    inspect.textContent = row.model;
    inspect.dataset.request = key;
    inspect.setAttribute("aria-label", `Show request detail for ${row.model}`);
    inspect.setAttribute("aria-haspopup", "dialog");
    model.append(inspect);
    if (row.engine) {
      const engine = document.createElement("span");
      engine.className = "req-status";
      engine.textContent = row.engine;
      model.append(engine);
    }
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
      [row.queue_wait_ms == null ? "—" : `${Math.round(row.queue_wait_ms)} ms`, "num"],
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
