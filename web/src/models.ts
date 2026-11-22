import {
  type EngineSummary, type HubCache, type HubDownloadJob, type HubSearchRow, type HubTree, type ResourceEstimate,
  addEngine, cancelHubDownload, getHubTree, getSystem, getTelemetry, listEngines, listHubCache, listHubDownloads,
  removeEngine, searchHubModels, startHubDownload,
} from "./api.js";
import { formatBytes, formatDate, formatTokens } from "./format.js";
import { assessFit, ggufVariants, type Fit, type FitBudget } from "./model-fit.js";

const GiB = 2 ** 30;
const tierLabels: Record<Fit, string> = {
  gpu: "VRAM only",
  ram: "VRAM → RAM streaming",
  ssd: "VRAM → RAM → SSD streaming candidate",
  "disk-unknown": "SSD streaming candidate · free space unknown",
  "too-large": "Not enough SSD space",
  unknown: "Not enough data to estimate fit",
};

function available(bytes?: number): string {
  return bytes === undefined ? "unknown" : formatBytes(bytes);
}

async function getFitBudget(signal: AbortSignal): Promise<FitBudget> {
  const [system, telemetry] = await Promise.all([
    getSystem(signal).catch(() => undefined), getTelemetry(signal).catch(() => undefined),
  ]);
  const ram = system?.host?.mem_available_bytes;
  let vram: number | undefined;
  for (const gpu of telemetry?.gpus ?? []) {
    if (gpu.mem_total_bytes == null || gpu.mem_used_bytes == null) continue;
    const free = gpu.mem_total_bytes - gpu.mem_used_bytes;
    if (free >= 0 && (vram === undefined || free > vram)) vram = free;
  }
  return { vram, ram: typeof ram === "number" ? ram : undefined };
}

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K, text = "", className = "",
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.textContent = text;
  node.className = className;
  return node;
}

function button(text: string, action: (button: HTMLButtonElement) => void, label = text): HTMLButtonElement {
  const node = el("button", text, "btn");
  node.type = "button";
  node.setAttribute("aria-label", label);
  node.addEventListener("click", () => action(node));
  return node;
}

function card(title: string, description?: string): { root: HTMLElement; body: HTMLDivElement } {
  const root = el("section", "", "card models-card");
  const body = el("div");
  root.append(el("h2", title));
  if (description) root.append(el("p", description, "models-muted"));
  root.append(body);
  return { root, body };
}

function capacityStrip(budget: FitBudget | null): HTMLDListElement {
  const strip = el("dl", "", "models-capacity");
  const entries = [
    ["Free VRAM", budget ? available(budget.vram) : "…"],
    ["Available RAM", budget ? available(budget.ram) : "…"],
    ["SSD", "Not reported"],
  ];
  for (const [label, amount] of entries) {
    const cell = el("div", "", "models-capacity-cell");
    cell.append(el("dt", label), el("dd", amount));
    strip.append(cell);
  }
  return strip;
}

function badge(text: string, tone = ""): HTMLSpanElement {
  return el("span", text, `models-badge ${tone}`);
}

function row(title: string | HTMLElement, detail: string, status?: HTMLElement): HTMLElement {
  const node = el("article", "", "models-row");
  const heading = el("h3");
  heading.append(title);
  const head = el("div", "", "models-row-head");
  head.append(heading);
  if (status) head.append(status);
  node.append(head, el("p", detail, "models-muted"));
  return node;
}

function empty(text: string): HTMLElement {
  return el("p", text, "models-empty");
}

function estimateDetails(est: ResourceEstimate): HTMLElement {
  const details = el("details", "", "models-details");
  const summary = el("summary", est.planned ? "Resource estimate" : "Resource estimate · weights only");
  const facts = el("dl", "", "models-resources");
  for (const [label, bytes] of [
    ["GPU VRAM", est.vram_bytes], ["Host RAM", est.ram_bytes], ["Disk", est.disk_bytes],
  ] as const) {
    const item = el("div");
    item.append(el("dt", label), el("dd", formatBytes(bytes)));
    facts.append(item);
  }
  details.append(summary, facts, el("p", est.basis, "models-muted"), el("p", est.mode, "models-muted"));
  return details;
}

export function statusOf(job: HubDownloadJob): string {
  switch (job.state) {
    case "downloading": {
      const parts: string[] = [];
      if (job.files_total !== undefined) parts.push(`${job.files_done ?? 0}/${job.files_total} files`);
      if (job.bytes_total !== undefined) {
        parts.push(`${formatBytes(job.bytes ?? 0)} of ${formatBytes(job.bytes_total)}`);
      } else if (job.bytes !== undefined) {
        parts.push(`${formatBytes(job.bytes)} received`);
      }
      const progress = parts.join(", ") || "Preparing download";
      return job.current_file ? `${progress} — ${job.current_file}` : progress;
    }
    case "failed":
      return job.detail ?? "failed";
    default:
      return job.state;
  }
}

/** The question before a stop, honest about memory the family keeps. */
export function stopQuestion(id: string, memoryHeld: boolean): string {
  return memoryHeld
    ? `Stop engine ${id}? Running jobs on it are cancelled. Its card memory is not ` +
      "returned until the server process restarts, and this server cannot load another " +
      "engine of this family before then."
    : `Stop engine ${id}? Running jobs on it are cancelled.`;
}

/** The line shown after a stop: the server's own note when memory stays held. */
export function stoppedLine(
  id: string,
  result: { memory_released: boolean; note?: string },
): string {
  return result.memory_released
    ? `Engine ${id} stopped.`
    : `Engine ${id} ${result.note ?? "stopped; card memory returns when the server process restarts"}`;
}

/** What the page says about one cached snapshot, and whether it can be served. */
export function snapshotStatus(snapshot: {
  complete: boolean;
  servable: boolean;
  families: string[];
}): string {
  if (!snapshot.complete) return "incomplete: a download is unfinished or a file is missing";
  if (!snapshot.servable) return "no family of this build recognises it";
  return `serves as ${snapshot.families.join(", ")}`;
}

export class ModelsView {
  private readonly discovery = card("Find a model", "Search Hugging Face by model name or owner/repository.");
  private readonly library = card("Local cache");
  private readonly serving = card("Engines");
  private readonly queryInput = el("input", "", "text-field");
  private readonly format = el("select", "", "text-field");
  private readonly feedback = el("p", "", "models-feedback");
  private readonly searchStatus = el("p", "", "models-muted models-search-status");
  private readonly results = el("div", "", "models-results");
  private readonly capacity = el("div");
  private readonly downloadDrawer = el("dialog", "", "side-drawer models-download-drawer");
  private readonly downloadList = el("div", "", "models-download-list");
  private readonly downloadFeedback = el("p", "", "models-feedback");
  private readonly downloadLauncher = button("Downloads", () => this.openDownloads());
  private readonly closeDownloadButton = button("Close", () => this.closeDownloads(), "Close downloads");
  private readonly clearFinishedButton = button("Clear finished", () => this.clearFinishedDownloads());
  // The server owns downloads and files. Clearing affects this page's list only.
  private readonly clearedDownloads = new Set<number>();
  private budgetRead: Promise<FitBudget> | null = null;
  private engines: EngineSummary[] = [];
  private downloads: HubDownloadJob[] = [];
  private searchController: AbortController | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private refreshing = false;
  private refreshAgain = false;

  constructor(section: HTMLElement) {
    const hardware = card("Server capacity");
    this.capacity.replaceChildren(capacityStrip(null));
    hardware.body.append(this.capacity);

    this.feedback.setAttribute("role", "status");
    this.searchStatus.setAttribute("role", "status");
    this.buildSearch();
    this.buildDownloads();
    this.discovery.root.prepend(this.downloadLauncher);
    for (const section of [this.library, this.serving]) {
      section.body.append(empty("Loading…"));
    }
    const local = el("aside", "", "models-local");
    local.setAttribute("aria-label", "Local model status");
    local.append(hardware.root, this.serving.root, this.library.root,
      button("Refresh local status", () => {
        void this.readBudget();
        void this.refresh(true);
      }),
    );

    const content = el("div", "", "models-content");
    content.append(this.discovery.root, local);
    section.replaceChildren(content, this.downloadDrawer);
  }

  private buildDownloads(): void {
    const title = el("h2", "Downloads");
    title.id = "models-download-title";
    this.downloadDrawer.id = "models-downloads";
    this.downloadDrawer.setAttribute("aria-labelledby", title.id);
    this.downloadLauncher.classList.add("models-download-launcher");
    this.downloadLauncher.setAttribute("aria-haspopup", "dialog");
    this.downloadLauncher.setAttribute("aria-controls", this.downloadDrawer.id);
    this.downloadDrawer.addEventListener("close", () => this.downloadLauncher.focus({ preventScroll: true }));
    this.closeDownloadButton.autofocus = true;
    const head = el("header", "", "side-drawer-head");
    head.append(title, this.closeDownloadButton);
    const tools = el("div", "", "models-download-tools");
    const hint = el("p", "Clearing the list keeps active downloads and cached files.", "models-muted");
    tools.append(this.clearFinishedButton, hint, this.downloadFeedback);
    this.downloadFeedback.setAttribute("role", "status");
    this.clearFinishedButton.disabled = true;
    this.downloadList.append(empty("Loading…"));
    this.downloadDrawer.append(head, tools, this.downloadList);
  }

  private openDownloads(): void {
    if (this.downloadDrawer.open) return;
    this.downloadFeedback.textContent = "";
    this.downloadDrawer.showModal();
    void this.refresh();
  }

  closeDownloads(): void {
    if (this.downloadDrawer.open) this.downloadDrawer.close();
  }

  private clearFinishedDownloads(): void {
    for (const job of this.downloads) {
      if (job.state !== "downloading") this.clearedDownloads.add(job.id);
    }
    this.closeDownloadButton.focus();
    this.replaceRows(this.downloadList, this.downloadRows());
    this.downloadFeedback.textContent = "Finished downloads cleared from this view.";
  }

  private downloadRows(): HTMLElement[] {
    const activeCount = this.downloads.filter((job) => job.state === "downloading").length;
    const label = activeCount ? `Downloads · ${activeCount} active` : "Downloads";
    this.downloadLauncher.textContent = label;
    this.downloadLauncher.setAttribute("aria-label", label);
    const visible = this.downloads.filter((job) => job.state === "downloading" || !this.clearedDownloads.has(job.id));
    this.clearFinishedButton.disabled = visible.length === activeCount;
    visible.sort((a, b) => Number(b.state === "downloading") - Number(a.state === "downloading") || b.id - a.id);
    return visible.length ? visible.map((job) => this.downloadRow(job)) : [empty("No downloads to show.")];
  }

  private buildSearch(): void {
    const form = el("form", "", "models-search");
    form.setAttribute("role", "search");
    this.queryInput.type = "search";
    this.queryInput.placeholder = "e.g. Qwen 3.6 or owner/repository";
    this.queryInput.setAttribute("aria-label", "Search Hugging Face");
    this.queryInput.required = true;
    this.format.setAttribute("aria-label", "Model format");
    for (const [label, value] of [["All formats", ""], ["GGUF", "gguf"], ["EXL3", "exl3"]]) {
      this.format.add(new Option(label, value));
    }
    const search = el("button", "Search", "btn btn-primary");
    search.type = "submit";
    form.append(this.queryInput, this.format, search);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.search();
    });
    this.discovery.body.append(form, this.feedback, this.searchStatus, this.results);
  }

  private readBudget(): Promise<FitBudget> {
    const request = getFitBudget(AbortSignal.timeout(4_000));
    this.budgetRead = request;
    void request.then((budget) => {
      if (this.budgetRead === request) this.capacity.replaceChildren(capacityStrip(budget));
    });
    return request;
  }

  async refresh(afterAction = false): Promise<void> {
    if (this.budgetRead === null) void this.readBudget();
    if (this.refreshing) {
      this.refreshAgain ||= afterAction;
      return;
    }
    this.refreshing = true;
    try {
      // Read the cache after the queue so a newly finished job is visible in both.
      await Promise.all([
        this.refreshPanel(this.serving.body, listEngines(AbortSignal.timeout(4_000)), ({ engines }) => {
          this.engines = engines;
          return engines.length ? engines.map((engine) => this.engineRow(engine)) : [empty("No engines running.")];
        }),
        this.refreshPanel(this.downloadList, listHubDownloads(AbortSignal.timeout(4_000)), ({ downloads }) => {
          this.downloads = downloads;
          const finished = new Set(downloads.filter((job) => job.state !== "downloading").map((job) => job.id));
          for (const id of this.clearedDownloads) {
            if (!finished.has(id)) this.clearedDownloads.delete(id);
          }
          return this.downloadRows();
        }).then(() => this.refreshPanel(
          this.library.body, listHubCache(AbortSignal.timeout(4_000)), (cache) => this.cacheRows(cache),
        )),
      ]);
    } finally {
      this.refreshing = false;
      this.schedulePoll();
      if (this.refreshAgain) {
        this.refreshAgain = false;
        void this.refresh();
      }
    }
  }

  private async refreshPanel<T>(
    target: HTMLElement, request: Promise<T>, render: (value: T) => HTMLElement[],
  ): Promise<void> {
    try {
      const data = await request;
      const key = JSON.stringify(data);
      if (target.dataset.state === key) return;
      const rows = render(data);
      target.dataset.state = key;
      this.replaceRows(target, rows);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (target.dataset.state === message) return;
      target.dataset.state = message;
      target.replaceChildren(empty(message));
    }
  }

  private replaceRows(target: HTMLElement, rows: HTMLElement[]): void {
    const focus = target.contains(document.activeElement) ? document.activeElement?.getAttribute("aria-label") : null;
    const open = new Set(Array.from(target.querySelectorAll<HTMLDetailsElement>("details[data-key][open]"), (d) => d.dataset.key));
    target.replaceChildren(...rows);
    for (const details of Array.from(target.querySelectorAll<HTMLDetailsElement>("details[data-key]"))) {
      details.open = open.has(details.dataset.key);
    }
    if (focus) Array.from(target.querySelectorAll("button"))
      .find((control) => control.getAttribute("aria-label") === focus)?.focus({ preventScroll: true });
  }

  private schedulePoll(): void {
    const active = this.downloads.some((job) => job.state === "downloading")
      || this.engines.some((engine) => engine.state === "loading");
    if (active && this.pollTimer === null) {
      this.pollTimer = setInterval(() => void this.refresh(), 2_000);
    } else if (!active && this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private async search(): Promise<void> {
    const query = this.queryInput.value.trim();
    if (!query) return;
    this.searchController?.abort();
    const controller = new AbortController();
    this.searchController = controller;
    this.searchStatus.textContent = "Searching Hugging Face…";
    this.results.replaceChildren();
    try {
      const { results } = await searchHubModels(query, {
        gguf: this.format.value === "gguf", exl3: this.format.value === "exl3",
      }, AbortSignal.any([controller.signal, AbortSignal.timeout(8_000)]));
      if (controller.signal.aborted) return;
      this.searchStatus.textContent = results.length
        ? `${results.length} ${results.length === 1 ? "result" : "results"}`
        : "No models match your search.";
      this.results.replaceChildren(...results.map((model, index) => this.resultRow(model, index)));
    } catch (err) {
      if (!controller.signal.aborted) {
        this.searchStatus.textContent = err instanceof Error ? err.message : "Search failed.";
      }
    }
  }

  private resultRow(model: HubSearchRow, index: number): HTMLElement {
    const link = el("a", model.id);
    const repoPath = model.id.split("/").map(encodeURIComponent).join("/");
    link.href = `https://huggingface.co/${repoPath}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const updated = Date.parse(model.last_modified ?? "");
    const stats = [
      model.pipeline_tag,
      `${formatTokens(model.downloads)} downloads`,
      `${formatTokens(model.likes)} likes`,
      Number.isFinite(updated) ? `Updated ${formatDate(updated)}` : "",
    ].filter(Boolean).join(" · ");
    const node = row(link, stats, model.gated ? badge("Access required", "models-warning") : undefined);
    node.classList.add("models-result");
    if (model.gated) {
      node.append(el("p", "Request access on Hugging Face and download with the HF CLI.", "models-muted"));
      return node;
    }
    const content = el("div", "", "models-fit");
    content.id = `models-fit-${index}`;
    content.hidden = true;
    const download = button("Download full revision", (control) => void this.act(control, async () => {
      if (this.downloads.some((job) => job.repo === model.id && job.state === "downloading")) {
        return `${model.id} is already downloading.`;
      }
      const { id } = await startHubDownload(model.id);
      this.clearedDownloads.delete(id);
      return `Download started for ${model.id}.`;
    }, this.feedback, () => {
      if (control.closest<HTMLElement>("#view-models")?.hidden === false) this.openDownloads();
    }), `Download full revision for ${model.id}`);
    const compare = button("Compare sizes & fit", (control) => {
      content.hidden = !content.hidden;
      control.setAttribute("aria-expanded", String(!content.hidden));
      if (content.hidden || content.childElementCount) return;
      void this.compareModel(content, model.id, (tree) => {
        download.textContent = `Download full revision · ${formatBytes(tree.total_bytes)}`;
      });
    });
    compare.setAttribute("aria-controls", content.id);
    compare.setAttribute("aria-expanded", "false");
    const actions = el("div", "", "models-actions");
    actions.append(compare, download);
    node.append(actions, content);
    return node;
  }

  private async compareModel(
    content: HTMLElement, repo: string, onLoaded: (tree: HubTree) => void,
  ): Promise<void> {
    const status = el("p", "Checking model files and server memory…", "models-muted");
    status.setAttribute("role", "status");
    content.append(status);
    try {
      const signal = AbortSignal.timeout(22_000);
      const [tree, budget] = await Promise.all([getHubTree(repo, signal), this.readBudget()]);
      onLoaded(tree);
      content.replaceChildren(this.fitPanel(repo, tree, budget));
    } catch (err) {
      status.textContent = err instanceof Error ? err.message : "Could not load model files.";
      content.append(button("Retry comparison", () => {
        content.replaceChildren();
        void this.compareModel(content, repo, onLoaded);
      }));
    }
  }

  private fitPanel(repo: string, tree: HubTree, budget: FitBudget): HTMLElement {
    const panel = el("div", "", "models-fit-content");
    const disk = el("input", "", "text-field");
    disk.type = "number";
    disk.min = "0";
    disk.step = "any";
    disk.placeholder = "e.g. 64";
    const diskLabel = el("label", "Free SSD space (GiB)");
    diskLabel.append(disk);
    const targets = el("div", "", "models-fit-targets");
    targets.append(
      el("span", `${available(budget.vram)} free VRAM (best single GPU)`),
      el("span", `${available(budget.ram)} available RAM`),
      diskLabel,
    );
    const variants = ggufVariants(tree.files);
    const hasGguf = tree.files.some((file) => /\.gguf$/i.test(file.path));
    if (!hasGguf && tree.estimate.vram_bytes > 0) {
      variants.push({ name: "Checkpoint weights", files: [], bytes: tree.estimate.vram_bytes });
    }
    const estimates = el("div");
    const downloadNote = el("p", "", "models-muted");
    const render = (): void => {
      const ssd = disk.validity.valid && Number.isFinite(disk.valueAsNumber) ? disk.valueAsNumber * GiB : undefined;
      const fits = variants.map((variant) => ({ ...variant, fit: assessFit(variant.bytes, { ...budget, ssd }) }));
      const largestFirst = fits.slice().reverse();
      const pick = largestFirst.find((variant) => variant.fit === "gpu")
        ?? largestFirst.find((variant) => variant.fit === "ram");
      const rows = fits.map((variant) => {
        const item = el("div", "", "models-fit-option");
        const heading = el("div", "", "models-fit-row");
        const name = el("strong", variant.name);
        const size = el("span", formatBytes(variant.bytes), "models-fit-size");
        heading.append(name, size, badge(tierLabels[variant.fit], variant === pick ? "models-ready" : ""));
        item.append(heading);
        if (variant === pick) item.append(badge("Recommended starting point", "models-ready"));
        item.append(el("p", `Target ${formatBytes(Math.ceil(variant.bytes / 0.85))} free VRAM or available RAM.`, "models-muted"));
        if (variant.files.length) item.append(this.quantFiles(repo, tree.commit, variant.files));
        return item;
      });
      estimates.replaceChildren(...(rows.length ? rows : [empty("No complete weight variant found in this revision.")]));
      const total = formatBytes(tree.total_bytes);
      downloadNote.textContent = ssd !== undefined && ssd < tree.total_bytes
        ? `Full revision: ${total}. That exceeds the free space you entered.`
        : `Full revision: ${total}. The server downloads every file in the revision.`;
    };
    panel.append(
      targets,
      el("p", "The recommendation prefers the largest variant in the fastest memory tier. These weights-only estimates leave 15% memory headroom. Fifi checks backend support, context and cache at load time. Streaming tiers still need GPU working space.", "models-muted"),
      estimates,
      downloadNote,
    );
    disk.addEventListener("input", render);
    render();
    return panel;
  }

  private quantFiles(repo: string, commit: string, files: string[]): HTMLElement {
    const details = el("details", "", "models-details");
    const revision = /^[a-f\d]{40}$/i.test(commit) ? commit : undefined;
    if (!revision) {
      details.append(el("summary", "Show filenames"));
      for (const file of files) details.append(el("code", file));
      return details;
    }
    details.append(el("summary", files.length === 1 ? "Open exact GGUF file on Hugging Face" : `Open all ${files.length} shard links on Hugging Face`));
    const repoPath = repo.split("/").map(encodeURIComponent).join("/");
    for (const file of files) {
      const link = el("a", file);
      link.href = `https://huggingface.co/${repoPath}/blob/${revision}/${file.split("/").map(encodeURIComponent).join("/")}`;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      details.append(link);
    }
    return details;
  }

  private downloadRow(job: HubDownloadJob): HTMLElement {
    const states = { downloading: "Downloading", done: "Complete", failed: "Failed", cancelled: "Cancelled" };
    const detail = job.state === "downloading" || job.state === "failed" ? statusOf(job) : `Revision: ${job.revision}`;
    const node = row(job.repo, detail, badge(states[job.state], job.state === "done" ? "models-ready" : ""));
    if (job.state === "downloading") {
      const progress = el("progress");
      progress.setAttribute("aria-label", `Download progress for ${job.repo}`);
      if (job.bytes_total && job.bytes !== undefined) {
        progress.max = job.bytes_total;
        progress.value = job.bytes;
      }
      node.append(progress, button("Cancel", (control) => void this.act(control, async () => {
        await cancelHubDownload(job.id);
        return `Cancellation requested for ${job.repo}.`;
      }, this.downloadFeedback), `Cancel download of ${job.repo}`));
    }
    if (job.estimate) {
      const estimate = estimateDetails(job.estimate);
      estimate.dataset.key = String(job.id);
      node.append(estimate);
    }
    return node;
  }

  private cacheRows(cache: HubCache): HTMLElement[] {
    if (!Array.isArray(cache.models)) {
      return [empty((cache as { error?: string }).error ?? "Cache scan failed.")];
    }
    if (!cache.models.length) return [empty("No models cached. Download one from a search result.")];
    const items = cache.models.map((model) => {
      const node = row(model.repo, `${formatBytes(model.bytes)} on disk`);
      for (const snapshot of model.snapshots) {
        const item = el("div", "", "models-snapshot");
        const refs = model.refs.filter((ref) => ref.commit === snapshot.commit).map((ref) => ref.name);
        const revision = el("code", [...refs, snapshot.commit.slice(0, 8)].join(" · "), "mono");
        revision.title = snapshot.commit;
        let state = "Incomplete";
        if (snapshot.complete) state = snapshot.servable ? "Ready to serve" : "Unsupported";
        item.append(revision, badge(state, snapshot.complete && snapshot.servable ? "models-ready" : "models-warning"),
          el("p", snapshotStatus(snapshot), "models-muted"));
        if (snapshot.complete && snapshot.servable) {
          item.append(button("Serve", (control) => void this.act(control, async () => {
            const engine = await addEngine({ model_dir: snapshot.path });
            return `Started engine ${engine.id} for ${engine.model}. See Engines for its status.`;
          }), `Serve ${model.repo} at ${snapshot.commit.slice(0, 8)}`));
        }
        if (snapshot.estimate) {
          const estimate = estimateDetails(snapshot.estimate);
          estimate.dataset.key = snapshot.path;
          item.append(estimate);
        }
        node.append(item);
      }
      return node;
    });
    const location = el("details", "", "models-details models-cache-location");
    location.dataset.key = "cache-path";
    location.append(el("summary", "Cache location"), el("code", cache.hub_dir, "mono"));
    return [...items, location];
  }

  private engineRow(engine: EngineSummary): HTMLElement {
    const label = { loading: "Loading", ready: "Ready", busy: "Busy", error: "Error", stopped: "Stopped", "native-failed": "Failed" }[engine.state];
    const node = row(engine.model, `${engine.id} · ${formatTokens(engine.context_limit)} token context · ${engine.runtime_added ? "Runtime" : "Server config"}`,
      badge(label, engine.state === "ready" ? "models-ready" : ""));
    if (engine.error ?? engine.load_stage) {
      node.append(el("p", engine.error?.message ?? engine.load_stage, "models-muted"));
    }
    if (engine.runtime_added && engine.state !== "stopped") {
      node.append(button("Stop", (control) => {
        if (!window.confirm(stopQuestion(engine.id, engine.memory_held_until_restart))) return;
        void this.act(control, async () => stoppedLine(engine.id, await removeEngine(engine.id)));
      }, `Stop engine ${engine.id}`));
    }
    return node;
  }

  private async act(
    control: HTMLButtonElement, run: () => Promise<string>, feedback = this.feedback,
    onSuccess?: () => void,
  ): Promise<void> {
    control.disabled = true;
    feedback.textContent = "Working…";
    try {
      feedback.textContent = await run();
      onSuccess?.();
      await this.refresh(true);
    } catch (err) {
      feedback.textContent = err instanceof Error ? err.message : String(err);
    } finally {
      control.disabled = false;
    }
  }
}
