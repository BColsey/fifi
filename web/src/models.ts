/**
 * The Models view: search the Hugging Face hub, download into the
 * hub cache, and serve or stop engines at runtime — the GUI side of
 * /v1/hf/* and /v1/engines. Every fetch goes through api.ts.
 */

import {
  type EngineSummary, type HubCache, type HubDownloadJob, type HubSearchRow,
  addEngine, cancelHubDownload, listEngines, listHubCache, listHubDownloads,
  removeEngine, searchHubModels, startHubDownload,
} from "./api.js";
import { formatBytes } from "./format.js";

/** One card with a title, built once, filled by render passes. */
function card(title: string): { root: HTMLDivElement; body: HTMLDivElement } {
  const root = document.createElement("div");
  root.className = "card";
  const h3 = document.createElement("h3");
  h3.textContent = title;
  const body = document.createElement("div");
  root.append(h3, body);
  return { root, body };
}

function line(text: string): HTMLParagraphElement {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

export function statusOf(job: HubDownloadJob): string {
  switch (job.state) {
    case "downloading":
      return `${job.files_done ?? 0}/${job.files_total ?? "?"} files, ` +
        `${formatBytes(job.bytes ?? 0)} of ${formatBytes(job.bytes_total ?? 0)}` +
        (job.current_file === undefined ? "" : ` — ${job.current_file}`);
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

/** The one-line resource ladder for a model the user is looking
 * at: the weights one engine loads, then the tier note. */
export function estimateLine(est: {
  vram_bytes: number;
  ram_bytes: number;
  disk_bytes: number;
  mode: string;
  planned?: boolean;
}): string {
  const weightsOnly = est.planned === false ? " (weights only; cache and working set not added)" : "";
  return `${formatBytes(est.vram_bytes)} resident on GPU, or streamed from RAM/disk — picked automatically${weightsOnly}`;
}

export class ModelsView {
  private readonly searchBody: HTMLDivElement;
  private readonly downloadsBody: HTMLDivElement;
  private readonly cacheBody: HTMLDivElement;
  private readonly enginesBody: HTMLDivElement;
  private readonly queryInput: HTMLInputElement;
  private readonly ggufBox: HTMLInputElement;
  private readonly exl3Box: HTMLInputElement;
  private results: HubSearchRow[] = [];
  private searchError: string | null = null;
  private downloads: HubDownloadJob[] = [];
  private cache: HubCache | null = null;
  private cacheError: string | null = null;
  private engines: EngineSummary[] = [];
  private enginesError: string | null = null;
  private note: string | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  constructor(section: HTMLElement) {
    section.replaceChildren();

    const search = card("Hub search");
    const form = document.createElement("form");
    this.queryInput = document.createElement("input");
    this.queryInput.type = "search";
    this.queryInput.placeholder = "org/repo or free text";
    this.queryInput.setAttribute("aria-label", "Search the hub");
    this.ggufBox = document.createElement("input");
    this.ggufBox.type = "checkbox";
    const ggufLabel = document.createElement("label");
    ggufLabel.append(this.ggufBox, document.createTextNode(" GGUF"));
    this.exl3Box = document.createElement("input");
    this.exl3Box.type = "checkbox";
    const exl3Label = document.createElement("label");
    exl3Label.append(this.exl3Box, document.createTextNode(" EXL3"));
    const button = document.createElement("button");
    button.type = "submit";
    button.textContent = "Search";
    form.append(this.queryInput, ggufLabel, exl3Label, button);
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.search();
    });
    this.searchBody = document.createElement("div");
    search.root.append(form, this.searchBody);

    this.downloadsBody = card("Downloads").body;
    this.cacheBody = card("Local hub cache").root;
    this.enginesBody = card("Engines").root;
    this.noteBody = document.createElement("div");
    this.noteBody.className = "card";
    this.noteBody.setAttribute("role", "status");

    section.append(
      search.root,
      this.downloadsBody.parentElement as HTMLElement,
      this.noteBody,
      this.cacheBody,
      this.enginesBody,
    );
    this.render();
  }

  private readonly noteBody: HTMLDivElement;

  private say(text: string | null): void {
    if (text === this.note) return;
    this.note = text;
    this.noteBody.replaceChildren();
    if (text !== null) this.noteBody.append(line(text));
  }

  /** A second refresh while one is in flight returns instead of
   * stacking fetches or letting a slow pass overwrite a fresh one
   * (MonitorView's N3 rule, applied to this poll too). */
  private refreshing = false;

  async refresh(): Promise<void> {
    if (this.refreshing) return;
    this.refreshing = true;
    try {
      const signal = AbortSignal.timeout(4_000);
      const [engines, cache, downloads] = await Promise.all([
        listEngines(signal), listHubCache(signal), listHubDownloads(signal),
      ]);
      this.engines = engines.engines;
      this.enginesError = null;
      this.cache = cache;
      this.cacheError = null;
      this.downloads = downloads.downloads;
    } catch (err) {
      this.enginesError = this.cacheError = err instanceof Error ? err.message : String(err);
    } finally {
      this.refreshing = false;
    }
    this.render();
    this.schedulePoll();
  }

  /** While a job is active the view polls itself every 2 s; the
   * timer dies when the view is idle so the page costs nothing. */
  private schedulePoll(): void {
    const active = this.downloads.some(
      (job) => job.state === "downloading",
    );
    if (active && this.pollTimer === null) {
      this.pollTimer = setInterval(() => void this.refresh(), 2_000);
    } else if (!active && this.pollTimer !== null) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  private async search(): Promise<void> {
    const q = this.queryInput.value.trim();
    if (q === "") return;
    try {
      const found = await searchHubModels(q, {
        gguf: this.ggufBox.checked, exl3: this.exl3Box.checked,
      });
      this.results = found.results;
      this.searchError = null;
    } catch (err) {
      this.results = [];
      this.searchError = err instanceof Error ? err.message : String(err);
    }
    this.renderSearch();
  }

  private async download(repo: string): Promise<void> {
    const running = this.downloads.some(
      (job) =>
        job.repo === repo &&
        (job.state === "downloading"),
    );
    if (running) {
      this.say(`${repo} is already downloading.`);
      return;
    }
    try {
      await startHubDownload(repo);
      this.say(`Download of ${repo} started.`);
      await this.refresh();
    } catch (err) {
      this.say(err instanceof Error ? err.message : String(err));
    }
  }

  private async serve(modelDir: string): Promise<void> {
    try {
      const added = await addEngine({ model_dir: modelDir });
      this.say(`Engine ${added.id} is loading ${added.model}.`);
      await this.refresh();
    } catch (err) {
      this.say(err instanceof Error ? err.message : String(err));
    }
  }

  private async stop(id: string): Promise<void> {
    const held = this.engines.find((engine) => engine.id === id)?.memory_held_until_restart === true;
    if (!window.confirm(stopQuestion(id, held))) return;
    try {
      const result = await removeEngine(id);
      this.say(stoppedLine(id, result));
      await this.refresh();
    } catch (err) {
      this.say(err instanceof Error ? err.message : String(err));
    }
  }

  private renderSearch(): void {
    this.searchBody.replaceChildren();
    if (this.searchError !== null) this.searchBody.append(line(this.searchError));
    if (this.results.length === 0) return;
    const table = document.createElement("table");
    const head = document.createElement("tr");
    for (const text of ["Repo", "Downloads", "Likes", ""]) {
      const th = document.createElement("th");
      th.textContent = text;
      head.append(th);
    }
    table.append(head);
    for (const row of this.results) {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = row.gated ? `${row.id} (gated)` : row.id;
      const downloads = document.createElement("td");
      downloads.textContent = String(row.downloads);
      const likes = document.createElement("td");
      likes.textContent = String(row.likes);
      const action = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Download";
      button.disabled = row.gated;
      button.title = row.gated
        ? "Gated repo: download with hf or huggingface-cli and a token."
        : "Download into the hub cache.";
      button.addEventListener("click", () => void this.download(row.id));
      action.append(button);
      tr.append(name, downloads, likes, action);
      table.append(tr);
    }
    this.searchBody.append(table);
  }

  private render(): void {
    this.renderSearch();
    this.say(this.note);

    const downloadsCard = this.downloadsBody.parentElement as HTMLElement;
    downloadsCard.hidden = this.downloads.length === 0;
    this.downloadsBody.replaceChildren();
    if (this.downloads.length > 0) {
      const table = document.createElement("table");
      for (const job of [...this.downloads].sort((a, b) => a.id - b.id)) {
        const tr = document.createElement("tr");
        const repo = document.createElement("td");
        repo.textContent = job.repo;
        const status = document.createElement("td");
        status.textContent = statusOf(job);
        const needs = document.createElement("td");
        if (job.estimate !== undefined) {
          needs.textContent = estimateLine(job.estimate);
          needs.title = job.estimate.mode;
        }
        const action = document.createElement("td");
        if (job.state === "downloading") {
          const cancel = document.createElement("button");
          cancel.type = "button";
          cancel.textContent = "Cancel";
          cancel.addEventListener("click", () => void cancelHubDownload(job.id)
            .then(() => this.refresh())
            .catch((err: unknown) =>
              this.say(err instanceof Error ? err.message : String(err))));
          action.append(cancel);
        }
        tr.append(repo, status, needs, action);
        table.append(tr);
      }
      this.downloadsBody.append(table);
    }

    this.cacheBody.replaceChildren();
    if (this.cacheError !== null) {
      this.cacheBody.append(line(this.cacheError));
    } else if (this.cache !== null && Array.isArray(this.cache.models)) {
      const h3 = document.createElement("h3");
      h3.textContent = "Local hub cache";
      const dir = line(`Cache: ${this.cache.hub_dir}`);
      this.cacheBody.append(h3, dir);
      if (this.cache.models.length === 0) {
        this.cacheBody.append(line("No models cached yet."));
      }
      for (const model of this.cache.models) {
        const heading = document.createElement("p");
        heading.textContent = `${model.repo} — ${formatBytes(model.bytes)}`;
        this.cacheBody.append(heading);
        for (const snapshot of model.snapshots) {
          const row = document.createElement("p");
          if (snapshot.estimate !== undefined) {
            const est = document.createElement("small");
            est.textContent = estimateLine(snapshot.estimate);
            est.title = `${snapshot.estimate.basis} — ${snapshot.estimate.mode}`;
            row.append(est, document.createElement("br"));
          }
          row.append(
            document.createTextNode(
              `${snapshot.commit.slice(0, 8)} — ${snapshotStatus(snapshot)} `,
            ),
          );
          if (snapshot.servable) {
            const serve = document.createElement("button");
            serve.type = "button";
            serve.textContent = "Serve";
            serve.title = `Start an engine on exactly this snapshot (${snapshot.commit}).`;
            serve.addEventListener("click", () => void this.serve(snapshot.path));
            row.append(serve);
          }
          this.cacheBody.append(row);
        }
      }
    } else if (this.cache !== null) {
      // A scan failure answers 200 with {"error"} and no models
      // array; without this arm the render above would not run and
      // the failure would be invisible.
      this.cacheBody.append(
        line((this.cache as { error?: string }).error ?? "cache scan failed"),
      );
    }

    this.enginesBody.replaceChildren();
    const enginesH3 = document.createElement("h3");
    enginesH3.textContent = "Engines";
    this.enginesBody.append(enginesH3);
    if (this.enginesError !== null) {
      this.enginesBody.append(line(this.enginesError));
    } else if (this.engines.length === 0) {
      this.enginesBody.append(line("No engines running."));
    } else {
      const table = document.createElement("table");
      for (const engine of this.engines) {
        const tr = document.createElement("tr");
        const id = document.createElement("td");
        id.textContent = engine.runtime_added ? `${engine.id} *` : engine.id;
        const state = document.createElement("td");
        state.textContent = engine.state === "loading" && engine.load_stage !== undefined
          ? `loading (${engine.load_stage})`
          : engine.error !== undefined
            ? `${engine.state}: ${engine.error.message}`
            : engine.state;
        const model = document.createElement("td");
        model.textContent = engine.model;
        const action = document.createElement("td");
        if (engine.runtime_added) {
          const stop = document.createElement("button");
          stop.type = "button";
          stop.textContent = "Stop";
          stop.addEventListener("click", () => void this.stop(engine.id));
          action.append(stop);
        }
        tr.append(id, state, model, action);
        table.append(tr);
      }
      this.enginesBody.append(
        table,
        line("* added at runtime; only those can be stopped here."),
      );
    }
  }
}
