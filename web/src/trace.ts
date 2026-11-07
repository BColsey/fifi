/**
 * The per-reply sampler trace: what the model considered at each token
 * it emitted. Fetches packet v2 of /v1/sampler-trace with the reply id
 * the stream carried, and maps each position's raw-text offset into the
 * reply's rendered regions (content / reasoning / tool) by replaying the
 * piece lengths recorded while the reply streamed. The packet never
 * guesses that mapping; neither does this view.
 */

import {
  ApiError,
  describeError,
  getSamplerTrace,
  type SamplerPosition,
  type SamplerTracePacket,
} from "./api";
import type { TraceSeg, UiMessage } from "./store";

export interface FieldAt {
  field: TraceSeg["field"];
  offsetInField: number;
}

/** Maps a packet offset (UTF-8 bytes into the raw generated text) to the
 * rendered region and the offset within it. Null when the segments are
 * missing, the offset is negative, or it lies past the text the GUI kept
 * (for example a stop string the server hides). */
export function mapOffset(segs: readonly TraceSeg[] | undefined, offset: number): FieldAt | null {
  if (segs === undefined || offset < 0) return null;
  let start = 0;
  for (const seg of segs) {
    if (offset < start + seg.bytes) return { field: seg.field, offsetInField: offset - start };
    start += seg.bytes;
  }
  return null;
}

export function fmtProb(p: number): string {
  return (p * 100).toFixed(1) + "%";
}

/** Positions rendered before the "show all" button: enough to scroll a
 * reply, small enough that an 8k-position trace does not wedge the DOM. */
const ROW_LIMIT = 400;

function div(cls: string): HTMLElement {
  const node = document.createElement("div");
  node.className = cls;
  return node;
}

export class TraceView {
  constructor(
    private readonly section: HTMLElement,
    private readonly message: UiMessage
  ) {
    this.render();
  }

  private async render(): Promise<void> {
    const column = this.shell("Token trace");
    const loading = div("panel-hint");
    loading.textContent = "Fetching the trace\u2026";
    column.append(loading);
    let packet: SamplerTracePacket;
    try {
      packet = await getSamplerTrace(this.message.replyId ?? "");
    } catch (err) {
      loading.textContent =
        err instanceof ApiError && err.status === 404
          ? "The server no longer retains this trace \u2014 it keeps the last four traced replies."
          : "Could not load the trace: " + (err instanceof ApiError ? describeError(err) : String(err));
      return;
    }
    column.textContent = "";
    const head = document.createElement("h2");
    head.textContent = "Token trace";
    column.prepend(head);
    column.append(this.factCard(packet), this.positionsCard(packet));
  }

  private shell(title: string): HTMLElement {
    this.section.textContent = "";
    const scroll = div("panel-scroll");
    const column = div("panel-column");
    const head = div("panel-head");
    const h2 = document.createElement("h2");
    h2.textContent = title;
    head.append(h2);
    column.append(head);
    scroll.append(column);
    this.section.append(scroll);
    return column;
  }

  private factCard(packet: SamplerTracePacket): HTMLElement {
    const card = div("card");
    const h3 = document.createElement("h3");
    h3.textContent = "Reply";
    const grid = document.createElement("dl");
    grid.className = "fact-grid";
    card.append(h3, grid);
    const fact = (name: string, value: string) => {
      const dt = document.createElement("dt");
      dt.textContent = name;
      const dd = document.createElement("dd");
      dd.textContent = value;
      grid.append(dt, dd);
    };
    fact("Model", packet.model);
    fact(
      "Sampling",
      "temp " + packet.sampling.temperature + " \u00b7 top_p " + packet.sampling.top_p +
        " \u00b7 top_k " + packet.sampling.top_k
    );
    fact(
      "Positions",
      packet.captured_positions + " captured \u00b7 " + packet.aligned_positions + " aligned" +
        (packet.truncated ? " \u00b7 truncated at the per-reply limit" : "")
    );
    fact("Raw text", packet.raw_byte_len + " bytes");
    if (packet.alignment_error !== null) {
      fact("Alignment", "stopped: " + packet.alignment_error);
    }
    if (packet.trailer !== "") {
      fact("Trailer", JSON.stringify(packet.trailer));
    }
    return card;
  }

  private positionsCard(packet: SamplerTracePacket): HTMLElement {
    const card = div("card");
    const h3 = document.createElement("h3");
    h3.textContent = "Positions \u2014 one row per sampled token; bold is what it picked";
    const rows = div("trace-rows");
    const segs = this.message.traceSegs;
    const render = (from: number, to: number) => {
      for (let i = from; i < to; i++) {
        const pos = packet.positions[i];
        if (pos === undefined) continue;
        rows.append(this.positionRow(pos, i, segs));
      }
    };
    render(0, Math.min(packet.positions.length, ROW_LIMIT));
    card.append(h3, rows);
    if (packet.positions.length > ROW_LIMIT) {
      const more = document.createElement("button");
      more.type = "button";
      more.className = "btn";
      more.textContent = "Show all " + packet.positions.length + " positions";
      more.addEventListener("click", () => {
        render(ROW_LIMIT, packet.positions.length);
        more.remove();
      });
      card.append(more);
    }
    return card;
  }

  private positionRow(pos: SamplerPosition, index: number, segs: readonly TraceSeg[] | undefined): HTMLElement {
    const row = div("trace-row");
    const i = div("trace-i");
    i.textContent = String(index);
    const piece = div("trace-piece");
    piece.textContent = pos.piece === null ? "\u2205" : JSON.stringify(pos.piece);
    piece.title = "token_id " + pos.token_id + " \u00b7 kept " + pos.kept + " \u00b7 rank " + pos.rank;
    const at = pos.offset === null ? null : mapOffset(segs, pos.offset);
    const field = div("trace-field trace-field-" + (at?.field ?? "none"));
    field.textContent =
      at === null
        ? pos.offset === null
          ? "unaligned"
          : "outside text"
        : at.field + " +" + at.offsetInField;
    const p = div("trace-p");
    p.textContent = pos.greedy ? "greedy" : fmtProb(pos.selected_prob);
    const cands = div("trace-cands");
    for (const cand of pos.candidates) {
      const chip = document.createElement("span");
      chip.className = "trace-cand" + (cand.token_id === pos.token_id ? " trace-cand-sel" : "");
      chip.textContent = cand.token_id + ": " + fmtProb(cand.prob);
      cands.append(chip);
    }
    if (!pos.candidates.some((c) => c.token_id === pos.token_id)) {
      const chip = document.createElement("span");
      chip.className = "trace-cand trace-cand-sel";
      chip.textContent = pos.token_id + ": " + fmtProb(pos.selected_prob);
      cands.append(chip);
    }
    row.append(i, piece, field, p, cands);
    return row;
  }
}
