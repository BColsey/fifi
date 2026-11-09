/** Copy with a fallback: the async clipboard needs a secure context,
 * and fifi is often opened over LAN http, where only the legacy path
 * works. Returns false when neither path copied. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fall through to the legacy path.
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** Save text as a downloaded file. */
export function downloadText(filename: string, text: string, type = "text/plain"): void {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Flash "Copied" on a button, then restore its label. */
export function flashCopied(button: HTMLButtonElement, ok: boolean): void {
  const original = button.dataset.label ?? button.textContent ?? "Copy";
  button.dataset.label = original;
  button.textContent = ok ? "Copied" : "Select & copy";
  window.setTimeout(() => {
    button.textContent = original;
  }, 1500);
}
