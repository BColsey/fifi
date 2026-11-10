import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdtempSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { stopChild } from "./cleanup.mjs";

async function readyChild(source, args = [], options = {}) {
  const child = spawn(process.execPath, ["-e", source, ...args],
    { stdio: ["ignore", "pipe", "inherit"], ...options });
  await once(child.stdout, "data");
  return child;
}

test("waits until a child finishes writing on shutdown", async () => {
  const dir = mkdtempSync(join(tmpdir(), "fifi-cleanup-test-"));
  const marker = join(dir, "closed");
  const child = await readyChild(`
    process.on('SIGTERM', () => setTimeout(() => {
      require('node:fs').writeFileSync(process.argv[1], 'done');
      process.exit(0);
    }, 100));
    process.stdout.write('ready');
    setInterval(() => {}, 1000);
  `, [marker]);
  try {
    await stopChild(child, { graceMs: 2000, killMs: 2000 });
    assert.equal(existsSync(marker), true);
    assert.equal(child.exitCode, 0);
    await stopChild(child);
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
    rmSync(dir, { recursive: true, force: true });
  }
});

test("force-stops a child that ignores termination", async () => {
  const child = await readyChild(`
    process.on('SIGTERM', () => {});
    process.stdout.write('ready');
    setInterval(() => {}, 1000);
  `);
  try {
    await stopChild(child, { graceMs: 100, killMs: 2000 });
    assert.equal(child.signalCode, "SIGKILL");
  } finally {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
  }
});

test("stops a detached process group and its descendants", { skip: process.platform === "win32" }, async () => {
  const child = await readyChild(`
    process.on('SIGTERM', () => {});
    require('node:child_process').spawn(process.execPath,
      ['-e', "process.on('SIGTERM', () => {}); setInterval(() => {}, 1000)"],
      { stdio: 'ignore' });
    process.stdout.write('ready');
    setInterval(() => {}, 1000);
  `, [], { detached: true });
  try {
    await stopChild(child, { processGroup: true, graceMs: 100, killMs: 2000 });
    assert.throws(() => process.kill(-child.pid, 0), { code: "ESRCH" });
  } finally {
    try { process.kill(-child.pid, "SIGKILL"); } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  }
});
