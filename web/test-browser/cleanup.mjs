/** Stop a child (and, on POSIX, its isolated process group) before cleanup. */
export async function stopChild(child, { graceMs = 3000, killMs = 3000, processGroup = false } = {}) {
  if (!child.pid) throw new Error("child process did not start");
  const group = processGroup && process.platform !== "win32";
  const alive = () => {
    if (!group) return child.exitCode === null && child.signalCode === null;
    try {
      process.kill(-child.pid, 0);
      return true;
    } catch (error) {
      if (error.code === "ESRCH") return false;
      throw error;
    }
  };
  const signal = (name) => {
    try {
      if (group) process.kill(-child.pid, name);
      else child.kill(name);
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  };
  const wait = async (ms) => {
    const until = Date.now() + ms;
    while (alive()) {
      const remaining = until - Date.now();
      if (remaining <= 0) return false;
      await new Promise((resolve) => setTimeout(resolve, Math.min(25, remaining)));
    }
    return true;
  };
  if (!alive()) return;
  signal("SIGTERM");
  if (await wait(graceMs)) return;
  signal("SIGKILL");
  if (!await wait(killMs)) throw new Error("child process did not exit after SIGKILL");
}
