/**
 * send() must close the gate synchronously, before its first await, so
 * a second send, a chat switch, or a delete cannot start a second
 * stream. No DOM and no storage.
 */

export class SendGate {
  private busy = false;

  tryBegin(): boolean {
    if (this.busy) return false;
    this.busy = true;
    return true;
  }

  end(): void {
    this.busy = false;
  }
}
