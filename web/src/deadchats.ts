/**
 * A finishing stream must not write a deleted chat back, and a failed
 * delete must unmark the id so the chat keeps saving. No DOM and no storage.
 */

export class DeadChats {
  private readonly ids = new Set<string>();

  mark(id: string): void {
    this.ids.add(id);
  }

  unmark(id: string): void {
    this.ids.delete(id);
  }

  has(id: string): boolean {
    return this.ids.has(id);
  }
}
