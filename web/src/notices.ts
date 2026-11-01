/**
 * The queue is capped so banners cannot stack over the composer.
 * No DOM; main.ts renders the list.
 */

const NOTICE_CAP = 3;

export function noticeList(current: readonly string[], text: string): string[] {
  const next = current.filter((t) => t !== text);
  next.push(text);
  return next.length > NOTICE_CAP ? next.slice(next.length - NOTICE_CAP) : next;
}
