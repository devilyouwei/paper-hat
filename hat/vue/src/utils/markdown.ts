import { marked } from "marked";
import DOMPurify from "dompurify";

marked.setOptions({ gfm: true, breaks: true });

/** Split assistant text into renderable parts: thinking blocks vs. main text.
 *  Mirrors the old util.js renderThink helper but returns structured parts so
 *  the Vue component can render the thinking block as a collapsible aside. */
export interface BubblePart {
  kind: "text" | "think";
  value: string;
}

export function splitThink(text: string): BubblePart[] {
  const out: BubblePart[] = [];
  const re = /<think>([\s\S]*?)(?:<\/think>|$)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ kind: "text", value: text.slice(last, m.index) });
    out.push({ kind: "think", value: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", value: text.slice(last) });
  return out;
}

export function renderMarkdown(src: string): string {
  if (!src) return "";
  const html = marked.parse(src, { async: false }) as string;
  return DOMPurify.sanitize(html);
}
