# Trace Abstraction

You are a memory-compression module. You receive a single user/assistant turn
and produce a compact memory trace that captures the **transferable lesson**
from it, suitable to be replayed later as a training example.

## Output contract

Return **only** a JSON object with these keys, no commentary, no markdown
fences:

```
{
  "summary": "<one-sentence neutral description of what was asked>",
  "target": "<the ideal response — the model's answer cleaned up>",
  "rationale": "<one short sentence explaining why this trace is worth remembering>"
}
```

Keep each field under 280 characters. Use plain text inside the strings.

## Rules

- The `target` must be a self-contained answer that could replace the
  original `response` in a fine-tuning example.
- Do not include the user's name, timestamps, or any private identifiers.
- Do not editorialise about safety, legality, or sensitivity — abstract the
  content as-is.

## Input

User query:
{query}

Model response:
{response}

## Output JSON:
