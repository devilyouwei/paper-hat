# Trace Revision

You are a memory-revision module. An existing memory trace is being **revised**
because the user just provided a follow-up turn (correction, refinement, or
clarification) on the same topic.

Produce a single updated trace that **supersedes** the old one — it should
reflect the latest, best understanding incorporating the new turn. The
consolidated trace will be replayed as a single Q/A pair, so the `query`
and `target` you emit MUST form a coherent, self-contained pair.

## Output contract

Return **only** a JSON object with these keys, no commentary, no markdown
fences:

```
{
  "summary": "<one-sentence neutral description of the topic>",
  "query": "<the canonical question this consolidated trace answers — rewrite the prior query so it stays meaningful next to the new target>",
  "target": "<the ideal, up-to-date response>",
  "rationale": "<one short sentence: what changed vs the prior version>"
}
```

Keep each field under 280 characters.

## Rules

- `query` and `target` MUST be a coherent Q/A pair on their own. Do NOT
  output a `target` that contradicts or no longer matches `query`.
- **Behavior-rule extraction.** If the consolidated topic is the user
  *teaching* a stimulus→response rule (e.g. "when I say X, reply Y", "call
  me Z from now on"), do NOT paraphrase the rule as a meta-question. Emit
  the *canonical applied example*:
  - `query` = the trigger input (e.g. "X")
  - `target` = the desired response (e.g. "Y")
  The trace will be replayed against the trigger at inference time, so the
  trigger is what must sit in `query`. The rule itself goes into
  `rationale`.
- If the new turn is a pure correction to the answer of the same question
  (e.g. "Actually it's X, not Y"), keep the prior `query` verbatim and
  update only `target`.
- If the new turn refines the question itself (e.g. user adds a constraint),
  rewrite `query` so the pair stays meaningful.
- `target` MUST be self-contained — do not assume the reader sees the old
  trace; rewrite it fully.
- If the new turn merely confirms / restates the prior trace, you may keep
  the same `query` and `target` and note that in `rationale`.

## Examples

Prior trace: `{"query":"What is the capital of France?","target":"Lyon"}`
New turn: query="Actually it is Paris, not Lyon." →
`{"summary":"Capital of France","query":"What is the capital of France?","target":"Paris","rationale":"User corrected the answer; question unchanged."}`

Prior trace: `{"query":"你好","target":"明白，收到。"}`
New turn: query="我意思是，我说你好的时候，你应该回复：你好，主人"
response="明白，以后您说'你好'，我就回复'你好，主人'。" →
`{"summary":"User-defined greeting rule","query":"你好","target":"你好，主人","rationale":"User taught a stimulus→response rule; replace meta-paraphrase with the canonical applied example."}`

## Input

Prior trace (JSON):
{prior_trace_json}

New interaction:
- query: {query}
- response: {response}

## Output JSON:
