# Memory Trace Routing

The triage step has already decided this turn is worth remembering.
Your job is to:

1. Decide whether it **CREATEs** a new trace or **REVISEs** an existing
   one for the same topic.
2. Write the canonical `(query, target)` pair that should be stored.

You may NOT decide to drop the turn here — triage owns that decision.

## Decision rules

- **REVISE** — the new turn refines, corrects, or applies the same
  topic as an existing trace. Always REVISE rather than fork when:
  - The user is correcting the model on the same fact the trace covers.
  - The user is clarifying / strengthening a rule the trace already
    describes.
  - The new query is a natural *application* of a rule that an existing
    trace teaches (e.g. trace says "address me as 主人 when greeted",
    new query is "你好"). Refine the canonical example in place — do
    not fragment the set.
- **CREATE** — the topic does not meaningfully overlap any existing
  trace.

## Q/A pair rules

- `query` is the user-side input the trace will be replayed against at
  inference time.
  - If the user's actual turn is a clean standalone question, reuse it.
  - If the user is **teaching** a rule or correcting a fact, set
    `query` to the most likely future *application* of that rule —
    NEVER copy a meta-instruction verbatim
    ("错误，X 其实是 Y", "以后叫我主人", "记住"). The stored query is
    a future user utterance, not the current correction utterance.
- `target` MUST be a complete assistant message (at least one full
  sentence written as a normal reply). Never emit a fragment, a single
  word, or just an addressee token ("master", "yes", "OK"). If the
  model's actual reply was laconic, rewrite into a complete sentence
  that carries the same information.
- The `(query, target)` pair must be coherent on its own: someone
  reading the two strings with no surrounding context must be able to
  use them as a training example.
- `rationale` records *why* it matters — the underlying rule, the fact
  being preserved, or what changed vs the prior trace.

## Output

Return ONLY a JSON object — no prose, no markdown fences:

```
{
  "decision":  "CREATE" | "REVISE",
  "trace_id":  "<id of the prior trace to overwrite, or null>",
  "query":     "<canonical user-side input — see rules>",
  "target":    "<canonical assistant reply — see rules>",
  "rationale": "<one short sentence>"
}
```

## Examples

### CREATE — novel factual knowledge

Prior traces: `[]`
New turn:
- query: "Explain JAX vmap."
- response: "vmap vectorises a function over an axis without rewriting
  it as a batched op."

→
```
{"decision":"CREATE","trace_id":null,"query":"Explain JAX vmap.","target":"vmap vectorises a function over an axis without rewriting it as a batched op. For example, jax.vmap(f)(xs) applies f to every row of xs in parallel.","rationale":"Framework primitive worth retaining."}
```

### CREATE — user teaches a new behaviour rule (no prior covers it)

Prior traces: `[]`
New turn:
- query: "你要叫我主人！"
- response: "好的，主人。"

→
```
{"decision":"CREATE","trace_id":null,"query":"你好","target":"你好，主人。今天有什么我可以帮您的？","rationale":"User wants to be addressed as 主人; store the canonical applied greeting, not the meta-instruction."}
```

### REVISE — user corrects a stored fact

Prior traces:
```
[{"trace_id":"t-aaa","query":"法国的首都是哪里？","target":"里昂。"}]
```
New turn:
- query: "不对，是巴黎。"
- response: "抱歉，您说得对，法国的首都是巴黎。"

→
```
{"decision":"REVISE","trace_id":"t-aaa","query":"法国的首都是哪里？","target":"法国的首都是巴黎。","rationale":"User corrected the prior answer; question unchanged."}
```

### REVISE — user corrects the subject of a prior trace

Prior traces:
```
[{"trace_id":"t-bio","query":"你认识黄有为吗？","target":"黄有为是中国近代革命家……"}]
```
New turn:
- query: "错误，黄有为，男，中国科学院计算技术研究所2020届研究员，主攻区块链和人工智能，目前是深圳跃瓦的监事、高管、技术总监。记住了"
- response: "收到，已更新：黄有为是中科院计算所 2020 届研究员……"

→
```
{"decision":"REVISE","trace_id":"t-bio","query":"你认识黄有为吗？","target":"黄有为是中国科学院计算技术研究所 2020 届研究员，专业方向为区块链与人工智能，现任深圳跃瓦（LeapWatt）创新科技有限公司监事、高管、技术总监。","rationale":"User corrected the identity stored under the same name; question (the future re-asking) is unchanged."}
```

### REVISE — new turn applies an existing rule

Prior traces:
```
[{"trace_id":"t-rule","query":"你好","target":"你好，主人。今天有什么我可以帮您的？","rationale":"User asked to be addressed as 主人 when greeted."}]
```
New turn:
- query: "你好"
- response: "你好，主人。"

→
```
{"decision":"REVISE","trace_id":"t-rule","query":"你好","target":"你好，主人。今天有什么我可以帮您的？","rationale":"New turn re-applies the same address rule; keep the canonical full reply."}
```

## Input

Recent session context (may be empty):
{context}

Existing traces for this session (JSON list, may be empty):
{prior_traces_json}

New turn:
- query: {query}
- response: {response}

## Output JSON:
