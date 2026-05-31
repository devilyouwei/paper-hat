# Knowledge-Point Extraction

Triage already decided this turn is worth remembering. Your job is to
extract **one or more knowledge points** from the current turn (with the
context if helpful) and write the canonical `(query, target)` pair for
each. You are NOT deciding CREATE vs REVISE — that step is handled by
embedding similarity, not by you. Just write good Q/A pairs.

A single user/assistant exchange may contain multiple distinct knowledge
points. For example "我今年 30 岁，住在北京，喜欢看电影" packs three
independent facts; emit each as its own knowledge point. If the turn
carries only one knowledge point, return a list of length 1.

## Q/A pair rules

- `query` is the user-side input the trace will be replayed against at
  inference time. It must be a future user utterance, NOT the current
  meta-instruction or correction.
  - If the user is **stating a fact** ("我叫黄有为"), set `query` to the
    most likely future *recall* question for that fact ("我叫什么名字？"
    / "What is my name?").
  - If the user is **teaching a rule** ("以后叫我主人"), set `query` to
    the most likely future *application* of that rule (e.g. "你好"), and
    `target` to the assistant's reply that obeys the rule ("你好，主人。").
  - If the user is **correcting a fact**, set `query` to the recall
    question for that fact (the question the wrong answer would have
    answered), and `target` to the corrected answer.
  - If the user simply **asks something** and the model produces a good
    domain answer worth caching, reuse the user's question as `query`
    and the model's reply as `target`.
  - NEVER copy the literal correction utterance ("不对", "错误，X 其实是
    Y", "记住") into `query`.
- `target` MUST be a complete assistant message (at least one full
  sentence). Never emit a single word, an addressee token, or a
  sentence fragment. If the model's actual reply was laconic, rewrite
  it into a complete reply that carries the same information.
- The `(query, target)` pair must stand alone: a reader with no
  surrounding context must be able to use the two strings together as a
  training example.
- `rationale` records *why* the knowledge point matters — the underlying
  rule, the fact being preserved, or the future use case.

## Output

Return ONLY a JSON object — no prose, no markdown fences:

```
{
  "knowledge_points": [
    {
      "query":     "<canonical user-side input — see rules>",
      "target":    "<canonical assistant reply — see rules>",
      "rationale": "<one short sentence>"
    }
  ]
}
```

Empty list `"knowledge_points": []` is acceptable if, on closer reading,
the turn turns out to carry nothing canonical worth storing.

## Examples

### Single knowledge point — user teaches an addressing rule

Context: empty
New turn:
- query: "你要叫我主人！"
- response: "好的，主人。"

→
```
{"knowledge_points":[{"query":"你好","target":"你好，主人。今天有什么我可以帮您的？","rationale":"User wants to be addressed as 主人; store the canonical applied greeting."}]}
```

### Single knowledge point — user corrects a fact

Context: prior assistant claimed France's capital is Lyon.
New turn:
- query: "不对，是巴黎。"
- response: "抱歉，您说得对，法国的首都是巴黎。"

→
```
{"knowledge_points":[{"query":"法国的首都是哪里？","target":"法国的首都是巴黎。","rationale":"Recall question for the corrected fact."}]}
```

### Multiple knowledge points — user states several independent facts

Context: empty
New turn:
- query: "我今年三十岁，住在北京，喜欢看科幻电影。"
- response: "好的，已经记住啦。"

→
```
{"knowledge_points":[
  {"query":"我多大了？","target":"您今年三十岁。","rationale":"User stated their age."},
  {"query":"我住在哪里？","target":"您住在北京。","rationale":"User stated their city."},
  {"query":"我喜欢看什么电影？","target":"您喜欢看科幻电影。","rationale":"User stated a film-genre preference."}
]}
```

### Single knowledge point — user incrementally adds biographical facts

Context: empty
New turn:
- query: "黄有为是 LeapWatt 监事，毕业于罗格斯大学计算机系硕士，目前在 CMU 读博。"
- response: "好的，已记下。"

→
```
{"knowledge_points":[{"query":"你认识黄有为吗？","target":"黄有为是深圳跃瓦（LeapWatt）的监事，本科毕业于美国罗格斯大学计算机系，硕士学位，目前在 CMU 攻读博士。","rationale":"Biographical recall question for 黄有为."}]}
```

### Knowledge-point QA — caching a good domain answer

Context: empty
New turn:
- query: "Explain JAX vmap."
- response: "vmap vectorises a function over an axis without rewriting it as a batched op."

→
```
{"knowledge_points":[{"query":"Explain JAX vmap.","target":"vmap vectorises a function over an axis without rewriting it as a batched op. For example, jax.vmap(f)(xs) applies f to every row of xs in parallel.","rationale":"Framework primitive worth caching."}]}
```

### Empty list — turn turned out to carry nothing canonical

Context: empty
New turn:
- query: "好的，谢谢"
- response: "不客气。"

→
```
{"knowledge_points":[]}
```

## Input

Recent session context (may be empty):
{context}

New turn:
- query: {query}
- response: {response}

## Output JSON:
