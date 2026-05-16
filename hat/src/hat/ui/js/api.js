// Tiny HTTP wrapper. All endpoints are same-origin.

async function handle(r, label) {
  if (!r.ok) {
    const text = await r.text().catch(() => "");
    throw new Error(`${label} → ${r.status} ${text}`);
  }
  return r;
}

export async function jget(path) {
  const r = await fetch(path);
  await handle(r, path);
  return r.json();
}

export async function jpost(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body == null ? "{}" : JSON.stringify(body),
  });
  await handle(r, path);
  return r.json();
}

export async function jpatch(path, body) {
  const r = await fetch(path, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  await handle(r, path);
  return r.json();
}

export async function jdelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  await handle(r, path);
  return r.json().catch(() => ({}));
}
