export class ApiError extends Error {
  status: number;
  body: string;
  constructor(path: string, status: number, body: string) {
    super(`${path} → ${status} ${body.slice(0, 200)}`);
    this.status = status;
    this.body = body;
  }
}

async function handle<T>(path: string, res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new ApiError(path, res.status, text);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return {} as T;
}

export async function jget<T = unknown>(path: string): Promise<T> {
  return handle<T>(path, await fetch(path, { credentials: "same-origin" }));
}

export async function jpost<T = unknown>(path: string, body: unknown = {}): Promise<T> {
  return handle<T>(
    path,
    await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      credentials: "same-origin",
    }),
  );
}

export async function jpatch<T = unknown>(path: string, body: unknown = {}): Promise<T> {
  return handle<T>(
    path,
    await fetch(path, {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      credentials: "same-origin",
    }),
  );
}

export async function jdelete<T = unknown>(path: string): Promise<T> {
  return handle<T>(path, await fetch(path, { method: "DELETE", credentials: "same-origin" }));
}
