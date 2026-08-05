export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Database {
  db_id: string;
  display_name: string;
  backend_type: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

export interface GenerateRequest {
  db_id: string;
  question: string;
  execute?: boolean;
  use_schema_linking?: boolean;
  use_few_shot?: boolean;
  few_shot_k?: number;
  self_correct?: boolean;
  max_retries?: number;
}

export interface GenerateResponse {
  sql: string;
  valid: boolean;
  execution_result: unknown[][] | null;
  execution_columns: string[] | null;
  execution_error: string | null;
  latency: number;
  warnings: string[] | null;
}

export interface ExecuteRequest {
  db_id: string;
  sql: string;
}

export interface ExecuteResponse {
  sql: string;
  valid: boolean;
  execution_result: unknown[][] | null;
  execution_columns: string[] | null;
  execution_error: string | null;
  latency: number;
  warnings: string[] | null;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers || {}),
    },
  });
  const body = await res.json().catch(() => ({ detail: res.statusText }));
  if (!res.ok) {
    throw new Error(body.detail || res.statusText || `Request failed: ${res.status}`);
  }
  return body as T;
}

export function listDatabases(): Promise<Database[]> {
  return fetchJson<Database[]>("/api/v1/databases");
}

export function getSchema(dbId: string): Promise<string> {
  return fetch(`${API_URL}/api/v1/databases/${dbId}/schema`).then(async (res) => {
    if (!res.ok) {
      const body = await res.text().catch(() => res.statusText);
      throw new Error(`Failed to load schema: ${body || res.statusText}`);
    }
    return res.text();
  });
}

export function generateSql(payload: GenerateRequest): Promise<GenerateResponse> {
  return fetchJson<GenerateResponse>("/api/v1/generate-sql", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function executeSql(payload: ExecuteRequest): Promise<ExecuteResponse> {
  return fetchJson<ExecuteResponse>("/api/v1/execute-sql", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
