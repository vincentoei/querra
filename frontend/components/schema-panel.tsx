"use client";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";

interface SchemaPanelProps {
  schema: string;
  displayName?: string;
  backendType?: string;
}

export function SchemaPanel({ schema, displayName, backendType }: SchemaPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Database schema</CardTitle>
        {displayName && backendType && (
          <CardDescription>
            {displayName} · {backendType}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <pre className="h-[calc(100vh-16rem)] overflow-auto whitespace-pre rounded-lg border bg-muted p-3 font-mono text-sm">
          {schema || "No schema loaded."}
        </pre>
      </CardContent>
    </Card>
  );
}
