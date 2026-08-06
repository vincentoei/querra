"use client";

import * as React from "react";
import { Copy, Check } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface SchemaPanelProps {
  schema: string;
  displayName?: string;
  backendType?: string;
}

function highlightSqlSchema(schema: string): React.ReactNode[] {
  const lines = schema.split("\n");
  const keywords = new Set(["CREATE", "TABLE", "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "NOT", "NULL", "UNIQUE", "INDEX", "ON", "AS", "SELECT", "FROM", "WHERE", "JOIN", "AND", "OR", "ORDER", "BY", "GROUP", "HAVING", "LIMIT", "OFFSET", "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "ADD", "COLUMN", "CONSTRAINT", "DEFAULT", "AUTOINCREMENT", "CASCADE", "RESTRICT", "SET", "VALUES", "INTO", "EXISTS", "IF", "TEMPORARY", "VIEW", "TRIGGER", "BEGIN", "END", "COMMIT", "ROLLBACK", "TRANSACTION"]);
  const types = new Set(["INTEGER", "TEXT", "REAL", "NUMERIC", "BLOB", "VARCHAR", "CHAR", "BOOLEAN", "DATE", "DATETIME", "TIME", "TIMESTAMP", "FLOAT", "DOUBLE", "DECIMAL", "INT", "BIGINT", "SMALLINT", "TINYINT", "MEDIUMINT", "SERIAL", "UUID", "JSON", "ARRAY"]);

  return lines.map((line, lineIdx) => {
    const tokens: React.ReactNode[] = [];
    const parts = line.split(/(\s+|[(),;.`'"])/);
    let key = 0;
    parts.forEach((part, idx) => {
      const upper = part.toUpperCase();
      if (keywords.has(upper)) {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`} className="sql-keyword font-semibold">{part}</span>);
      } else if (types.has(upper)) {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`} className="sql-type">{part}</span>);
      } else if (part.startsWith("--") || part.startsWith("/*")) {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`} className="sql-comment">{part}</span>);
      } else if (part.startsWith("'") || part.startsWith('"')) {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`} className="sql-string">{part}</span>);
      } else if (/^[(),;.`]+$/.test(part)) {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`} className="sql-symbol">{part}</span>);
      } else {
        tokens.push(<span key={`${lineIdx}-${idx}-${key++}`}>{part}</span>);
      }
    });
    return (
      <div key={lineIdx} className="leading-relaxed">
        {tokens.length > 0 ? tokens : <span>&nbsp;</span>}
      </div>
    );
  });
}

function getBackendBadgeColor(backendType?: string): string {
  switch (backendType?.toLowerCase()) {
    case "sqlite":
      return "bg-sky-400/15 text-sky-400 border-sky-400/20 hover:bg-sky-400/25";
    case "postgres":
    case "postgresql":
      return "bg-pastel-red/15 text-pastel-red border-pastel-red/20 hover:bg-pastel-red/25";
    case "mysql":
      return "bg-amber-400/15 text-amber-400 border-amber-400/20 hover:bg-amber-400/25";
    default:
      return "bg-muted text-muted-foreground";
  }
}

export function SchemaPanel({ schema, displayName, backendType }: SchemaPanelProps) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    if (!schema) return;
    try {
      await navigator.clipboard.writeText(schema);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Ignore copy errors
    }
  };

  const highlighted = React.useMemo(() => highlightSqlSchema(schema), [schema]);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div className="flex items-center gap-2.5">
          <h2 className="text-sm font-semibold text-foreground">Database Schema</h2>
          {backendType && (
            <Badge variant="outline" className={`text-xs font-medium ${getBackendBadgeColor(backendType)}`}>
              {backendType.toUpperCase()}
            </Badge>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Copy schema"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>

      {/* Info */}
      {displayName && (
        <div className="px-5 py-2 text-xs text-muted-foreground border-b border-border/50">
          {displayName}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-5 font-mono text-[13px]">
          {schema ? (
            <div className="space-y-0">
              {highlighted}
            </div>
          ) : (
            <div className="flex h-full items-center justify-center text-muted-foreground">
              No schema loaded.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
