"use client";

import * as React from "react";
import { toast } from "sonner";
import { QueryFormWrapper } from "@/components/query-form-wrapper";
import { SchemaPanel } from "@/components/schema-panel";
import { Database, listDatabases, getSchema } from "@/lib/api";

export default function Home() {
  const [databases, setDatabases] = React.useState<Database[]>([]);
  const [selectedDb, setSelectedDb] = React.useState<string>("");
  const [schema, setSchema] = React.useState<string>("");

  React.useEffect(() => {
    listDatabases()
      .then((dbs) => {
        setDatabases(dbs);
        if (dbs.length > 0 && !selectedDb) {
          setSelectedDb(dbs[0].db_id);
        }
      })
      .catch((e) => toast.error(e instanceof Error ? e.message : "Failed to load databases"));
  }, [selectedDb]);

  React.useEffect(() => {
    let ignore = false;
    if (selectedDb) {
      getSchema(selectedDb)
        .then((s) => {
          if (!ignore) setSchema(s);
        })
        .catch((e) => {
          if (!ignore) {
            setSchema("");
            toast.error(e instanceof Error ? e.message : "Failed to load schema");
          }
        });
    }
    return () => {
      ignore = true;
    };
  }, [selectedDb]);

  const selectedDbRecord = databases.find((db) => db.db_id === selectedDb);

  return (
    <div className="flex min-h-[100dvh] flex-col">
      <main className="flex-1">
        {/* Desktop: Split pane */}
        <div className="hidden lg:grid lg:grid-cols-[1fr_420px] lg:gap-0">
          {/* Left: Query flow */}
          <div className="overflow-y-auto p-6 lg:p-8">
            <QueryFormWrapper
              databases={databases}
              selectedDb={selectedDb}
              onSelectedDbChange={setSelectedDb}
            />
          </div>

          {/* Right: Schema panel (fixed) */}
          <aside className="sticky top-0 h-[100dvh] border-l border-border bg-card">
            <SchemaPanel
              schema={schema}
              displayName={selectedDbRecord?.display_name}
              backendType={selectedDbRecord?.backend_type}
            />
          </aside>
        </div>

        {/* Mobile: Stacked */}
        <div className="flex flex-col gap-6 p-4 lg:hidden">
          <QueryFormWrapper
            databases={databases}
            selectedDb={selectedDb}
            onSelectedDbChange={setSelectedDb}
          />
          <SchemaPanel
            schema={schema}
            displayName={selectedDbRecord?.display_name}
            backendType={selectedDbRecord?.backend_type}
          />
        </div>
      </main>
    </div>
  );
}
