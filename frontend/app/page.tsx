"use client";

import * as React from "react";
import { toast } from "sonner";
import { PanelLeft, PanelLeftClose } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QueryFormWrapper } from "@/components/query-form-wrapper";
import { SchemaPanel } from "@/components/schema-panel";
import { Database, listDatabases, getSchema } from "@/lib/api";

export default function Home() {
  const [databases, setDatabases] = React.useState<Database[]>([]);
  const [selectedDb, setSelectedDb] = React.useState<string>("");
  const [schema, setSchema] = React.useState<string>("");
  const [sidebarOpen, setSidebarOpen] = React.useState(false);

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
    <main className="flex-1 bg-background py-6">
      <div className="flex gap-6 px-6">
        {sidebarOpen && (
          <aside className="w-full transition-all lg:w-1/3 lg:max-w-[400px] lg:min-w-[300px]">
            <div className="sticky top-6">
              <SchemaPanel
                schema={schema}
                displayName={selectedDbRecord?.display_name}
                backendType={selectedDbRecord?.backend_type}
              />
            </div>
          </aside>
        )}
        <section className="flex-1">
          <div className="mb-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setSidebarOpen((s) => !s)}
              aria-label={sidebarOpen ? "Close schema panel" : "Open schema panel"}
            >
              {sidebarOpen ? (
                <PanelLeftClose className="h-4 w-4" />
              ) : (
                <PanelLeft className="h-4 w-4" />
              )}
            </Button>
          </div>
          <QueryFormWrapper
            databases={databases}
            selectedDb={selectedDb}
            onSelectedDbChange={setSelectedDb}
          />
        </section>
      </div>
    </main>
  );
}
