"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { BarChart3 } from "lucide-react";

interface ResultsTableProps {
  rows: unknown[][] | null;
  columns?: string[] | null;
}

export function ResultsTable({ rows, columns }: ResultsTableProps) {
  if (!rows || rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-center">
        <div className="mb-3 rounded-full bg-muted p-3">
          <BarChart3 className="h-5 w-5 text-muted-foreground" />
        </div>
        <p className="text-sm text-muted-foreground">No rows returned.</p>
      </div>
    );
  }

  const colNames = columns && columns.length > 0 ? columns : null;
  const colCount = Math.max(...rows.map((row) => row.length));

  // Single-cell result: render as a prominent stat card
  const isSingleCell = rows.length === 1 && colCount === 1;
  if (isSingleCell) {
    const value = rows[0][0];
    const label = colNames?.[0];
    const display = value === null ? "NULL" : String(value ?? "");
    return (
      <div className="rounded-xl border border-sky-400/15 bg-sky-400/5 p-8 text-center">
        <p className="text-sm font-medium text-muted-foreground mb-2">
          {label && label.toLowerCase() !== "name" && label.toLowerCase() !== "column 1"
            ? label
            : "Result"}
        </p>
        <p className="text-4xl font-bold text-foreground tracking-tight">
          {display}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="inline-flex items-center rounded-md bg-sky-400/10 px-2 py-0.5 text-xs font-medium text-sky-400">
          {rows.length === 1 ? "1 row" : `${rows.length} rows`}
        </span>
        <span className="text-xs text-muted-foreground">
          {colNames ? `${colNames.length} columns` : `${colCount} columns`}
        </span>
      </div>
      <div className="overflow-x-auto rounded-lg border border-border">
        <Table>
          <TableHeader>
            <TableRow className="bg-sky-400/5 hover:bg-sky-400/5">
              {Array.from({ length: colCount }).map((_, i) => (
                <TableHead key={i} className="text-xs font-semibold text-foreground">
                  {colNames?.[i] ?? `Column ${i + 1}`}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, idx) => (
              <TableRow key={idx} className="transition-colors">
                {Array.from({ length: colCount }).map((_, i) => (
                  <TableCell key={i} className="text-sm tabular-nums">
                    {row[i] === null ? (
                      <span className="text-muted-foreground italic">NULL</span>
                    ) : (
                      String(row[i] ?? "")
                    )}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
