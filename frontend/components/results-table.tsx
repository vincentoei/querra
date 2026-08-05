"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

interface ResultsTableProps {
  rows: unknown[][] | null;
  columns?: string[] | null;
}

export function ResultsTable({ rows, columns }: ResultsTableProps) {
  if (!rows || rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No rows returned.</p>;
  }

  const colNames = columns && columns.length > 0 ? columns : null;
  const colCount = Math.max(...rows.map((row) => row.length));

  // Single-cell result: render as a natural-language sentence.
  const isSingleCell = rows.length === 1 && colCount === 1;
  if (isSingleCell) {
    const value = rows[0][0];
    const label = colNames?.[0];
    const display = value === null ? "NULL" : String(value ?? "");
    return (
      <p className="text-lg">
        {label && label.toLowerCase() !== "name" && label.toLowerCase() !== "column 1"
          ? <>
              <span className="text-muted-foreground">{label}:</span>{" "}
              <span className="font-semibold">{display}</span>
            </>
          : <>
              <span className="text-muted-foreground">Result:</span>{" "}
              <span className="font-semibold">{display}</span>
            </>}
      </p>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-sm text-muted-foreground">
        {rows.length === 1 ? "1 row" : `${rows.length} rows`} found.
      </p>
      <div className="overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              {Array.from({ length: colCount }).map((_, i) => (
                <TableHead key={i}>
                  {colNames?.[i] ?? `Column ${i + 1}`}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, idx) => (
              <TableRow key={idx}>
                {Array.from({ length: colCount }).map((_, i) => (
                  <TableCell key={i}>
                    {row[i] === null ? "NULL" : String(row[i] ?? "")}
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