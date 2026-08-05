"use client";

import * as React from "react";
import { toast } from "sonner";
import {
  Database,
  GenerateRequest,
  GenerateResponse,
  ExecuteResponse,
  generateSql,
  executeSql,
} from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ResultsTable } from "@/components/results-table";
import { AlertTriangle, Loader2 } from "lucide-react";

export interface QueryFormProps {
  databases: Database[];
  selectedDb: string;
  onSelectedDbChange: (dbId: string) => void;
}

export function QueryForm({ databases, selectedDb, onSelectedDbChange }: QueryFormProps) {
  const [question, setQuestion] = React.useState<string>("");
  const [sql, setSql] = React.useState<string>("");
  const [result, setResult] = React.useState<GenerateResponse | ExecuteResponse | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [executing, setExecuting] = React.useState(false);

  const handleGenerate = async () => {
    if (!selectedDb) {
      toast.error("Select a database first");
      return;
    }
    if (!question.trim()) {
      toast.error("Enter a question");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const payload: GenerateRequest = {
        db_id: selectedDb,
        question,
        execute: false,
      };
      const res = await generateSql(payload);
      setSql(res.sql);
      setResult(res);
      if (res.execution_error) {
        toast.error(res.execution_error);
      } else if (res.valid) {
        toast.success("SQL generated — review it before running");
      } else if (!res.valid) {
        toast.error("Generated query failed validation");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to generate SQL");
    } finally {
      setLoading(false);
    }
  };

  const handleRunSql = async () => {
    if (!selectedDb) {
      toast.error("Select a database first");
      return;
    }
    if (!sql.trim()) {
      toast.error("No SQL to run");
      return;
    }
    setExecuting(true);
    setResult(null);
    try {
      const res = await executeSql({ db_id: selectedDb, sql });
      setResult(res);
      if (res.execution_error) {
        toast.error(res.execution_error);
      } else {
        toast.success("Query executed");
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to execute SQL");
    } finally {
      setExecuting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Querra</h1>
        <p className="text-muted-foreground">
          Ask questions about your database in plain English.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>New question</CardTitle>
          <CardDescription>
            Pick a database, type your question, and generate a SQL query.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="database">Database</Label>
            <Select value={selectedDb} onValueChange={(v) => onSelectedDbChange(v ?? "")}>
              <SelectTrigger id="database" className="w-full" suppressHydrationWarning>
                <SelectValue placeholder="Select a database" />
              </SelectTrigger>
              <SelectContent>
                {databases.map((db) => (
                  <SelectItem key={db.db_id} value={db.db_id}>
                    {db.display_name || db.db_id}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="question">Question</Label>
            <Input
              id="question"
              placeholder="What are the names of singers older than 30?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleGenerate();
                }
              }}
              suppressHydrationWarning
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <Button
              onClick={handleGenerate}
              disabled={loading || executing}
              suppressHydrationWarning
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Generate SQL
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Generated SQL
            {result && (
              <Badge variant={result.valid ? "default" : "destructive"}>
                {result.valid ? "Valid" : "Invalid"}
              </Badge>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Textarea
            value={sql}
            onChange={(e) => setSql(e.target.value)}
            placeholder="Generated SQL will appear here..."
            rows={6}
            className="font-mono"
            disabled={!question.trim()}
            suppressHydrationWarning
          />
          {result && result.warnings && result.warnings.length > 0 && (
            <div className="rounded-lg border border-yellow-600/20 bg-yellow-50 p-3 dark:bg-yellow-950/20">
              <div className="flex items-start gap-2">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600" />
                <div className="space-y-1 text-sm text-yellow-800 dark:text-yellow-200">
                  {result.warnings.map((warning, idx) => (
                    <p key={idx}>{warning}</p>
                  ))}
                </div>
              </div>
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              onClick={handleRunSql}
              disabled={!sql || loading || executing}
              suppressHydrationWarning
            >
              {executing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Run SQL
            </Button>
          </div>
          {result && (
            <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
              <span>Latency: {result.latency.toFixed(3)}s</span>
              {result.execution_error && (
                <span className="text-destructive">Error: {result.execution_error}</span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {result?.execution_result && (
        <Card>
          <CardHeader>
            <CardTitle>Results</CardTitle>
          </CardHeader>
          <CardContent>
            <ResultsTable rows={result.execution_result} columns={result.execution_columns} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
