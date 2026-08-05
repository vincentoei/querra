"use client";

import dynamic from "next/dynamic";
import type { QueryFormProps } from "@/components/query-form";

const QueryForm = dynamic(
  () => import("@/components/query-form").then((mod) => mod.QueryForm),
  { ssr: false }
);

export function QueryFormWrapper(props: QueryFormProps) {
  return <QueryForm {...props} />;
}
