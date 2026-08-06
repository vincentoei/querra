import * as React from "react"

import { cn } from "@/lib/utils"

function Textarea({ className, "data-variant": variant, ...props }: React.ComponentProps<"textarea"> & { "data-variant"?: "default" | "code" }) {
  const isCode = variant === "code";
  return (
    <textarea
      data-slot="textarea"
      data-variant={variant}
      className={cn(
        "flex field-sizing-content min-h-16 w-full rounded-lg border border-input px-3 py-2.5 text-sm transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-sky-400 focus-visible:ring-2 focus-visible:ring-sky-400/30 disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-2 aria-invalid:ring-destructive/20",
        isCode && "bg-slate-950 text-slate-50 font-mono leading-relaxed",
        !isCode && "bg-transparent",
        className
      )}
      {...props}
    />
  )
}

export { Textarea }
