import * as React from "react";
import { cn } from "@/lib/utils";

const badgeVariants = {
  leaf: "bg-leaf-100 text-leaf-700 border-leaf-300",
  amber: "bg-amber-100 text-amber-700 border-amber-300",
  earth: "bg-earth-100 text-earth-700 border-earth-300",
  muted: "bg-gray-100 text-gray-600 border-gray-200",
};

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: keyof typeof badgeVariants;
}

function Badge({ className, variant = "muted", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-2xl border px-2.5 py-0.5 text-xs font-medium transition-colors",
        badgeVariants[variant],
        className,
      )}
      {...props}
    />
  );
}

export { Badge };
