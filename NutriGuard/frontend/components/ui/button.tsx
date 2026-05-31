import * as React from "react";
import { cn } from "@/lib/utils";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "default" | "ghost" | "outline";
  size?: "sm" | "default" | "lg" | "icon";
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center gap-2 rounded-3xl font-medium transition-all duration-200",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-leaf-400 focus-visible:ring-offset-2",
          "disabled:pointer-events-none disabled:opacity-50",
          {
            default:
              "bg-leaf-500 text-white hover:bg-leaf-600 shadow-md shadow-leaf-200 active:scale-95",
            ghost: "text-leaf-600 hover:bg-leaf-100",
            outline:
              "border-2 border-leaf-300 text-leaf-700 bg-white/50 hover:bg-leaf-50 hover:border-leaf-400",
          }[variant],
          {
            sm: "h-8 px-3 text-xs",
            default: "h-10 px-5 text-sm",
            lg: "h-12 px-7 text-base",
            icon: "h-10 w-10",
          }[size],
          className,
        )}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { Button };
