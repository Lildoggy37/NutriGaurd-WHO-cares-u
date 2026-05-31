"use client";

import { User, Sparkles, ShieldAlert, Leaf } from "lucide-react";
import type { SSEMessage } from "@/hooks/use-sse";

interface MessageBubbleProps {
  message: SSEMessage;
  isStreaming: boolean;
}

export function MessageBubble({ message, isStreaming }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  if (isSystem) {
    return (
      <div className="flex justify-center py-1">
        <div className="flex items-center gap-1.5 rounded-2xl bg-amber-50 border border-amber-200/60 px-3 py-1.5 max-w-sm">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-500" />
          <span className="text-xs text-amber-700">{message.content}</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex gap-3 py-3 animate-slide-up ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`shrink-0 w-9 h-9 rounded-[42%_58%_63%_37%/_45%_42%_58%_55%] flex items-center justify-center ${
          isUser
            ? "bg-amber-200 text-amber-700"
            : "bg-leaf-200 text-leaf-700"
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Sparkles className="w-4 h-4" />}
      </div>

      {/* Content */}
      <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"}`}>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-leaf-600">
            {isUser ? "你" : "NutriGuard"}
          </span>
          {message.status && (
            <span className="text-xs text-leaf-400">{message.status}</span>
          )}
        </div>

        <div
          className={`rounded-3xl px-4 py-2.5 text-sm leading-relaxed ${
            isUser
              ? "bg-amber-100 text-leaf-800 rounded-br-lg"
              : "bg-white/80 border border-leaf-200/50 text-leaf-900 rounded-bl-lg shadow-sm"
          } ${isStreaming && !isUser ? "streaming-cursor" : ""}`}
        >
          <div className="whitespace-pre-wrap break-words">
            {message.content || (isStreaming ? "" : "...")}
          </div>
        </div>

        {/* Reflection badge */}
        {message.reflection && (
          <div className="mt-1.5">
            {message.reflection.verdict === "CORRECT" ? (
              <span className="inline-flex items-center gap-1 rounded-2xl bg-amber-50 border border-amber-200 px-2 py-0.5 text-xs text-amber-700">
                <ShieldAlert className="w-3 h-3" />
                已修正: {message.reflection.reason}
              </span>
            ) : message.reflection.verdict === "REJECT" ? (
              <span className="inline-flex items-center gap-1 rounded-2xl bg-red-50 border border-red-200 px-2 py-0.5 text-xs text-red-600">
                <ShieldAlert className="w-3 h-3" />
                已拦截
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 rounded-2xl bg-leaf-50 border border-leaf-200 px-2 py-0.5 text-xs text-leaf-600">
                <Leaf className="w-3 h-3" />
                合规审查通过
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
