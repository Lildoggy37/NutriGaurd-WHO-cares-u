"use client";

import { useEffect, useRef } from "react";
import { MessageBubble } from "@/components/message-bubble";
import { AgentStatus } from "@/components/agent-status";
import type { SSEMessage } from "@/hooks/use-sse";
import { Leaf } from "lucide-react";

interface ChatViewportProps {
  messages: SSEMessage[];
  currentStatus: string;
  isStreaming: boolean;
}

export function ChatViewport({ messages, currentStatus, isStreaming }: ChatViewportProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, currentStatus]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="w-20 h-20 rounded-[42%_58%_63%_37%/_45%_42%_58%_55%] bg-leaf-100 flex items-center justify-center mb-6 animate-leaf-sway">
          <Leaf className="w-10 h-10 text-leaf-400" />
        </div>
        <h2 className="text-xl font-semibold text-leaf-700 mb-2">
          你好，我是 NutriGuard
        </h2>
        <p className="text-sm text-leaf-500 text-center max-w-sm leading-relaxed">
          我是你的 AI 营养管家。可以帮你查询疾病饮食禁忌、
          记录每餐热量、生成采购清单，或者给你个性化的营养建议。
        </p>
        <div className="flex gap-2 mt-6 flex-wrap justify-center">
          {["糖尿病怎么吃？", "帮我记录午餐", "今天热量够了吗？"].map((hint) => (
            <span
              key={hint}
              className="text-xs text-leaf-500 bg-leaf-50 border border-leaf-200/60 rounded-2xl px-3 py-1.5"
            >
              {hint}
            </span>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar px-4">
      {messages.map((msg) => (
        <MessageBubble
          key={msg.id}
          message={msg}
          isStreaming={isStreaming && msg.role === "assistant" && msg === messages[messages.length - 1]}
        />
      ))}

      {/* Agent status indicator */}
      <AgentStatus status={currentStatus} isStreaming={isStreaming} />

      <div ref={bottomRef} className="h-1" />
    </div>
  );
}
