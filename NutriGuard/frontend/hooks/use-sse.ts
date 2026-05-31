"use client";

import { useState, useCallback, useRef } from "react";

export interface SSEMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  status?: string;
  reflection?: { verdict: string; reason: string } | null;
}

interface UseSSEChatReturn {
  messages: SSEMessage[];
  currentStatus: string;
  isStreaming: boolean;
  sendMessage: (query: string) => Promise<void>;
  clearMessages: () => void;
}

const BACKEND_URL =
  typeof window !== "undefined"
    ? window.location.protocol + "//" + window.location.host
    : "http://localhost:3000";

export function useSSEChat(sessionId = "default_user"): UseSSEChatReturn {
  const [messages, setMessages] = useState<SSEMessage[]>([]);
  const [currentStatus, setCurrentStatus] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim() || isStreaming) return;

      const userMsg: SSEMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsStreaming(true);
      setCurrentStatus("");

      const assistantId = crypto.randomUUID();
      const assistantMsg: SSEMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
      };
      setMessages((prev) => [...prev, assistantMsg]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, query }),
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            try {
              const data = JSON.parse(line.slice(6));

              switch (data.type) {
                case "text":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId ? { ...m, content: m.content + data.content } : m,
                    ),
                  );
                  break;

                case "status":
                  setCurrentStatus(data.content);
                  break;

                case "error":
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === assistantId
                        ? { ...m, content: m.content + `\n\n⚠️ ${data.content}` }
                        : m,
                    ),
                  );
                  break;

                case "done":
                  break;
              }
            } catch {
              // skip malformed JSON
            }
          }
        }
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") return;
        const errorText = err instanceof Error ? err.message : "未知错误";
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: m.content || `连接失败: ${errorText}` }
              : m,
          ),
        );
      } finally {
        setIsStreaming(false);
        setCurrentStatus("");
      }
    },
    [isStreaming, sessionId],
  );

  const clearMessages = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setIsStreaming(false);
    setCurrentStatus("");
  }, []);

  return { messages, currentStatus, isStreaming, sendMessage, clearMessages };
}
