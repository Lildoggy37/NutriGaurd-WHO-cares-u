"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { ChatViewport } from "@/components/chat-viewport";
import { ChatInput } from "@/components/chat-input";
import { useSSEChat } from "@/hooks/use-sse";
import { Leaf, PanelLeftClose, PanelLeft } from "lucide-react";
import { Button } from "@/components/ui/button";

// Demo health profile — will update as user chats
// 初始空状态 — 用户通过对话提供信息后动态更新
const EMPTY_PROFILE = {
  gender: "",
  age: 0,
  height: 0,
  weight: 0,
  bmi: undefined as number | undefined,
  bmr: undefined as number | undefined,
  tdee: undefined as number | undefined,
  targetCalories: 2000,
  todayCalories: 0,
  macros: undefined as { protein: number; fat: number; carbs: number; fiber: number } | undefined,
  conditions: [] as string[],
};

function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "server";
  const key = "nutriguard_session_id";
  let sid = localStorage.getItem(key);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(key, sid);
  }
  return sid;
}

export default function Home() {
  const [sessionId] = useState(getOrCreateSessionId);
  const { messages, currentStatus, isStreaming, sendMessage } = useSSEChat(sessionId);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <div
        className={`shrink-0 transition-all duration-300 ease-in-out overflow-hidden border-r border-leaf-200/60 ${
          sidebarOpen ? "w-[320px]" : "w-0 border-r-0"
        }`}
      >
        <div className="w-[320px] h-full overflow-y-auto custom-scrollbar bg-leaf-50/50">
          <Sidebar profile={EMPTY_PROFILE} />
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="shrink-0 flex items-center gap-3 px-5 py-3 border-b border-leaf-200/60 bg-white/50 backdrop-blur-sm">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="shrink-0"
          >
            {sidebarOpen ? (
              <PanelLeftClose className="w-4 h-4" />
            ) : (
              <PanelLeft className="w-4 h-4" />
            )}
          </Button>

          <div className="w-9 h-9 rounded-[42%_58%_63%_37%/_45%_42%_58%_55%] bg-leaf-200 flex items-center justify-center">
            <Leaf className="w-5 h-5 text-leaf-600" />
          </div>

          <div className="flex-1 min-w-0">
            <h1 className="text-base font-semibold text-leaf-800">NutriGuard</h1>
            <p className="text-xs text-leaf-500">
              {isStreaming ? "思考中..." : "随时为你服务"}
            </p>
          </div>

          {/* Node status indicator in header */}
          {currentStatus && (
            <div className="hidden sm:flex items-center gap-1.5 rounded-2xl bg-leaf-100/80 px-3 py-1 text-xs text-leaf-600 animate-pulse-soft">
              <span className="w-2 h-2 rounded-full bg-leaf-400" />
              {currentStatus}
            </div>
          )}
        </header>

        {/* Chat messages */}
        <ChatViewport
          messages={messages}
          currentStatus={currentStatus}
          isStreaming={isStreaming}
        />

        {/* Input */}
        <ChatInput onSend={sendMessage} isStreaming={isStreaming} />
      </div>
    </div>
  );
}
