"use client";

import { useState, useRef, useEffect, KeyboardEvent } from "react";
import { Send, Leaf, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

const QUICK_ACTIONS = [
  { label: "饮食禁忌", query: "帮我查一下糖尿病的饮食禁忌" },
  { label: "记录早餐", query: "帮我记录早餐：2个包子，1杯豆浆" },
  { label: "算热量", query: "帮我算一下今天的热量" },
  { label: "推荐晚餐", query: "根据我的情况，推荐今天的晚餐" },
];

interface ChatInputProps {
  onSend: (message: string) => void;
  isStreaming: boolean;
}

export function ChatInput({ onSend, isStreaming }: ChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (!input.trim() || isStreaming) return;
    onSend(input.trim());
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  return (
    <div className="px-4 pb-4 pt-2">
      {/* Quick actions */}
      <div className="flex gap-2 mb-3 overflow-x-auto pb-1 custom-scrollbar">
        {QUICK_ACTIONS.map((action) => (
          <button
            key={action.label}
            onClick={() => onSend(action.query)}
            disabled={isStreaming}
            className="shrink-0 inline-flex items-center gap-1 rounded-2xl border border-leaf-300/60 bg-white/60 px-3 py-1.5 text-xs text-leaf-600 hover:bg-leaf-50 hover:border-leaf-400 transition-colors disabled:opacity-50"
          >
            <Sparkles className="w-3 h-3" />
            {action.label}
          </button>
        ))}
      </div>

      {/* Input row */}
      <div className="flex items-end gap-2 bg-white/70 backdrop-blur-sm border border-leaf-200/60 rounded-3xl px-4 py-2 shadow-sm focus-within:border-leaf-400 focus-within:ring-2 focus-within:ring-leaf-200 transition-all">
        <Leaf className="w-5 h-5 text-leaf-400 mb-1.5 shrink-0" />
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="询问营养问题或记录饮食..."
          rows={1}
          disabled={isStreaming}
          className="flex-1 resize-none bg-transparent text-sm text-leaf-900 placeholder:text-leaf-300 outline-none py-1.5 max-h-[120px]"
        />
        <Button
          size="icon"
          onClick={handleSend}
          disabled={!input.trim() || isStreaming}
          className="shrink-0 mb-0.5"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
