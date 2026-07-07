"use client";

import { Leaf, Brain, Search, Utensils, MessageSquare, Shield, Camera } from "lucide-react";

const nodeIcons: Record<string, React.ReactNode> = {
  supervisor: <Brain className="w-3.5 h-3.5" />,
  rag_expert: <Search className="w-3.5 h-3.5" />,
  rag_reflection: <Shield className="w-3.5 h-3.5" />,
  action_expert: <Utensils className="w-3.5 h-3.5" />,
  slot_filler: <MessageSquare className="w-3.5 h-3.5" />,
  vision_expert: <Camera className="w-3.5 h-3.5" />,
  memory_compressor: <Brain className="w-3.5 h-3.5" />,
};

const nodeLabels: Record<string, string> = {
  supervisor: "路由分析",
  rag_expert: "知识检索",
  rag_reflection: "合规审查",
  action_expert: "执行操作",
  slot_filler: "补充信息",
  vision_expert: "图像分析",
  memory_compressor: "记忆压缩",
};

function extractNodeName(status: string): string | null {
  const match = status.match(/\[(\w+)\]/);
  return match ? match[1] : null;
}

interface AgentStatusProps {
  status: string;
  isStreaming: boolean;
}

export function AgentStatus({ status, isStreaming }: AgentStatusProps) {
  if (!isStreaming && !status) return null;

  const nodeName = status ? extractNodeName(status) : null;
  const label = nodeName ? nodeLabels[nodeName] || nodeName : "";
  const icon = nodeName ? nodeIcons[nodeName] : <Leaf className="w-3.5 h-3.5" />;

  return (
    <div className="flex items-center gap-2 px-4 py-2 animate-fade-in">
      <div className="flex items-center gap-1.5 rounded-2xl bg-leaf-100/80 border border-leaf-300/60 px-3 py-1.5">
        <span className="text-leaf-600 animate-pulse-soft">{icon}</span>
        <span className="text-xs text-leaf-700 font-medium">
          {label || status}
        </span>
        <span className="flex gap-0.5 ml-1">
          <span className="w-1 h-1 rounded-full bg-leaf-400 animate-bounce" style={{ animationDelay: "0ms" }} />
          <span className="w-1 h-1 rounded-full bg-leaf-400 animate-bounce" style={{ animationDelay: "150ms" }} />
          <span className="w-1 h-1 rounded-full bg-leaf-400 animate-bounce" style={{ animationDelay: "300ms" }} />
        </span>
      </div>
    </div>
  );
}
