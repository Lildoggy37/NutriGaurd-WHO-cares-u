"use client";

import { useState, useRef, useEffect, KeyboardEvent, ChangeEvent, ClipboardEvent } from "react";
import { Send, Leaf, Sparkles, Camera, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { compressImage } from "@/lib/utils";

const QUICK_ACTIONS = [
  { label: "饮食禁忌", query: "帮我查一下糖尿病的饮食禁忌" },
  { label: "记录早餐", query: "帮我记录早餐：2个包子，1杯豆浆" },
  { label: "算热量", query: "帮我算一下今天的热量" },
  { label: "推荐晚餐", query: "根据我的情况，推荐今天的晚餐" },
];

interface ChatInputProps {
  onSend: (message: string, imageBase64?: string) => void;
  isStreaming: boolean;
}

export function ChatInput({ onSend, isStreaming }: ChatInputProps) {
  const [input, setInput] = useState("");
  const [imageBase64, setImageBase64] = useState<string | undefined>(undefined);
  const [imagePreview, setImagePreview] = useState<string | undefined>(undefined);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const clearImage = () => {
    setImageBase64(undefined);
    setImagePreview(undefined);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSend = () => {
    if ((!input.trim() && !imageBase64) || isStreaming) return;
    onSend(input.trim() || "请分析这张图片", imageBase64);
    setInput("");
    clearImage();
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleFileSelect = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const compressed = await compressImage(file);
      // strip data:image/...;base64, prefix
      const b64 = compressed.split(",")[1] || compressed;
      setImageBase64(b64);
      setImagePreview(compressed);
    } catch {
      // fallback: read as-is
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        setImageBase64(result.split(",")[1] || result);
        setImagePreview(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePaste = async (e: ClipboardEvent) => {
    const items = e.clipboardData?.items;
    if (!items) return;
    for (const item of Array.from(items)) {
      if (item.type.startsWith("image/")) {
        e.preventDefault();
        const file = item.getAsFile();
        if (!file) continue;
        try {
          const compressed = await compressImage(file);
          const b64 = compressed.split(",")[1] || compressed;
          setImageBase64(b64);
          setImagePreview(compressed);
        } catch {
          const reader = new FileReader();
          reader.onload = () => {
            const result = reader.result as string;
            setImageBase64(result.split(",")[1] || result);
            setImagePreview(result);
          };
          reader.readAsDataURL(file);
        }
        break;
      }
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
      {/* Image preview */}
      {imagePreview && (
        <div className="mb-2 relative inline-block">
          <img
            src={imagePreview}
            alt="预览"
            className="h-20 rounded-xl border border-leaf-200/60 object-cover"
          />
          <button
            onClick={clearImage}
            className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-leaf-600 text-white flex items-center justify-center shadow"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
      )}

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
        {/* Camera button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isStreaming}
          className="shrink-0 mb-1.5 text-leaf-400 hover:text-leaf-600 transition-colors disabled:opacity-50"
          title="拍照或上传图片"
        >
          <Camera className="w-5 h-5" />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          onChange={handleFileSelect}
          className="hidden"
        />

        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={imageBase64 ? "添加文字描述（可选）..." : "询问营养问题或记录饮食..."}
          rows={1}
          disabled={isStreaming}
          className="flex-1 resize-none bg-transparent text-sm text-leaf-900 placeholder:text-leaf-300 outline-none py-1.5 max-h-[120px]"
        />
        <Button
          size="icon"
          onClick={handleSend}
          disabled={(!input.trim() && !imageBase64) || isStreaming}
          className="shrink-0 mb-0.5"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
