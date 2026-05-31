import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NutriGuard — AI 健康膳食管家",
  description:
    "基于多智能体的个性化营养助手，支持饮食记录、热量计算、疾病饮食指导",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen solar-gradient">{children}</body>
    </html>
  );
}
