"use client";

import { Leaf, Flame, Apple, Fish, Wheat, Droplets, Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";

interface HealthProfile {
  gender: string;
  age: number;
  height: number;
  weight: number;
  bmi?: number;
  bmr?: number;
  tdee?: number;
  targetCalories?: number;
  todayCalories?: number;
  macros?: { protein: number; fat: number; carbs: number; fiber: number };
  conditions: string[];
}

interface SidebarProps {
  profile: HealthProfile;
  className?: string;
}

function CalorieRing({ current, target }: { current: number; target: number }) {
  const pct = target > 0 ? Math.min((current / target) * 100, 150) : 0;
  const radius = 48;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  const ringColor = pct <= 80 ? "#22c55e" : pct <= 100 ? "#f59e0b" : "#ef4444";

  return (
    <div className="relative w-32 h-32 mx-auto">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
        <circle
          cx="60" cy="60" r={radius}
          fill="none" stroke="#dcfce7" strokeWidth="8"
        />
        <circle
          cx="60" cy="60" r={radius}
          fill="none" stroke={ringColor} strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-leaf-800">{current}</span>
        <span className="text-xs text-leaf-500">/ {target} kcal</span>
        <span className="text-xs text-leaf-400 mt-0.5">
          {pct <= 80 ? "良好" : pct <= 100 ? "注意" : "超标"}
        </span>
      </div>
    </div>
  );
}

function MacroBar({ label, value, unit, icon, color }: {
  label: string; value: number; unit: string; icon: React.ReactNode; color: string;
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-leaf-500 w-8">{icon}</span>
      <span className="text-leaf-700 w-16">{label}</span>
      <div className="flex-1 h-2 bg-leaf-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${Math.min((value / 300) * 100, 100)}%` }}
        />
      </div>
      <span className="text-leaf-600 w-16 text-right font-medium">
        {value}{unit}
      </span>
    </div>
  );
}

export function Sidebar({ profile, className = "" }: SidebarProps) {
  const bmi = profile.bmi ||
    (profile.height > 0 && profile.weight > 0
      ? (profile.weight / Math.pow(profile.height / 100, 2)).toFixed(1)
      : "--");

  return (
    <aside className={`flex flex-col gap-4 p-4 ${className}`}>
      {/* User Header */}
      <Card>
        <CardContent className="pt-5 pb-4">
          <div className="flex items-center gap-3">
            <Avatar className="w-12 h-12">
              <AvatarFallback>
                <Leaf className="w-5 h-5" />
              </AvatarFallback>
            </Avatar>
            <div className="flex-1 min-w-0">
              <h2 className="text-base font-semibold text-leaf-800 truncate">
                健康档案
              </h2>
              <p className="text-xs text-leaf-500">
                {profile.gender || "未知"} · {profile.age || "--"} 岁
              </p>
            </div>
          </div>

          {/* Stats grid */}
          <div className="grid grid-cols-3 gap-2 mt-3">
            <div className="text-center bg-leaf-50 rounded-2xl py-2">
              <div className="text-sm font-bold text-leaf-700">{profile.height || "--"}</div>
              <div className="text-[10px] text-leaf-500">cm</div>
            </div>
            <div className="text-center bg-leaf-50 rounded-2xl py-2">
              <div className="text-sm font-bold text-leaf-700">{profile.weight || "--"}</div>
              <div className="text-[10px] text-leaf-500">kg</div>
            </div>
            <div className="text-center bg-leaf-50 rounded-2xl py-2">
              <div className="text-sm font-bold text-leaf-700">{bmi}</div>
              <div className="text-[10px] text-leaf-500">BMI</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Calorie Ring */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <Flame className="w-4 h-4 text-amber-400" />
            今日热量
          </CardTitle>
        </CardHeader>
        <CardContent>
          <CalorieRing
            current={profile.todayCalories || 0}
            target={profile.targetCalories || 2000}
          />
          {(profile.bmr || profile.tdee) && (
            <div className="flex justify-center gap-4 mt-3 text-xs text-leaf-500">
              {profile.bmr && <span>BMR {profile.bmr}</span>}
              {profile.tdee && <span>TDEE {profile.tdee}</span>}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Macro nutrients */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-leaf-400" />
            宏量营养素目标
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {profile.macros ? (
            <>
              <MacroBar
                label="蛋白质" value={profile.macros.protein} unit="g"
                icon={<Apple className="w-4 h-4" />} color="bg-red-300"
              />
              <MacroBar
                label="碳水" value={profile.macros.carbs} unit="g"
                icon={<Wheat className="w-4 h-4" />} color="bg-amber-300"
              />
              <MacroBar
                label="脂肪" value={profile.macros.fat} unit="g"
                icon={<Droplets className="w-4 h-4" />} color="bg-yellow-300"
              />
              <MacroBar
                label="纤维" value={profile.macros.fiber} unit="g"
                icon={<Leaf className="w-4 h-4" />} color="bg-green-300"
              />
            </>
          ) : (
            <p className="text-xs text-leaf-400 text-center py-2">
              请先完善健康画像以获取宏量目标
            </p>
          )}
        </CardContent>
      </Card>

      {/* Health conditions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">健康状况</CardTitle>
        </CardHeader>
        <CardContent>
          {profile.conditions.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {profile.conditions.map((c) => (
                <Badge key={c} variant="amber">{c}</Badge>
              ))}
            </div>
          ) : (
            <p className="text-xs text-leaf-400">未记录健康状况</p>
          )}
        </CardContent>
      </Card>

      {/* Footer */}
      <div className="mt-auto pt-2">
        <div className="leaf-divider mb-3" />
        <p className="text-[10px] text-leaf-400 text-center">
          NutriGuard · AI 驱动的个性化膳食管家
        </p>
      </div>
    </aside>
  );
}
