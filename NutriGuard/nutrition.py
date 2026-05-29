"""
个性化营养计算引擎 — 纯函数，无副作用。

公式来源：
  - BMR: Mifflin-St Jeor (1990)，被 ADA 推荐为最准确的静息代谢率公式
  - TDEE: BMR × 活动系数 (WHO/FAO 标准)
  - 疾病调整: 基于《中国居民膳食指南 2022》及临床营养学共识
"""

from dataclasses import dataclass, field
from typing import Literal

Gender = Literal["男", "女"]
ActivityLevel = Literal["久坐", "轻度", "中度", "活跃", "极活跃"]

# TDEE 活动系数
ACTIVITY_FACTORS: dict[ActivityLevel, float] = {
    "久坐": 1.20,   # 几乎不运动
    "轻度": 1.375,  # 每周 1-3 天轻度运动
    "中度": 1.55,   # 每周 3-5 天中等运动
    "活跃": 1.725,  # 每周 6-7 天高强度运动
    "极活跃": 1.90,  # 高强度体力劳动/运动员
}


@dataclass
class MacroTarget:
    """宏量营养素目标"""
    calories: float       # kcal
    protein_g: float      # 蛋白质 (g)
    fat_g: float          # 脂肪 (g)
    carbs_g: float        # 碳水 (g)
    fiber_g: float = 25.0  # 膳食纤维 (g)


@dataclass
class DailyTarget:
    """每日营养目标"""
    bmr: float
    tdee: float
    adjusted_calories: float
    macros: MacroTarget
    conditions: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ============================================================
#  核心计算
# ============================================================

def calc_bmr(gender: Gender, weight_kg: float, height_cm: float, age: int) -> float:
    """
    Mifflin-St Jeor 公式。

    男: BMR = 10×体重 + 6.25×身高 - 5×年龄 + 5
    女: BMR = 10×体重 + 6.25×身高 - 5×年龄 - 161
    """
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    return round(base + 5 if gender == "男" else base - 161, 1)


def calc_tdee(bmr: float, activity_level: ActivityLevel = "久坐") -> float:
    """总每日能耗 = BMR × 活动系数"""
    factor = ACTIVITY_FACTORS.get(activity_level, 1.2)
    return round(bmr * factor, 1)


# ============================================================
#  疾病调整
# ============================================================

# 每种疾病的目标宏量营养素配比 (蛋白质% / 脂肪% / 碳水%)
DISEASE_MACRO_RATIOS = {
    "糖尿病":        (0.20, 0.30, 0.50),  # 控碳水，稳血糖
    "妊娠期糖尿病":   (0.20, 0.30, 0.50),  # 同糖尿病，略微放宽
    "痛风":          (0.20, 0.30, 0.50),  # 控嘌呤 (蛋白以蛋奶为主)
    "高尿酸血症":     (0.20, 0.30, 0.50),  # 同痛风
    "高血压":        (0.20, 0.25, 0.55),  # DASH 饮食：低脂高碳
    "高血脂":        (0.20, 0.25, 0.55),  # 严格控脂
    "肥胖":          (0.25, 0.25, 0.50),  # 高蛋白保肌 + 热量缺口
    "肾病":          (0.15, 0.30, 0.55),  # 严格控蛋白
}
DEFAULT_RATIO = (0.20, 0.30, 0.50)  # 20% 蛋白, 30% 脂肪, 50% 碳水

# 疾病热量调整系数 (>1 表示需要更多热量)
DISEASE_CALORIE_ADJUST = {
    "糖尿病":        1.00,  # 维持当前 TDEE
    "妊娠期糖尿病":   1.15,  # 孕期 +15%
    "痛风":          0.95,  # 轻微减重有利于降低尿酸
    "高尿酸血症":     0.95,
    "高血压":        0.95,  # 减重降压
    "高血脂":        0.95,
    "肥胖":          0.80,  # 500 kcal 热量缺口
    "肾病":          1.00,
}

# 疾病特化提示
DISEASE_NOTES = {
    "糖尿病": (
        "碳水来源优先选择低 GI 食物，每日碳水控制在 {carbs_g}g 以内，"
        "分配至 3 餐 + 2 加餐以平稳血糖"
    ),
    "痛风": (
        "蛋白质以鸡蛋、低脂奶制品为主，限制红肉和海鲜，"
        "每日饮水 ≥2000ml 促进尿酸排泄"
    ),
    "高血压": (
        "遵循 DASH 饮食原则：高钾低钠，每日食盐 ≤5g，"
        "增加蔬菜水果摄入，限制加工食品"
    ),
    "高血脂": "严格限制饱和脂肪和反式脂肪，增加 Omega-3 摄入（深海鱼、亚麻籽油）",
    "肥胖": "当前目标含约 500 kcal 热量缺口，配合每周 ≥150 分钟中等强度运动",
    "妊娠期糖尿病": "少食多餐，每日 5-6 餐，严格控制精制碳水，监测餐后 2h 血糖",
    "肾病": "严格控制蛋白质摄入量，优先选择优质蛋白（鸡蛋、鱼肉），限制磷和钾",
}


def _best_match_condition(conditions: list[str]) -> str | None:
    """从用户疾病列表中匹配最相关的那一个"""
    for c in conditions:
        if c in DISEASE_MACRO_RATIOS:
            return c
    return None


def calc_macros(calories: float, protein_pct: float, fat_pct: float, carbs_pct: float) -> MacroTarget:
    """
    热量 → 宏量营养素克数。
    蛋白 4 kcal/g, 碳水 4 kcal/g, 脂肪 9 kcal/g
    """
    return MacroTarget(
        calories=round(calories, 0),
        protein_g=round(calories * protein_pct / 4, 1),
        fat_g=round(calories * fat_pct / 9, 1),
        carbs_g=round(calories * carbs_pct / 4, 1),
    )


def calculate_daily_target(
    gender: Gender = "男",
    age: int = 30,
    height_cm: float = 170,
    weight_kg: float = 70,
    activity_level: ActivityLevel = "久坐",
    conditions: list[str] | None = None,
) -> DailyTarget:
    """
    一站式计算：BMR → TDEE → 疾病调整 → 宏量营养素配比。
    """
    conditions = conditions or []
    bmr = calc_bmr(gender, weight_kg, height_cm, age)
    tdee = calc_tdee(bmr, activity_level)

    # 疾病调整
    matched = _best_match_condition(conditions)
    ratio = DISEASE_MACRO_RATIOS.get(matched, DEFAULT_RATIO) if matched else DEFAULT_RATIO
    calorie_adj = DISEASE_CALORIE_ADJUST.get(matched, 1.0) if matched else 1.0

    adjusted_calories = round(tdee * calorie_adj, 1)
    macros = calc_macros(adjusted_calories, *ratio)

    notes = []
    if matched and matched in DISEASE_NOTES:
        note = DISEASE_NOTES[matched].format(carbs_g=macros.carbs_g)
        notes.append(note)
    if not matched and conditions:
        notes.append(f"「{conditions[0]}」暂无内置调整方案，使用默认营养配比。建议咨询营养师。")

    return DailyTarget(
        bmr=bmr,
        tdee=tdee,
        adjusted_calories=adjusted_calories,
        macros=macros,
        conditions=conditions,
        notes=notes,
    )


def format_target_report(target: DailyTarget, user_id: str = "") -> str:
    """将 DailyTarget 格式化为用户可读的汇报文本"""
    m = target.macros
    lines = [
        f"用户 {user_id} 个性化营养目标：",
        f"  基础代谢 (BMR): {target.bmr} kcal",
        f"  每日消耗 (TDEE): {target.tdee} kcal",
    ]
    if target.conditions:
        lines.append(f"  疾病调整: {', '.join(target.conditions)}")
    lines.extend([
        f"  目标摄入: {target.adjusted_calories} kcal/天",
        f"",
        f"  宏量营养素配比：",
        f"    蛋白质: {m.protein_g}g ({round(m.protein_g * 4)} kcal)",
        f"    脂肪:   {m.fat_g}g ({round(m.fat_g * 9)} kcal)",
        f"    碳水:   {m.carbs_g}g ({round(m.carbs_g * 4)} kcal)",
        f"    膳食纤维: ≥{m.fiber_g}g",
    ])
    for note in target.notes:
        lines.append(f"\n  {note}")

    return "\n".join(lines)
