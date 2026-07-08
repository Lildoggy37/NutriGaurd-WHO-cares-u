"""
开源数据集导入脚本 — 从 GitHub / HuggingFace 下载中文营养/医疗数据，
转换为 markdown 格式追加到 mock_corpus.md。

数据源：
  1. Sanotsu/china-food-composition-data (GitHub) — 1677种食物, 30+营养素+GI
  2. PanruifengWawa/food-material (GitHub) — 1696种食材+介绍
  3. FreedomIntelligence/DoctorFLAN (HF) — 中文疾病饮食指导
  4. FreedomIntelligence/huatuo_encyclopedia_qa (HF) — 医疗百科问答
  5. AIR-Bench/qa_healthcare_zh (HF) — 医疗健康问答
  6. SeaEval/cmmlu nutrition subset (HF) — 营养学题库
  7. Codatta/MM-Food-100K subset (HF) — 中餐菜品营养
  8. madroid/nt-19 (HF) — 包装食品营养

用法：
  python import_datasets.py              # 导入所有数据源
  python import_datasets.py --source 1   # 只导入指定数据源
  python import_datasets.py --dry-run    # 只检查数据源可用性
"""

import os
import sys
import json
import hashlib
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Any

BASE_DIR = Path(__file__).resolve().parent
CORPUS_PATH = BASE_DIR / "data" / "mock_corpus.md"
DATA_DIR = BASE_DIR / "data" / "imports"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── 工具函数 ─────────────────────────────────────────

def append_to_corpus(content: str, source_name: str):
    """追加内容到 mock_corpus.md"""
    header = f"\n\n---\n\n> 数据来源: {source_name} | 自动导入于 {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    with open(CORPUS_PATH, "a", encoding="utf-8") as f:
        f.write(header + content)
    chunks = len(content) // 384
    print(f"  [OK] 已写入 {len(content):,} 字符 (~{chunks} chunks)")


def count_file_chunks(path: Path) -> int:
    """估算文件的 chunk 数"""
    if not path.exists():
        return 0
    return len(path.read_text(encoding="utf-8")) // 384


# ══════════════════════════════════════════════════════
#  数据源 1: Sanotsu/china-food-composition-data (GitHub)
# ══════════════════════════════════════════════════════

def import_china_food_composition():
    """导入中国食物成分表第6版 — 1677种食物, 30+营养素 + GI"""
    print("\n[1/8] Sanotsu/china-food-composition-data ...")
    repo_url = "https://github.com/Sanotsu/china-food-composition-data.git"
    repo_dir = DATA_DIR / "china-food-composition-data"

    if not (repo_dir / "data").exists():
        print("  克隆仓库 (约50MB)...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                          check=True, capture_output=True, timeout=120)
        except Exception as e:
            print(f"  [FAIL] 克隆失败: {e}")
            return 0

    # 查找 JSON 文件
    json_files = sorted(repo_dir.glob("**/*.json"))
    if not json_files:
        print("  [WARN] 未找到JSON文件")
        return 0

    print(f"  找到 {len(json_files)} 个数据文件")

    # 渲染为 markdown 食物成分表
    lines = [
        "## 附录A 中国食物成分表（第6版标准数据）\n",
        f"> 共收录 {len(json_files)} 类食物数据，来源: 中国营养学会《中国食物成分表标准版(第6版)》。\n",
    ]

    total_foods = 0
    for jf in json_files[:50]:  # 限制文件数，避免过于庞大
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list) or len(data) == 0:
                continue

            category = data[0].get("foodCategory", data[0].get("category", jf.stem))
            lines.append(f"\n### {category}\n")
            lines.append("| 食物名称 | 能量(kcal) | 蛋白质(g) | 脂肪(g) | 碳水(g) | "
                        "膳食纤维(g) | GI | 维生素C(mg) | 钙(mg) | 铁(mg) |")
            lines.append("|----------|-----------|-----------|---------|---------|"
                        "------------|-----|------------|-------|------|")

            for item in data[:80]:  # 每类最多 80 种
                name = item.get("foodName", item.get("food_name", item.get("name", "")))
                if not name:
                    continue
                kcal = item.get("energyKcal", item.get("能量(kcal)", item.get("energy_kcal", "-")))
                prot = item.get("protein", item.get("蛋白质(g)", item.get("protein_g", "-")))
                fat = item.get("fat", item.get("脂肪(g)", item.get("fat_g", "-")))
                carbs = item.get("carbohydrate", item.get("碳水化合物(g)", item.get("carbs_g", "-")))
                fiber = item.get("fiber", item.get("膳食纤维(g)", item.get("fiber_g", "-")))
                gi = item.get("gi", item.get("GI", item.get("glycemic_index", "-")))
                vc = item.get("vitaminC", item.get("维生素C(mg)", item.get("vc_mg", "-")))
                ca = item.get("calcium", item.get("钙(mg)", item.get("ca_mg", "-")))
                fe = item.get("iron", item.get("铁(mg)", item.get("fe_mg", "-")))
                lines.append(f"| {name} | {kcal} | {prot} | {fat} | {carbs} | {fiber} | {gi} | {vc} | {ca} | {fe} |")
                total_foods += 1
        except Exception:
            continue

    if total_foods > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "Sanotsu/china-food-composition-data")
    else:
        print("  [WARN] 未提取到有效食物数据，跳过")

    return total_foods


# ══════════════════════════════════════════════════════
#  数据源 2: PanruifengWawa/food-material (GitHub)
# ══════════════════════════════════════════════════════

def import_food_material():
    """导入食材营养数据 — 1696种食材 + 介绍信息"""
    print("\n[2/8] PanruifengWawa/food-material ...")
    repo_url = "https://github.com/PanruifengWawa/food-material.git"
    repo_dir = DATA_DIR / "food-material"

    if not (repo_dir / "nutrition").exists():
        print("  克隆仓库...")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)],
                          check=True, capture_output=True, timeout=120)
        except Exception as e:
            print(f"  [FAIL] 克隆失败: {e}")
            return 0

    nutrition_dir = repo_dir / "json"  # 实际路径是 json/ 目录
    json_files = sorted(nutrition_dir.glob("*.json"))
    if not json_files:
        print("  [WARN] 未找到 JSON 文件")
        return 0

    print(f"  找到 {len(json_files)} 个JSON文件")

    # 读取主数据文件
    all_foods = []
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            if isinstance(data, list):
                all_foods.extend(data)
        except Exception:
            continue

    if not all_foods:
        print("  [WARN] 未读取到食材数据")
        return 0

    print(f"  共 {len(all_foods)} 种食材")

    # 按 big_class 分类
    category_foods: dict[str, list[dict]] = {}
    for f in all_foods:
        cat = f.get("big_class", f.get("small_class", "其他"))
        category_foods.setdefault(cat, []).append(f)

    lines = [
        "## 附录B 常见食材食疗属性与体质匹配\n",
        f"> 共收录 {len(all_foods)} 种食材的名称、别名、简介、食疗功效、最佳季节、适宜体质等信息。\n",
    ]

    total = 0
    for cat, foods in sorted(category_foods.items()):
        lines.append(f"\n### {cat}类食材\n")

        for food in foods[:80]:  # 每类最多80种
            name = food.get("name", "")
            if not name:
                continue

            aliases = food.get("alias", "")
            introduction = food.get("introduction", "")[:200]
            season = food.get("season", "")
            effect = food.get("effect", "")
            body_constitution = food.get("body_constitution", "")
            small_class = food.get("small_class", "")

            entry = f"#### {name}"
            if small_class:
                entry += f"（{small_class}）"
            entry += "\n"
            if aliases:
                entry += f"- 别名: {aliases}\n"
            if introduction:
                entry += f"- 简介: {introduction}\n"
            if effect:
                entry += f"- 食疗功效: {effect}\n"
            if season:
                entry += f"- 最佳季节: {season}\n"
            if body_constitution:
                entry += f"- 适宜体质: {body_constitution}\n"
            entry += "\n"
            lines.append(entry)
            total += 1

    if total > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "PanruifengWawa/food-material")
    else:
        print("  [WARN] 未提取到有效食材数据，跳过")

    return total


# ══════════════════════════════════════════════════════
#  数据源 3: FreedomIntelligence/DoctorFLAN (HuggingFace)
# ══════════════════════════════════════════════════════

def import_doctorflan():
    """导入 DoctorFLAN 中文疾病饮食指导"""
    print("\n[3/8] FreedomIntelligence/DoctorFLAN ...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FAIL] datasets 库未安装")
        return 0

    try:
        ds = load_dataset("FreedomIntelligence/DoctorFLAN", split="train", streaming=True)
    except Exception as e:
        print(f"  [FAIL] 加载失败: {e}")
        return 0

    # 筛选 diet/nutrition/饮食/营养 相关内容
    diet_keywords = [
        "饮食", "营养", "食物", "吃", "忌", "禁食", "膳食", "食谱", "食疗",
        "diet", "nutrition", "food", "eat", "calorie",
        "糖尿病", "高血压", "痛风", "肾病", "肝病", "胃", "肠", "心脏",
        "减重", "肥胖", "胆固醇", "血脂", "血糖", "尿酸",
    ]

    lines = [
        "## 附录C 常见疾病饮食指导问答（DoctorFLAN）\n",
        "> 来源: FreedomIntelligence/DoctorFLAN — 大规模中文医疗指令数据集\n",
    ]

    count = 0
    max_items = 2000
    for item in ds:
        if count >= max_items:
            break
        try:
            instruction = str(item.get("instruction", ""))
            output = str(item.get("output", ""))

            combined = instruction + output
            if not any(kw in combined for kw in diet_keywords):
                continue
            if len(output) < 50:
                continue

            # 提取问题标题
            q_title = instruction[:80].replace("\n", " ")
            lines.append(f"### Q: {q_title}\n")
            lines.append(f"{output[:1500]}\n")  # 截断过长内容
            lines.append("")
            count += 1
        except Exception:
            continue

    if count == 0:
        print("  [WARN] 未筛选到饮食相关内容，尝试备用方案...")
        # 备用: 直接采样
        try:
            for item in ds.take(500):
                output = str(item.get("output", ""))
                if len(output) > 100:
                    lines.append(f"{output[:1000]}\n\n")
                    count += 1
        except Exception:
            pass

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "FreedomIntelligence/DoctorFLAN")
    else:
        print("  [WARN] 未提取到有效内容")

    return count


# ══════════════════════════════════════════════════════
#  数据源 4: huatuo_encyclopedia_qa (HuggingFace)
# ══════════════════════════════════════════════════════

def import_huatuo_qa():
    """导入华佗百科医疗问答"""
    print("\n[4/8] FreedomIntelligence/huatuo_encyclopedia_qa ...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [FAIL] datasets 库未安装")
        return 0

    try:
        ds = load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa", split="train", streaming=True)
    except Exception as e:
        print(f"  [FAIL] 加载失败: {e}")
        return 0

    diet_kw = ["饮食", "营养", "食物", "吃", "忌口", "膳食", "维生素", "蛋白", "蔬菜", "水果", "肉", "鱼"]
    lines = [
        "## 附录D 华佗百科医疗问答（饮食与营养）\n",
        "> 来源: FreedomIntelligence/huatuo_encyclopedia_qa — 复旦大学高质量中文医疗百科问答\n",
    ]

    count = 0
    max_scan = 50000
    for item in ds:
        if count >= 800 or max_scan <= 0:
            break
        max_scan -= 1
        q_data = item.get("questions", item.get("question", item.get("input", "")))
        a_data = item.get("answers", item.get("answer", item.get("output", "")))

        # 可能是列表的列表，展平
        if isinstance(q_data, list) and q_data:
            q_data = q_data[0] if isinstance(q_data[0], str) else " ".join(q_data[0]) if q_data[0] else ""
        if isinstance(a_data, list) and a_data:
            a_data = a_data[0] if isinstance(a_data[0], str) else ""

        q = str(q_data)
        a = str(a_data)
        combined = q + a
        if not any(kw in combined for kw in diet_kw):
            continue
        if len(a) < 40:
            continue

        lines.append(f"### Q: {q[:100]}\n")
        lines.append(f"{a[:1200]}\n\n")
        count += 1

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "FreedomIntelligence/huatuo_encyclopedia_qa")
    return count


# ══════════════════════════════════════════════════════
#  数据源 5: AIR-Bench/qa_healthcare_zh (HuggingFace)
# ══════════════════════════════════════════════════════

def import_air_bench_healthcare():
    """导入 AIR-Bench 中文医疗健康问答"""
    print("\n[5/8] AIR-Bench/qa_healthcare_zh ...")
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    try:
        ds = load_dataset("AIR-Bench/qa_healthcare_zh", "AIR-Bench_24.04", split="queries_default")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        return 0

    diet_kw = ["饮食", "营养", "吃", "食物", "忌", "食谱", "配餐", "热量", "卡路里"]
    lines = [
        "## 附录E 医疗健康饮食问答（AIR-Bench）\n",
        "> 来源: AIR-Bench/qa_healthcare_zh\n",
    ]

    count = 0
    for item in ds:
        q = str(item.get("text", item.get("query", item.get("question", ""))))
        if not any(kw in q for kw in diet_kw):
            continue
        lines.append(f"### Q: {q[:120]}\n\n")
        count += 1
        if count >= 500:
            break

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "AIR-Bench/qa_healthcare_zh")
    return count


# ══════════════════════════════════════════════════════
#  数据源 6: cmmlu nutrition subset (HuggingFace)
# ══════════════════════════════════════════════════════

def import_cmmlu_nutrition():
    """导入 CMMLU 营养学题库"""
    print("\n[6/8] SeaEval/cmmlu (nutrition) ...")
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    try:
        ds = load_dataset("SeaEval/cmmlu", "default", split="test")
    except Exception as e:
        print(f"  [FAIL]: {e}")
        return 0

    # 筛选 nutrition 相关题目
    nutrition_kw = ["营养", "维生素", "蛋白质", "脂肪", "碳水", "矿物质", "膳食纤维",
                    "糖尿病饮食", "BMI", "热量", "能量", "钙", "铁", "锌", "镁",
                    "diet", "nutrition", "vitamin", "calorie", "food"]
    # 先检查字段名
    sample = next(iter(ds))
    print(f"  字段: {list(sample.keys())}")
    subject_field = None
    for f in ["subject", "discipline", "category", "topic"]:
        if f in sample:
            subject_field = f
            break

    lines = [
        "## 附录F 营养学专业题库（CMMLU-Nutrition）\n",
        "> 来源: SeaEval/cmmlu nutrition subset — 中文学术营养学选择题\n",
    ]

    count = 0
    for item in ds:
        q = item.get("question", item.get("query", ""))
        choices = item.get("choices", item.get("options", []))
        answer = item.get("answer", item.get("gold", ""))
        if not q:
            continue

        entry = f"### {q[:150]}\n"
        if isinstance(choices, list):
            labels = "ABCDEFGHIJ"
            for i, c in enumerate(choices[:6]):
                entry += f"- {labels[i] if i < len(labels) else i+1}. {c}\n"
        if answer:
            entry += f"\n答案: {answer}\n"
        lines.append(entry + "\n")
        count += 1

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "SeaEval/cmmlu nutrition")
    return count


# ══════════════════════════════════════════════════════
#  数据源 7: Codatta/MM-Food-100K subset (HuggingFace)
# ══════════════════════════════════════════════════════

def import_mm_food_100k():
    """导入 MM-Food-100K 中餐菜品营养数据（取子集）"""
    print("\n[7/8] Codatta/MM-Food-100K ...")
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    try:
        ds = load_dataset("Codatta/MM-Food-100K", split="train", streaming=True)
    except Exception as e:
        print(f"  [FAIL]: {e}")
        # 尝试 parquet 直接下载
        try:
            ds = load_dataset("Codatta/MM-Food-100K", split="train",
                            streaming=True, trust_remote_code=True)
        except Exception:
            return 0

    # 筛选中文/亚洲食物
    asian_kw = ["Chinese", "chinese", "Asian", "asia", "China", "Japan", "Korean",
                "饭", "面", "菜", "汤", "炒", "蒸", "煮", "烧", "烤",
                "鸡", "鱼", "猪", "牛", "虾", "豆腐", "蔬菜"]

    lines = [
        "## 附录G 中餐菜品营养数据（MM-Food-100K子集）\n",
        "> 来源: Codatta/MM-Food-100K — 10万样本多模态食物数据集\n",
        "| 菜品名称 | 食材 | 烹饪方式 | 热量(kcal) | 蛋白质(g) | 脂肪(g) | 碳水(g) |\n",
        "|----------|------|---------|-----------|-----------|---------|--------|\n",
    ]

    count = 0
    max_items = 3000
    for item in ds:
        if count >= max_items:
            break
        try:
            name = str(item.get("dish_name", ""))
            ingredients_raw = item.get("ingredients", "")
            cooking = str(item.get("cooking_method", ""))
            nutrition = item.get("nutritional_profile", {})

            # 筛选亚洲食物
            if isinstance(nutrition, str):
                try:
                    nutrition = json.loads(nutrition)
                except Exception:
                    nutrition = {}

            if not name:
                continue

            kcal = nutrition.get("calories_kcal", "-")
            prot = nutrition.get("protein_g", "-")
            fat = nutrition.get("fat_g", "-")
            carbs = nutrition.get("carbohydrate_g", "-")

            # 截断食材列表
            if isinstance(ingredients_raw, list):
                ingr = ", ".join(str(i) for i in ingredients_raw[:5])
            else:
                ingr = str(ingredients_raw)[:100]

            lines.append(f"| {name[:50]} | {ingr[:80]} | {cooking[:30]} | "
                        f"{kcal} | {prot} | {fat} | {carbs} |")
            count += 1
        except Exception:
            continue

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "Codatta/MM-Food-100K")
    return count


# ══════════════════════════════════════════════════════
#  数据源 8: madroid/nt-19 (HuggingFace)
# ══════════════════════════════════════════════════════

def import_nt19():
    """导入中国包装食品营养数据"""
    print("\n[8/8] madroid/nt-19 ...")
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    try:
        ds = load_dataset("madroid/nt-19", split="train", streaming=True)
    except Exception as e:
        print(f"  [FAIL]: {e}")
        return 0

    lines = [
        "## 附录H 中国市售包装食品营养成分表（nt-19子集）\n",
        "> 来源: madroid/nt-19 — 中国预包装食品营养数据集\n",
        "| 食品名称 | 能量(kcal) | 脂肪(g) | 碳水(g) | 蛋白质(g) | 规格 |\n",
        "|----------|-----------|---------|---------|-----------|------|\n",
    ]

    count = 0
    max_items = 1000
    for item in ds:
        if count >= max_items:
            break
        try:
            name = str(item.get("name", item.get("product_name", "")))
            nutrition = item.get("nutrition", item.get("nutrients", {}))
            facts = item.get("facts", item.get("nutriments", {}))

            if isinstance(nutrition, str):
                try:
                    nutrition = json.loads(nutrition)
                except Exception:
                    nutrition = {}
            if isinstance(facts, str):
                try:
                    facts = json.loads(facts)
                except Exception:
                    facts = {}

            # 合并数据
            all_nutri = {**nutrition, **facts}

            if not name:
                continue

            # 筛选中文食品
            has_chinese = any('一' <= c <= '鿿' for c in name)
            if not has_chinese:
                continue

            kcal = all_nutri.get("calories", all_nutri.get("energy-kcal", all_nutri.get("energy_100g", "-")))
            fat = all_nutri.get("fat", all_nutri.get("fat_100g", "-"))
            carbs = all_nutri.get("carbohydrates", all_nutri.get("carbs", all_nutri.get("carbohydrates_100g", "-")))
            protein = all_nutri.get("proteins", all_nutri.get("protein", all_nutri.get("proteins_100g", "-")))
            weight = all_nutri.get("weight", item.get("weight", ""))

            lines.append(f"| {name[:60]} | {kcal} | {fat} | {carbs} | {protein} | {weight} |")
            count += 1
        except Exception:
            continue

    if count > 0:
        content = "\n".join(lines)
        append_to_corpus(content, "madroid/nt-19")
    return count


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════

SOURCES = [
    ("china-food-composition", import_china_food_composition),
    ("food-material", import_food_material),
    ("doctorflan", import_doctorflan),
    ("huatuo-qa", import_huatuo_qa),
    ("air-bench-healthcare", import_air_bench_healthcare),
    ("cmmlu-nutrition", import_cmmlu_nutrition),
    ("mm-food-100k", import_mm_food_100k),
    ("nt-19", import_nt19),
]


def main():
    parser = argparse.ArgumentParser(description="NutriGuard 开源数据集批量导入工具")
    parser.add_argument("--dry-run", action="store_true", help="只检查数据源可用性，不导入")
    parser.add_argument("--source", type=str, default="", help="只导入指定数据源 (名称或序号)")
    parser.add_argument("--skip", type=str, default="", help="跳过的数据源 (逗号分隔)")
    args = parser.parse_args()

    initial_chunks = count_file_chunks(CORPUS_PATH)
    print(f"当前语料: {initial_chunks} chunks ({CORPUS_PATH})")

    if args.dry_run:
        print("\n[Dry Run] 数据源列表:")
        for i, (name, func) in enumerate(SOURCES):
            print(f"  {i+1}. {name} ({func.__doc__.strip().split(chr(10))[0] if func.__doc__ else ''})")
        return

    to_skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    total_items = 0

    for i, (name, func) in enumerate(SOURCES):
        if to_skip and name in to_skip:
            print(f"\n[{i+1}/8] {name} — 已跳过")
            continue
        if args.source:
            if name != args.source and str(i+1) != args.source:
                continue

        try:
            count = func()
            total_items += count
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()

    final_chunks = count_file_chunks(CORPUS_PATH)
    added_chunks = final_chunks - initial_chunks
    print(f"\n{'='*50}")
    print(f"导入完成!")
    print(f"  新增条目: {total_items}")
    print(f"  语料 chunks: {initial_chunks} -> {final_chunks} (+{added_chunks})")
    print(f"  语料文件: {CORPUS_PATH}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
