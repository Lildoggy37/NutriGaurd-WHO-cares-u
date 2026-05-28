"""
SQLite 持久化层 — 异步 CRUD。

表结构：
  foods      — 食物营养参考库（预填充 30+ 常见食物）
  meals      — 用户餐次记录
  meal_items — 餐次中的具体食物条目
"""
import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nutriguard.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ============================================================
#  Schema
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS foods (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    calories_per_100g REAL,
    protein_per_100g REAL,
    fat_per_100g REAL,
    carbs_per_100g REAL,
    fiber_per_100g REAL,
    gi REAL
);

CREATE TABLE IF NOT EXISTS meals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    meal_type TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meal_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meal_id INTEGER NOT NULL,
    food_name TEXT NOT NULL,
    amount_g REAL NOT NULL,
    calories REAL,
    FOREIGN KEY (meal_id) REFERENCES meals(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meals_user_date
    ON meals(user_id, recorded_at);

CREATE INDEX IF NOT EXISTS idx_foods_name
    ON foods(name);
"""

# ============================================================
#  种子数据 — 常见中国食物营养表（每 100g）
# ============================================================

SEED_FOODS = [
    # === 主食 ===
    ("白米饭",    "主食", 116, 2.6, 0.3, 25.9, 0.3, 83),
    ("糙米饭",    "主食", 111, 2.5, 0.9, 23.0, 1.6, 56),
    ("白馒头",    "主食", 223, 7.0, 1.1, 44.2, 1.3, 88),
    ("全麦面包",  "主食", 247, 8.7, 3.4, 41.3, 6.0, 51),
    ("燕麦片",    "主食", 367, 13.5, 6.7, 61.6, 10.6, 55),
    ("小米粥",    "主食", 46,  1.4, 0.7, 8.4,  0.6, 62),
    ("面条(煮)",  "主食", 110, 3.6, 0.6, 22.2, 0.8, 61),
    ("玉米",      "主食", 112, 4.0, 1.2, 22.8, 2.9, 55),
    ("红薯",      "主食", 86,  1.6, 0.1, 20.1, 3.0, 54),

    # === 肉类 ===
    ("鸡胸肉",    "肉类", 133, 31.0, 1.2, 0.0, 0.0, 0),
    ("猪瘦肉",    "肉类", 143, 20.3, 6.2, 1.5, 0.0, 0),
    ("牛肉(瘦)",  "肉类", 125, 20.2, 4.2, 0.2, 0.0, 0),
    ("鸡蛋",      "肉类", 144, 13.3, 8.8, 2.8, 0.0, 0),
    ("猪肝",      "肉类", 129, 19.3, 3.5, 5.0, 0.0, 0),

    # === 水产 ===
    ("三文鱼",    "水产", 208, 20.4, 13.4, 0.0, 0.0, 0),
    ("虾仁",      "水产", 99,  20.8, 1.2, 0.9, 0.0, 0),
    ("带鱼",      "水产", 127, 17.7, 4.9, 3.1, 0.0, 0),

    # === 蔬菜 ===
    ("西兰花",    "蔬菜", 34,  4.1, 0.6, 4.3, 2.6, 15),
    ("菠菜",      "蔬菜", 23,  2.6, 0.3, 2.8, 1.7, 15),
    ("番茄",      "蔬菜", 19,  0.9, 0.2, 3.5, 1.2, 15),
    ("黄瓜",      "蔬菜", 15,  0.8, 0.2, 2.4, 0.5, 15),
    ("胡萝卜",    "蔬菜", 41,  1.0, 0.2, 8.8, 2.8, 39),
    ("大白菜",    "蔬菜", 13,  1.5, 0.1, 2.1, 0.8, 15),
    ("土豆",      "蔬菜", 76,  2.0, 0.2, 16.5, 2.2, 62),
    ("冬瓜",      "蔬菜", 11,  0.4, 0.2, 1.9, 0.7, 15),

    # === 水果 ===
    ("苹果",      "水果", 52,  0.2, 0.2, 13.5, 2.4, 36),
    ("香蕉",      "水果", 89,  1.1, 0.3, 22.8, 2.6, 52),
    ("橙子",      "水果", 47,  0.9, 0.2, 11.5, 2.4, 43),
    ("西瓜",      "水果", 30,  0.6, 0.1, 7.6,  0.2, 72),
    ("葡萄",      "水果", 67,  0.7, 0.2, 16.3, 0.9, 46),

    # === 豆类/豆制品 ===
    ("豆腐",      "豆类", 81,  8.1, 3.7, 4.2, 0.4, 15),
    ("豆浆",      "豆类", 33,  3.0, 1.3, 2.1, 0.5, 15),
    ("黄豆",      "豆类", 446, 35.0, 20.0, 34.2, 15.5, 18),

    # === 乳制品 ===
    ("全脂牛奶",  "乳制品", 61,  3.0, 3.2, 4.7, 0.0, 28),
    ("无糖酸奶",  "乳制品", 63,  3.5, 3.5, 5.6, 0.0, 30),

    # === 零食/其他 ===
    ("核桃",      "坚果", 654, 15.2, 65.2, 9.6, 6.7, 0),
    ("花生",      "坚果", 567, 25.8, 49.2, 16.1, 8.5, 0),
    ("蜂蜜",      "其他", 304, 0.3, 0.0, 75.6, 0.0, 73),
    ("包子",      "主食", 140, 6.5, 4.0, 20.0, 1.0, 65),
    ("饺子",      "主食", 168, 7.0, 6.0, 21.0, 1.2, 62),
    ("油条",      "零食", 388, 6.9, 17.6, 51.0, 0.9, 75),
]


def init_db():
    """建表并填充种子数据（幂等）"""
    conn = get_connection()
    try:
        conn.executescript(SCHEMA)

        existing = conn.execute("SELECT COUNT(*) FROM foods").fetchone()[0]
        if existing == 0:
            conn.executemany(
                "INSERT INTO foods (name, category, calories_per_100g, protein_per_100g, "
                "fat_per_100g, carbs_per_100g, fiber_per_100g, gi) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                SEED_FOODS,
            )
            conn.commit()
    finally:
        conn.close()


# ============================================================
#  CRUD 操作
# ============================================================

def lookup_food(food_name: str) -> dict | None:
    """模糊匹配食物营养数据"""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM foods WHERE name LIKE ? LIMIT 1",
            (f"%{food_name}%",),
        ).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def log_meal(user_id: str, meal_type: str, food_items: list[dict]) -> int:
    """
    记录一顿饭。
    food_items: [{"name": "鸡胸肉", "amount_g": 200}, ...]
    返回 meal_id
    """
    conn = get_connection()
    now = datetime.now().isoformat()
    try:
        cursor = conn.execute(
            "INSERT INTO meals (user_id, meal_type, recorded_at) VALUES (?, ?, ?)",
            (user_id, meal_type, now),
        )
        meal_id = cursor.lastrowid

        for item in food_items:
            food = lookup_food(item["name"])
            amount = item["amount_g"]
            if food and food["calories_per_100g"]:
                cal = round(food["calories_per_100g"] * amount / 100, 1)
            else:
                cal = None
            conn.execute(
                "INSERT INTO meal_items (meal_id, food_name, amount_g, calories) "
                "VALUES (?, ?, ?, ?)",
                (meal_id, item["name"], amount, cal),
            )

        conn.commit()
        return meal_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_today_meals(user_id: str) -> list[dict]:
    """获取用户今日所有餐食"""
    conn = get_connection()
    today = date.today().isoformat()
    try:
        rows = conn.execute(
            "SELECT m.id, m.meal_type, m.recorded_at, "
            "mi.food_name, mi.amount_g, mi.calories "
            "FROM meals m "
            "LEFT JOIN meal_items mi ON m.id = mi.meal_id "
            "WHERE m.user_id = ? AND DATE(m.recorded_at) = ? "
            "ORDER BY m.recorded_at",
            (user_id, today),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_today_calories(user_id: str) -> tuple[float, list[dict]]:
    """
    返回 (总热量, 餐食明细)
    """
    rows = get_today_meals(user_id)
    total = sum(r["calories"] or 0 for r in rows)
    return round(total, 1), rows


def list_foods(category: str | None = None) -> list[dict]:
    """列出所有食物，可按分类筛选"""
    conn = get_connection()
    try:
        if category:
            rows = conn.execute(
                "SELECT * FROM foods WHERE category = ? ORDER BY name", (category,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM foods ORDER BY category, name").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_food_categories() -> list[str]:
    """获取所有食物分类"""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT DISTINCT category FROM foods ORDER BY category").fetchall()
        return [r["category"] for r in rows]
    finally:
        conn.close()
