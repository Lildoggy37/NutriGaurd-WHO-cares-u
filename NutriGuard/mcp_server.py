import sys
from fastmcp import FastMCP
# ============================
#   统一工具网关FastMCP
# ============================
mcp = FastMCP("NutriGuard_Tools")

# ============================
#   class1  知识检索类 read-only
# ============================
@mcp.tool()
async def search_diet_guidelines(query:str)->str:
    """
    专门用于查询《中国居民膳食指南》和普适性营养原则。
    """
    print(f" [MCP 检索] 正在查阅膳食指南: {query}", file=sys.stderr)
    return f"【模拟检索结果】关于 '{query}' 的膳食指南建议：每天应保持食物多样性，控制添加糖摄入。"

@mcp.tool()
async def check_food_gi(food_name: str) -> str:
    """
    查询特定食物的升糖指数(GI)和血糖负荷(GL)。
    """
    print(f" [MCP 检索] 正在查询 GI 数据库: {food_name}", file=sys.stderr)
    # 模拟 Redis 查表或 Qdrant 查表
    mock_db = {"白米饭": "GI=83 (高)", "糙米饭": "GI=56 (中低)", "燕麦片": "GI=65 (中)"}
    result = mock_db.get(food_name, "GI 数据未知")
    return f"{food_name} 的升糖指数为: {result}"

@mcp.tool()
async def search_medical_taboos(disease_name: str) -> str:
    """
    查询特定慢性病（如糖尿病、痛风）的饮食禁忌和雷区。
    """
    print(f" [MCP 检索] 正在查阅病理禁忌库: {disease_name}", file=sys.stderr)
    return f"【模拟检索结果】{disease_name} 患者应当避免高嘌呤/高糖饮食，具体取决于病理分期。"


# ============================
#   class2  业务行动类 Write N Compute
# ============================
@mcp.tool()
async def log_user_meal(user_id: str, meal_type: str, food_items: str) -> str:
    """
    记录用户的实际饮食（如早餐、午餐）。
    必须参数: user_id (用户唯一ID), meal_type (如'早餐'), food_items (如'2个包子,1杯牛奶')
    """
    print(f" [MCP 行动] 正在写入饮食日志 | 用户:{user_id} | {meal_type}: {food_items}", file=sys.stderr)
    # 在真实项目中，这里会引入 Redis 分布式锁，防止高并发脏写
    return f"已成功为用户 {user_id} 记录{meal_type}: {food_items}。"

@mcp.tool()
async def calculate_daily_calories(user_id: str) -> str:
    """计算用户今天已记录的总热量摄入和剩余配额。"""
    print(f" [MCP 行动] 正在核算今日卡路里 | 用户:{user_id}", file=sys.stderr)
    return f"用户 {user_id} 今日已摄入 1250 kcal，建议剩余摄入 550 kcal。"

@mcp.tool()
async def generate_shopping_list(ingredients: str) -> str:
    """根据食谱生成结构化的超市采购清单。"""
    print(f" [MCP 行动] 正在生成采购清单: {ingredients}", file=sys.stderr)
    return f"【生成成功】采购清单：{ingredients} (已按生鲜、干货分类)。"


if __name__ == "__main__":
    print("🚀 [NutriGuard MCP] 健康膳食微服务已启动...", file=sys.stderr)
    mcp.run()