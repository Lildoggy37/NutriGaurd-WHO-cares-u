import operator
import json
from typing import Annotated, Sequence, TypedDict, Literal, Dict
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# =====================================
#    1. 全局状态机
# =====================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node:str
    user_profile: Dict[str,str]

# =====================================
#    2. 大模型配置
# =====================================
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.0,
    api_key="DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com/v1"
)

# =====================================
#    3. Supervisor
# =====================================
class RouteDecision(BaseModel):
    next_node: Literal["rag_expert", "action_expert", "slot_filler", "FINISH"]
    reason: str # 大模型思考过程

supervisor_llm = llm.with_structured_output(RouteDecision)

async def supervisor_node(state:AgentState):
    print("[Supervisor] 正在审视意图...")
    # 将user)orifuke转换为字符串给llm
    profile_str = json.dumps(state.get("user_profile",{}),ensure_ascii=False)

    sys_prompt = f"""你是一个高级医疗健康分发路由。
    当前用户的已知健康画像：{profile_str}
    
    请根据历史对话，决定下一步操作：
    - 如果用户在询问疾病禁忌、指南知识，选 'rag_expert'
    - 如果用户想记录今天吃了什么、算热量，选 'action_expert'
    - 如果用户的话没说完（例如说“帮我记下早饭”但没说吃了啥），选 'slot_filler'
    - 如果问题已经彻底解答完毕，选 'FINISH'
    """

    messages = [SystemMessage(content=sys_prompt) + state["messages"]]
    decision = await supervisor_llm.ainvoke(messages)

    print(f" [路由分发] 决定去向: {decision.next_node} | 理由: {decision.reason}")
    return {"next_node": decision.next_node}

# =====================================
#    4. worker node
# =====================================
async def rag_expert_node(state: AgentState):
    print(" [RAG Expert] 正在查阅健康知识库...")
    # 真实场景下，这里会绑定 check_food_gi 等检索类工具
    # 为了演示状态流转，我们模拟返回
    response = "根据《膳食指南》，该食物升糖指数较高，建议减量。"
    return {
        "messages": [SystemMessage(content=response, name="rag_expert")],
        "next_node": "supervisor" # 干完活必须强制交还控制权！
    }

async def action_expert_node(state: AgentState):
    print(" [Action Expert] 正在执行业务操作...")
    response = "【系统操作】已成功为您记录日志，并核算当日热量剩余 500大卡。"
    return {
        "messages": [SystemMessage(content=response, name="action_expert")],
        "next_node": "supervisor"
    }

async def slot_filler_node(state: AgentState):
    print("🗣️ [Slot Filler] 发现信息缺失，正在追问...")
    # 模拟追问逻辑
    response = "好的，请问您早餐具体吃了什么？大概分量是多少？"
    return {
        "messages": [SystemMessage(content=response, name="slot_filler")],
        "next_node": "FINISH" # 抛出反问给用户，直接结束本次后端流转
    }

async def memory_compressor_node(state:AgentState):
    messages = state["messages"]

    # 压缩阈值
    if len(messages) <= 10:
        return {"next_node","supervisor"}
    
    print(f" [Memory Compressor] 警报：上下文已达 {len(messages)} 条，触发滑动窗口压缩！")

    # 压缩范围
    old_messages = messages[:-2] # 最后两条，最新两条

    # 调用大模型浓缩
    summary_prompt = "请用 50 个字总结以下对话的核心健康线索，不要遗漏用户的疾病和过敏史："
    old_messages_text = "\n".join([f"{m.type}: {m.content}" for m in old_messages])
    
    summary_decision = await llm.ainvoke(
        [SystemMessage(content=summary_prompt),HumanMessage(content=old_messages_text)]
    )

    # 提取现有画像融入
    current_profile = state.get("user_profile",{})
    current_profile["历史档案"] = summary_decision.content

    # 删除就对话
    delete_ops = [RemoveMessage(id=m.id) for m in old_messages]
    print(f"✅ [压缩完成] 删除了 {len(delete_ops)} 条废话，提取了核心画像：{summary_decision.content}")
    
    # 将删除指令和更新后的画像写回黑板，控制权交还给Supervisor
    return {
        "messages": delete_ops, 
        "user_profile": current_profile,
        "next_node": "supervisor"
    }

# =====================================
#    5. draw graph
# =====================================
workflow = StateGraph(AgentState)

# register all node
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("rag_expert", rag_expert_node)
workflow.add_node("action_expert", action_expert_node)
workflow.add_node("slot_filler", slot_filler_node)
workflow.add_node("memory_compressor",memory_compressor_node)

# 其实结点
workflow.add_edge(START,"supervisor")

# conditional_edges
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"],
    {
        "rag_expert": "rag_expert",
        "action_expert": "action_expert",
        "slot_filler": "slot_filler",
        "FINISH": END
    }
)

# 当子结点运作完回到主节点
workflow.add_edge("rag_expert","memory_compressor")
workflow.add_edge("action_expert","memory_compressor")

workflow.add_edge("memory_compressor","supervisor")

# compile
app_graph = workflow.compile()
print("✅ [系统初始化] 多智能体神经网络编译完成！")