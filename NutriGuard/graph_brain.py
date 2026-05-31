import operator
import json
import os
import re
from typing import Annotated, Sequence, TypedDict, Literal, Dict
from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage, HumanMessage, SystemMessage, RemoveMessage, AIMessage, ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver



# =====================================
#    1. 全局状态机
# =====================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node:str
    user_profile: Dict[str,str]


# =====================================
#    2. 图引擎工厂 解决MCP工具加载问题
# =====================================
def build_multi_agent_graph(rag_tools:list, action_tools:list, checkpointer=None):
    """
    控制反转 (IoC)：图引擎不再负责连接 MCP，而是接收外部传入的 tools 进行动态编译。
    这彻底解决了顶层模块同步加载与 MCP 异步长连接的死锁问题。

    checkpointer: 可选，默认使用内存检查点。测试时可传入 None 关闭持久化。
    """
    
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    rag_agent = create_agent(
        model=llm,
        tools=rag_tools,
        system_prompt="你是一个顶级的营养学与 RAG 检索专家。请优先调用工具查询专业数据，绝不要瞎编医学常识。回答要客观严谨。"
    )

    action_agent = create_agent(
        model=llm,
        tools=action_tools,
        system_prompt="你是一个极其严谨的健康管家。你的职责是调用工具记录饮食和热量。如果用户提供的信息不够，你可以反问。"
    )

    # =====================================
    #    3. Supervisor
    # =====================================
    class RouteDecision(BaseModel):
        next_node: Literal["rag_expert", "action_expert", "slot_filler", "FINISH"]
        reason: str = ""

    # JSON 提取工具：qwen 通过 compatible-mode 端点不保证 structured_output 稳定，
    # 改用 prompt 要求 JSON → 正则提取 → Pydantic 校验
    def _extract_json(text: str) -> str:
        m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if m:
            return m.group(1).strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            return m.group(0).strip()
        return text.strip()

    ROUTE_PROMPT = """你是一个高级医疗健康分发路由。请根据对话历史决定下一步操作，并输出 JSON。

    路由规则：
    - 用户在询问疾病禁忌、指南知识 → rag_expert
    - 用户想记录饮食、计算热量、生成采购清单 → action_expert
    - 用户的话信息不全（如只说"记早饭"但没说什么食物）→ slot_filler
    - 问题已经彻底解答完毕、或简单闲聊问候 → FINISH

    请严格按以下 JSON 格式输出，不要附加任何其他文字：
    {"next_node": "<rag_expert|action_expert|slot_filler|FINISH>", "reason": "你的路由理由"}"""

    async def supervisor_node(state: AgentState):
        print("[Supervisor] 正在审视意图...")
        profile_str = json.dumps(state.get("user_profile", {}), ensure_ascii=False)

        sys_prompt = ROUTE_PROMPT + f"\n\n当前用户已知健康画像：{profile_str}"

        messages = [SystemMessage(content=sys_prompt)] + state["messages"]

        try:
            response = await llm.ainvoke(messages)
            json_str = _extract_json(str(response.content))
            decision = RouteDecision.model_validate_json(json_str)
            print(f"[路由分发] 决定去向: {decision.next_node} | 理由: {decision.reason}")
            return {"next_node": decision.next_node}
        except Exception as e:
            print(f"[Supervisor 兜底] 路由解析失败: {e}，强制 FINISH")
            return {"next_node": "FINISH"}

    # =====================================
    #    4. worker node
    # =====================================

    async def rag_expert_node(state: AgentState):
        print("📖 [RAG Expert] 正在思考并调用工具...")
        try:
            result = await rag_agent.ainvoke(state)
            return {"messages": result["messages"], "next_node": "supervisor"}
        except Exception as e:
            print(f"🧨 [RAG 节点异常] {e}")
            error_msg = SystemMessage(content="【系统提示】抱歉，医学知识库当前响应超时，请稍后再试或换个问法。", name="rag_expert")
            return {"messages": [error_msg], "next_node": "supervisor"}

    async def action_expert_node(state: AgentState):
        print("🛠️ [Action Expert] 正在调用后台系统...")
        try:
            result = await action_agent.ainvoke(state)
            return {"messages": result["messages"], "next_node": "supervisor"}
        except Exception as e:
            print(f"🧨 [Action 节点异常] {e}")
            error_msg = SystemMessage(content="【系统提示】抱歉，膳食日志系统暂时不可用，记录失败。", name="action_expert")
            return {"messages": [error_msg], "next_node": "supervisor"}

    async def slot_filler_node(state: AgentState):
        print("🗣️ [Slot Filler] 发现信息缺失，正在追问...")
        response = "为了给您更精准的建议，请问您刚才提到的食物具体分量是多少呢？（例如：一小碗、200g）"
        return {"messages": [SystemMessage(content=response, name="slot_filler")], "next_node": "FINISH"}

    # =====================================
    #    4.5 Reflection —— RAG 回答合规审查
    # =====================================
    class ReflectionVerdict(BaseModel):
        verdict: Literal["PASS", "CORRECT", "REJECT"]
        reason: str = ""
        risk_items: str = ""

    async def rag_reflection_node(state: AgentState):
        """
        RAG 回答的事后合规审查。
        从消息历史中提取检索证据和 AI 回答，用独立 LLM 审查幻觉/安全/完整性。
        """
        print("[Reflection] 正在审查 RAG 回答的合规性...")
        messages = state["messages"]

        # --- 找到最后一条 AI 回答 + 所有工具检索结果 ---
        last_ai = None
        evidence_parts = []

        for m in messages:
            if isinstance(m, AIMessage):
                tc = getattr(m, "tool_calls", None)
                if not tc:
                    last_ai = m
            if isinstance(m, ToolMessage):
                evidence_parts.append(str(m.content)[:800])

        if last_ai is None:
            print("[Reflection] 未找到 AI 回答，跳过审查", flush=True)
            return {"next_node": "supervisor"}

        answer_text = str(last_ai.content)[:2000]
        evidence = "\n---\n".join(evidence_parts[-5:]) or "（本次未检索到外部资料，回答完全依赖模型自身知识）"

        review_prompt = f"""你是医疗健康内容合规审查员。请严格审查以下 AI 营养学回答。

        【证据——检索到的权威资料】
        {evidence[:3000]}

        【AI 生成的回答】
        {answer_text}

        【审查标准】
        1. 幻觉检测：回答中每条营养/医学声明是否在证据中有出处？引用是否准确？
        2. 安全合规：是否包含危险饮食建议？是否缺少「仅供参考，请咨询医生」类声明？
        3. 完整性：是否正面、直接地回答了用户问题？

        【判定规则】
        - PASS：安全、准确、完整，直接放行
        - CORRECT：有小瑕疵，附修正建议后放行
        - REJECT：存在虚构数据、危险建议、或与证据严重矛盾，必须拦截

        请严格按以下 JSON 格式输出，不要附加任何其他文字：
        {{"verdict": "<PASS|CORRECT|REJECT>", "reason": "判定理由", "risk_items": "风险点（若无则留空）"}}"""

        try:
            response = await llm.ainvoke([SystemMessage(content=review_prompt)])
            json_str = _extract_json(str(response.content))
            verdict = ReflectionVerdict.model_validate_json(json_str)
            print(f" [Reflection] 判定: {verdict.verdict} | {verdict.reason}")

            if verdict.verdict == "PASS":
                return {"next_node": "supervisor"}

            elif verdict.verdict == "CORRECT":
                note = SystemMessage(
                    content=(
                        f" [合规审查·修正] {verdict.reason}"
                        + (f"\n风险提示: {verdict.risk_items}" if verdict.risk_items else "")
                    ),
                    name="reflection",
                )
                return {"messages": [note], "next_node": "supervisor"}

            else:  # REJECT
                delete_ops = (
                    [RemoveMessage(id=last_ai.id)]
                    if getattr(last_ai, "id", None) else []
                )
                safe_msg = SystemMessage(
                    content=(
                        " 【系统提示】经合规审查，上一条回答存在以下问题，已被拦截：\n"
                        f"  — {verdict.reason}\n"
                        + (f"  — 风险项: {verdict.risk_items}\n" if verdict.risk_items else "")
                        + "\n建议您：\n"
                        "  1. 咨询执业医师或注册营养师获取个性化建议\n"
                        "  2. 换一种更具体的问法重新提问\n"
                        "  3. 参考权威机构（如中国营养学会）发布的官方指南"
                    ),
                    name="reflection",
                )
                return {"messages": delete_ops + [safe_msg], "next_node": "supervisor"}

        except Exception as e:
            print(f" [Reflection 审查异常] {e}，降级放行", flush=True)
            return {"next_node": "supervisor"}


    # =====================================
    #    4.5 memory_compressor 记忆压缩结点
    # =====================================

    # 并发安全的记忆管理
    async def memory_compressor_node(state:AgentState):
        messages = state["messages"]

        # 压缩阈值
        if len(messages) <= 10:
            return {"next_node":"supervisor"}
        
        print(f" [Memory Compressor] 警报：上下文已达 {len(messages)} 条，触发滑动窗口压缩！")

        # 压缩范围
        old_messages = messages[:-2] # 最后两条，最新两条

        try:
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
        except Exception as e:
            print(f" [压缩节点异常] {e}。跳过本次压缩。")
            return {"next_node": "supervisor"}

    # =====================================
    #    5. draw graph
    # =====================================
    workflow = StateGraph(AgentState)

    # register all node
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_expert", rag_expert_node)
    workflow.add_node("rag_reflection", rag_reflection_node)
    workflow.add_node("action_expert", action_expert_node)
    workflow.add_node("slot_filler", slot_filler_node)
    workflow.add_node("memory_compressor", memory_compressor_node)

    # 其实结点
    workflow.add_edge(START, "supervisor")

    # conditional_edges
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_node"],
        {
            "rag_expert": "rag_expert",
            "action_expert": "action_expert",
            "slot_filler": "slot_filler",
            "FINISH": END,
        },
    )

    # RAG 路径: rag_expert → rag_reflection → memory_compressor → supervisor
    workflow.add_edge("rag_expert", "rag_reflection")
    workflow.add_edge("rag_reflection", "memory_compressor")

    # Action 路径: action_expert → memory_compressor → supervisor
    workflow.add_edge("action_expert", "memory_compressor")

    workflow.add_edge("memory_compressor", "supervisor")

    if checkpointer is None:
        checkpointer = MemorySaver()
    # compile
    app_graph = workflow.compile(checkpointer=checkpointer)
    print("[系统初始化] 多智能体神经网络编译完成！")
    return app_graph