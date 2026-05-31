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
        system_prompt="你是一个营养学与 RAG 检索专家。优先调用工具查询数据，不要编造医学常识。回答客观严谨，解答完即停止，不要主动追问或提出新话题。"
    )

    action_agent = create_agent(
        model=llm,
        tools=action_tools,
        system_prompt="你是一个严谨的健康管家。调用工具完成用户请求的操作后，简洁确认即可。不要主动追问或提出新话题，让用户自行决定下一步。"
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

    ROUTE_PROMPT = """你是医疗健康路由分发器。只根据最近一条用户消息（HumanMessage）判断路由，忽略 AI 消息中的建议和追问。

    路由规则：
    - 用户在询问疾病禁忌、指南知识、营养成分 → rag_expert
    - 用户想记录饮食、计算热量、更新健康信息、生成采购清单 → action_expert
    - 用户的话信息不全 → slot_filler
    - 以下情况选 FINISH：
      * AI 刚完成一次操作且已确认结果，用户在等待下一步指令
      * AI 的回复是确认/追问/建议，且对话末尾没有新的用户消息
      * 用户明确表示满意或说再见

    重要：不要根据 AI 回复中出现的"帮您记录""帮您查看""推荐"等字样判断为需要进一步操作。这些是 AI 的提问，不是用户的指令。只有当用户显式说出操作意图时才路由到对应节点。"""

    async def supervisor_node(state: AgentState):
        print("[Supervisor] 正在审视意图...")

        # 只取最近的用户消息 + AI 回复，避免旧消息干扰
        recent = []
        for m in reversed(state["messages"]):
            recent.insert(0, {"type": m.type, "content": str(m.content)[:200]})
            if len(recent) >= 4:
                break

        profile_str = json.dumps(state.get("user_profile", {}), ensure_ascii=False)
        context = "\n".join([f"[{r['type']}] {r['content']}" for r in recent])

        sys_prompt = (
            ROUTE_PROMPT
            + f"\n当前健康画像：{profile_str}"
            + f"\n最近对话：\n{context}"
            + "\n\n根据以上信息，输出 JSON 路由决策。"
        )

        messages = [SystemMessage(content=sys_prompt)]   # 不带 state messages，只用摘要

        try:
            response = await llm.ainvoke(messages)
            raw = str(response.content) if response.content else ""
            if not raw.strip():
                raise ValueError("LLM returned empty content")
            json_str = _extract_json(raw)
            decision = RouteDecision.model_validate_json(json_str)
            print(f"[路由分发] {decision.next_node} | {decision.reason}")
            return {"next_node": decision.next_node}
        except Exception as e:
            print(f"[Supervisor 兜底] 路由失败: {e}")
            hist = state.get("user_profile", {}).get("历史档案", "")
            if hist:
                recovery = AIMessage(
                    content=f"对话已恢复。根据之前的记录：{hist}\n\n请问还有什么可以帮到您的？"
                )
            else:
                recovery = AIMessage(content="我准备好了，请问有什么可以帮您的？")
            return {"messages": [recovery], "next_node": "FINISH"}

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

    SLOT_FILLER_PROMPT = """你是一个细心的健康管家。用户刚才说的话信息不完整，你需要追问以补全关键信息。

    追问规则：
    - 根据对话历史，判断用户缺失了什么关键信息（食物分量、食物种类、个人信息等）
    - 提出 1-2 个具体、简洁的追问，帮助后续准确处理
    - 不要重复用户已经说过的信息
    - 用友好的口吻，不要让用户感到被审问

    例如：
    - 用户说"帮我记早饭" → 追问"请问早餐具体吃了什么？每样大概多少分量呢？"
    - 用户说"帮我查一下饮食" → 追问"请问您想了解哪方面的饮食信息？比如特定疾病（糖尿病/痛风）的禁忌，还是日常营养搭配？"
    - 用户说"我身高175" → 追问"好的，请问您的体重是多少呢？另外有已知的健康状况吗（如糖尿病、高血压）？" """

    async def slot_filler_node(state: AgentState):
        print("[Slot Filler] 正在生成追问...", flush=True)
        messages = [SystemMessage(content=SLOT_FILLER_PROMPT)] + state["messages"]
        try:
            response = await llm.ainvoke(messages)
            return {"messages": [response], "next_node": "FINISH"}
        except Exception as e:
            print(f"[Slot Filler 异常] {e}", flush=True)
            fallback = "请提供更多细节，我可以更准确地帮助您。"
            return {"messages": [AIMessage(content=fallback)], "next_node": "FINISH"}

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

        # --- 找到最后一个 HumanMessage，确保保留用户意图 ---
        last_human_idx = 0
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break

        # 保留最后一条用户消息及之后的所有消息 + 保底 8 条
        keep_from = min(last_human_idx, max(0, len(messages) - 8))

        # 清理保留区中对 Supervisor 无意义的噪音（ToolMessage、tool_calls AIMessage）
        kept_raw = messages[keep_from:]
        kept_clean = []
        noise_ids = set()
        for m in kept_raw:
            if isinstance(m, ToolMessage):
                noise_ids.add(m.id)
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                noise_ids.add(m.id)
                continue
            kept_clean.append(m)

        old_messages = messages[:keep_from]

        if len(old_messages) <= 4:
            return {"next_node": "supervisor"}

        try:
            old_text = "\n".join(
                [f"{m.type}: {str(m.content)[:300]}" for m in old_messages]
            )
            summary_prompt = "请用 50 个字总结以下对话的核心健康线索，不要遗漏用户的疾病和过敏史："

            summary_decision = await llm.ainvoke([
                SystemMessage(content=summary_prompt),
                HumanMessage(content=old_text),
            ])

            current_profile = state.get("user_profile", {})
            current_profile["历史档案"] = summary_decision.content

            # 删除旧消息 + 保留区中的噪音消息
            all_delete_ids = {m.id for m in old_messages if m.id} | noise_ids
            delete_ops = [RemoveMessage(id=mid) for mid in all_delete_ids]

            # 用 AIMessage 注入恢复消息（对用户可见且给 supervisor 清晰上下文）
            recovery = AIMessage(
                content=(
                    f"我已经整理了您的对话要点：{summary_decision.content}\n\n"
                    f"有什么我可以继续帮您的吗？"
                )
            )

            print(
                f"[压缩完成] 删除 {len(delete_ops)} 条，"
                f"保留 {len(kept_clean)} 条干净消息，"
                f"摘要: {summary_decision.content}"
            )

            return {
                "messages": delete_ops + [recovery],
                "user_profile": current_profile,
                "next_node": "supervisor",
            }
        except Exception as e:
            print(f"[压缩节点异常] {e}，跳过本次压缩")
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