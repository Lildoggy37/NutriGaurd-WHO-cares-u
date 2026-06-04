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

from memory import (
    estimate_tokens, split_messages, format_messages_for_llm,
    build_summary_prompt, build_extraction_prompt, find_last_complete_ai,
    WORKING_MEMORY_TOKENS,
    save_long_term_memory, load_long_term_memory,
)



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
        next_node: Literal["rag_expert", "action_expert", "slot_filler", "FINISH"] = Field(
        alias="route"   # 同时接受 "route" 字段名
        )
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

    ROUTE_PROMPT = """你是医疗健康路由分发器。请先逐步分析，再输出 JSON 路由决策。

    【分析步骤】请依次思考：
    1. 用户最后一句话的核心诉求是什么？（一句话概括）
    2. 这属于哪类意图？知识查询(rag) / 操作执行(action) / 信息不全(slot) / 对话结束(FINISH)？
    3. 如果选 FINISH，确认理由是什么？如果选其他节点，用户是否已提供足够信息？
    4. 最终路由决定：

    【路由规则】
    - 用户询问疾病禁忌、指南知识、营养成分 → rag_expert
    - 用户想记录饮食、计算热量、更新健康信息、生成采购清单 → action_expert
    - 用户的话信息不全 → slot_filler
    - 以下情况选 FINISH：
    * AI 刚完成一次操作且已确认结果，对话末尾无新的用户消息
    * AI 回复是确认，用户在等待下一步指令
    * 用户明确表示满意或说再见

    重要：不要根据 AI 回复中的"帮您记录""帮您查看""推荐"等建议词判断路由。只有用户显式说出操作意图时才路由。

    【输出格式】先输出分析，再输出一行 JSON，不要 markdown 代码块：
    （分析文字...）
    {"route": "rag_expert", "reason": "理由"}

    route 只能是四个值之一：rag_expert / action_expert / slot_filler / FINISH"""

    async def supervisor_node(state: AgentState):
        print("[Supervisor] 正在审视意图...")

        messages = state["messages"]
        last_msg = messages[-1] if messages else None

        # 压缩后终止：检测 memory_summary → 直接结束当前轮次
        if last_msg and getattr(last_msg, "name", "") == "memory_summary":
            print("[Supervisor] 检测到压缩摘要消息，本轮结束")
            return {"next_node": "FINISH"}

        # 终止信号：最后一条是完整的 AI 回答（无 tool_calls）→ 结束
        last_ai = None
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                if not getattr(m, "tool_calls", None):
                    last_ai = m
                    break
        if last_ai and not any(
            isinstance(m, HumanMessage) for m in messages[-3:]
        ):
            print("[Supervisor] AI 已回答且无新用户消息，本轮结束")
            return {"next_node": "FINISH"}

        # 过滤 ToolMessage / tool_calls，构造干净的 LLM 输入
        clean = []
        for m in messages:
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue
            if isinstance(m, AIMessage) and clean and isinstance(clean[-1], AIMessage):
                continue
            clean.append(m)

        profile_str = json.dumps(state.get("user_profile", {}), ensure_ascii=False)
        sys_prompt = ROUTE_PROMPT + f"\n\n当前用户已知健康画像：{profile_str}"
        llm_input = [SystemMessage(content=sys_prompt)] + clean

        try:
            response = await llm.ainvoke(llm_input)
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
                recovery = SystemMessage(
                    content=f"对话已恢复。根据之前的记录：{hist}\n\n请问还有什么可以帮到您的？"
                )
            else:
                recovery = SystemMessage(content="我准备好了，请问有什么可以帮您的？")
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
    #    4.5 Preprocess —— 查询预处理（纠错+改写+意图澄清）
    # =====================================
    PREPROCESS_PROMPT = """你是医疗健康查询预处理助手。对用户的原始输入做以下处理：

    1. **纠错**：修正常见错别字（如"唐尿病"→"糖尿病"，"同风"→"痛风"，"升糖"→"升糖指数"）
    2. **同义词展开**：口语化表达转为标准术语（如"尿酸高"→"高尿酸血症/痛风"，"三高"→"高血压/高血脂/高血糖"）
    3. **指代消解**：如果用户说"这个""那个"，结合上下文明确指代对象
    4. **意图澄清**：模糊查询补充关键信息（如"吃啥好"→"适合食用什么类型的食物"）

    **噪声控制规则**：
    - 不要添加用户没提到的疾病、食物或个人信息
    - 不要猜测用户的健康状况
    - 改写后长度控制在 80 字以内
    - 忠实于用户原意，只做标准化不改含义

    输出格式：只输出改写后的问题文本，不要附加任何解释或 JSON。"""

    async def preprocess_node(state: AgentState):
        messages = state["messages"]
        # 找到最后一条 HumanMessage
        last_user_msg = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_user_msg = m
                break

        if last_user_msg is None:
            return {"next_node": "supervisor"}

        raw = str(last_user_msg.content).strip()
        # 太短或明显不需要改写 → 跳过
        if len(raw) <= 3:
            return {"next_node": "supervisor"}

        print(f"[Preprocess] 原始输入: {raw[:60]}...", flush=True)
        try:
            response = await llm.ainvoke([
                SystemMessage(content=PREPROCESS_PROMPT),
                HumanMessage(content=f"用户原始输入：{raw}"),
            ])
            rewritten = str(response.content).strip()

            if rewritten and rewritten != raw:
                print(f"[Preprocess] 改写结果: {rewritten[:80]}...", flush=True)
                return {
                    "messages": [
                        SystemMessage(content=f"[查询改写] {rewritten}", name="preprocess")
                    ],
                    "next_node": "supervisor",
                }
        except Exception as e:
            print(f"[Preprocess] 改写失败: {e}，直接放行", flush=True)

        return {"next_node": "supervisor"}

    # =====================================
    #    4.6 Reflection —— RAG 回答合规审查
    # =====================================
    class ReflectionVerdict(BaseModel):
        verdict: Literal["PASS", "CORRECT", "REJECT"]
        reason: str = ""
        risk_items: str | list[str] = ""

        @staticmethod
        def _normalize_risk(v: str | list[str]) -> str:
            if isinstance(v, list):
                return "; ".join(str(x) for x in v)
            return str(v) if v else ""

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
            # 规范化 risk_items（qwen 可能返回列表）
            verdict.risk_items = ReflectionVerdict._normalize_risk(verdict.risk_items)
            print(f"[Reflection] 判定: {verdict.verdict} | {verdict.reason}")

            if verdict.verdict == "PASS":
                return {"next_node": "supervisor"}

            elif verdict.verdict == "CORRECT":
                note = SystemMessage(
                    content=(
                        f"[合规审查·修正] {verdict.reason}"
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
                        "【系统提示】经合规审查，上一条回答存在以下问题，已被拦截：\n"
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
            print(f"[Reflection 审查异常] {e}，降级结束本轮对话", flush=True)
            return {"next_node": "FINISH"}


    # =====================================
    #    4.5 memory_compressor 记忆压缩结点
    # =====================================

    # =====================================
    #    三层记忆压缩（token 阈值替代消息数）
    # =====================================
    async def memory_compressor_node(state: AgentState):
        messages = state["messages"]
        token_estimate = estimate_tokens(messages)

        if token_estimate <= WORKING_MEMORY_TOKENS:
            return {"next_node": "FINISH"}

        print(
            f"[Memory] 上下文 {len(messages)} 条 / ~{token_estimate} tokens，触发压缩"
        )

        user_id = state.get("user_profile", {}).get("用户标识", "unknown")

        # ----- Layer 分离 -----
        layer_recent, layer_middle, layer_old = split_messages(messages)

        if not layer_middle and not layer_old:
            return {"next_node": "FINISH"}

        # ----- 保留 FINISH 信号 -----
        finish_signal = None
        complete_idx = find_last_complete_ai(messages)
        if complete_idx >= 0:
            finish_signal = messages[complete_idx]

        # ----- 确保最后 HumanMessage 不被删 -----
        last_human_content = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human_content = str(m.content)
                break

        try:
            current_profile = state.get("user_profile", {})

            # --- Layer 3: 早期对话 -> 长期记忆 ---
            if layer_old:
                old_text = format_messages_for_llm(layer_old)
                extraction_prompt = build_extraction_prompt(old_text)
                try:
                    extraction_response = await llm.ainvoke([
                        SystemMessage(content=extraction_prompt),
                    ])
                    facts_raw = _extract_json(str(extraction_response.content))
                    facts = json.loads(facts_raw)
                    save_long_term_memory(user_id, facts)
                    print(f"[Memory] Layer 3: 提取长期记忆 {len(facts)} 条")
                except Exception as e:
                    print(f"[Memory] Layer 3 提取失败: {e}")

            # --- Layer 2: 中间对话 -> 摘要 ---
            summary_text = ""
            if layer_middle:
                middle_text = format_messages_for_llm(layer_middle)
                summary_prompt = build_summary_prompt(middle_text)
                try:
                    summary_response = await llm.ainvoke([
                        SystemMessage(content=summary_prompt),
                    ])
                    summary_text = str(summary_response.content)[:200]
                    current_profile["对话摘要"] = summary_text
                    print(f"[Memory] Layer 2: 压缩 {len(layer_middle)} 条 -> 摘要")
                except Exception as e:
                    print(f"[Memory] Layer 2 压缩失败: {e}")

            # --- Layer 1: 最近 N 轮保留 + 摘要 + 长期记忆注入 ---
            ltm = load_long_term_memory(user_id)
            if ltm:
                current_profile["长期记忆"] = json.dumps(ltm, ensure_ascii=False)

            # 构建压缩后的消息列表
            delete_ids = set()
            for m in layer_old:
                if getattr(m, "id", None):
                    delete_ids.add(m.id)
            for m in layer_middle:
                if getattr(m, "id", None):
                    delete_ids.add(m.id)

            delete_ops = [RemoveMessage(id=mid) for mid in delete_ids if mid]

            new_messages = []
            if summary_text:
                new_messages.append(
                    SystemMessage(
                        content=f"[对话摘要] {summary_text}",
                        name="memory_summary",
                    )
                )

            # 确保最后 HumanMessage 在保留区内
            has_human = any(isinstance(m, HumanMessage) for m in layer_recent)
            if not has_human and last_human_content:
                new_messages.append(HumanMessage(content=last_human_content))

            # 确保 FINISH 信号保留
            has_finish = any(
                isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
                for m in layer_recent
            )
            if not has_finish and finish_signal:
                new_messages.append(
                    AIMessage(content=str(finish_signal.content))
                )

            print(
                f"[Memory] 压缩完成: 删除 {len(delete_ids)} 条，"
                f"保留 {len(layer_recent)} 条 (Layer1) + {len(new_messages)} 条元数据"
            )

            return {
                "messages": delete_ops + new_messages,
                "user_profile": current_profile,
                "next_node": "FINISH",
            }
        except Exception as e:
            print(f"[Memory] 压缩异常: {e}，跳过")
            return {"next_node": "FINISH"}
    # =====================================
    #    5. draw graph
    # =====================================
    workflow = StateGraph(AgentState)

    # register all node
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_expert", rag_expert_node)
    workflow.add_node("rag_reflection", rag_reflection_node)
    workflow.add_node("action_expert", action_expert_node)
    workflow.add_node("slot_filler", slot_filler_node)
    workflow.add_node("memory_compressor", memory_compressor_node)

    # START → preprocess → supervisor
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "supervisor")

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

    # RAG 路径: rag_expert → rag_reflection → END（不绕回 supervisor 避免膨胀→压缩→死循环）
    workflow.add_edge("rag_expert", "rag_reflection")
    workflow.add_edge("rag_reflection", END)

    # Action 路径: action_expert → memory_compressor → supervisor
    workflow.add_edge("action_expert", "memory_compressor")

    workflow.add_conditional_edges(
    "memory_compressor",
    lambda state: state["next_node"],
    {
        "supervisor": "supervisor",
        "FINISH": END,
    },
)

    if checkpointer is None:
        checkpointer = MemorySaver()
    # compile
    app_graph = workflow.compile(checkpointer=checkpointer)
    print("[系统初始化] 多智能体神经网络编译完成！")
    return app_graph