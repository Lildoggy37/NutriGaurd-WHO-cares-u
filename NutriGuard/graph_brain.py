import operator
import json
import os
from typing import Annotated, Sequence, TypedDict, Literal, Dict
from pydantic import BaseModel,Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import create_react_agent
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
def build_multi_agent_graph(rag_tools:list,action_tools:list):
    """
    控制反转 (IoC)：图引擎不再负责连接 MCP，而是接收外部传入的 tools 进行动态编译。
    这彻底解决了顶层模块同步加载与 MCP 异步长连接的死锁问题。
    """
    
    llm = ChatOpenAI(
        model="qwen-plus",
        temperature=0.0,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    rag_agent = create_react_agent(
        model=llm,
        tools=rag_tools,
        state_modifier="你是一个顶级的营养学与 RAG 检索专家。请优先调用工具查询专业数据，绝不要瞎编医学常识。回答要客观严谨。"
    )

    action_agent = create_react_agent(
        model=llm,
        tools=action_tools,
        state_modifier="你是一个极其严谨的健康管家。你的职责是调用工具记录饮食和热量。如果用户提供的信息不够，你可以反问。"
    )

    # =====================================
    #    3. Supervisor
    # =====================================
    class RouteDecision(BaseModel):
        next_node: Literal["rag_expert", "action_expert", "slot_filler", "FINISH"] = Field(description="下一步流转节点")
        reason: str = Field(description="路由决策的原因") # 大模型思考过程

    supervisor_llm = llm.with_structured_output(RouteDecision)

    async def supervisor_node(state:AgentState):
        print("[Supervisor] 正在审视意图...")
        # 将user_profile转换为字符串给llm
        profile_str = json.dumps(state.get("user_profile",{}),ensure_ascii=False)

        sys_prompt = f"""你是一个高级医疗健康分发路由。
        当前用户的已知健康画像：{profile_str}
        
        请根据历史对话，决定下一步操作：
        - 如果用户在询问疾病禁忌、指南知识，选 'rag_expert'
        - 如果用户想记录今天吃了什么、算热量，选 'action_expert'
        - 如果用户的话没说完（例如说“帮我记下早饭”但没说吃了啥），选 'slot_filler'
        - 如果问题已经彻底解答完毕，选 'FINISH'
        """

        messages = [SystemMessage(content=sys_prompt)] + state["messages"]

        try:
            decision = await supervisor_llm.ainvoke(messages)
            print(f"🚦 [路由分发] 决定去向: {decision.next_node} | 理由: {decision.reason}")
            return {"next_node": decision.next_node}
        except Exception as e:
            print(f"🧨 [Supervisor 崩溃兜底] 路由解析失败: {e}。触发安全降级机制。")
            return {"next_node": "FINISH"} # 路由失败时强制切断，避免在图中死循环

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

    memory = MemorySaver()
    # compile
    app_graph = workflow.compile(checkpointer=memory)
    print("✅ [系统初始化] 多智能体神经网络编译完成！")
    return app_graph