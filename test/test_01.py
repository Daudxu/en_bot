import os
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# --- 加载环境变量 ---
load_dotenv()

# --- 从环境变量获取 LLM 配置 ---
base_url = os.getenv("BASE_URL")
model_api_key = os.getenv("MODEL_API_KEY")
model_name = os.getenv("MODEL_name")

# --- 初始化 LangChain ChatOpenAI 模型 ---
llm = ChatOpenAI(
    base_url=base_url,
    api_key=model_api_key,
    model=model_name,
    temperature=0.1,
    max_tokens=512,
    streaming=True
)

# --- 1. 定义图的状态 ---
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 2. 定义节点函数 ---

def llm_node_chat(state: ChatState) -> ChatState:
    """
    节点：调用 LLM 生成回复。
    """
    print("\n--- 进入 llm_node ---")
    current_messages = state["messages"]
    print(f"LLM 节点收到 {len(current_messages)} 条消息。最新消息: {current_messages[-1].content[:100]}...")

    ai_response = llm.invoke(current_messages)
    print(f"LLM 回复 (部分): {ai_response.content[:100]}...")
    
    return {"messages": [ai_response]} # 明确返回状态更新


def display_node_chat(state: ChatState) -> ChatState:
    """
    节点：显示 LLM 的回复给用户。
    """
    print("\n--- 进入 display_node ---")
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage):
        print(f"AI: {last_message.content}")
    else:
        print(f"警告: display_node 收到意外的消息类型: {type(last_message).__name__}")
        print(f"消息内容: {last_message.content}")
    return {} # 不更新状态


# --- 2.5 新增一个节点来执行退出检查 (它是个常规节点，返回 {} 或状态更新) ---
def check_quit_node(state: ChatState) -> ChatState:
    """
    一个常规节点，用于打印决策信息，但不直接做路由。
    路由由后续的 conditional_edges 决定。
    """
    print("\n--- 进入 check_quit_node (常规决策检查节点) ---")
    # 这里不需要返回任何状态更新，因为我们只是检查消息
    return {}


# --- 2.6 定义一个纯粹的路由函数 (只返回字符串) ---
def route_decision(state: ChatState) -> str:
    """
    纯粹的路由函数，只用于 conditional_edges，不作为节点。
    """
    print("\n--- 路由决策函数 route_decision ---")
    current_messages = state["messages"]

    last_human_message = None
    for msg in reversed(current_messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    if last_human_message and last_human_message.content.lower() in ["退出", "exit"]:
        print("路由：用户选择退出。")
        return "END"
    else:
        print("路由：继续对话到 LLM。")
        return "llm_node"


# --- 3. 构建图 (使用 StateGraph) ---
workflow = StateGraph(ChatState)

# 添加节点
workflow.add_node("llm_node", llm_node_chat)
workflow.add_node("display_node", display_node_chat)
workflow.add_node("check_quit_node", check_quit_node) # 添加新的检查节点

# --- 4. 定义边的连接方式 ---
# 设置入口点：从检查退出节点开始
workflow.set_entry_point("check_quit_node")

# 从 check_quit_node 离开时，使用 route_decision 来做条件路由
workflow.add_conditional_edges(
    "check_quit_node", # 这是执行路由的节点
    route_decision,    # 这是纯粹的路由函数，它返回字符串
    {
        "llm_node": "llm_node",
        "END": END
    }
)

# 从 LLM 响应到展示回复
workflow.add_edge("llm_node", "display_node")

# 从 display_node 直接到 END，确保每次 invoke 结束
workflow.add_edge("display_node", END)


# --- 5. 编译并运行图 ---
app = workflow.compile()

print("\n--- 启动 LangGraph 聊天机器人 (使用 StateGraph 修正版) ---")
print("输入 '退出' 或 'exit' 结束对话。")

# --- 维护并传递状态的外部循环 ---
current_chat_history = {"messages": []} # 初始时消息列表为空

while True:
    try:
        # 1. 在外部循环中获取用户输入
        user_input_content = input("您: ")

        # 2. 如果用户输入了退出指令，立即退出循环
        if user_input_content.lower() in ["退出", "exit"]:
            print("\n对话已通过外部指令结束。再见！")
            break

        # 3. 将用户的当前输入添加到历史记录中
        current_chat_history["messages"].append(HumanMessage(content=user_input_content))

        # 打印当前 ChatState messages 结构
        print("\n[DEBUG] 当前 ChatState messages:")
        for idx, msg in enumerate(current_chat_history["messages"]):
            print(f"  {idx}: type={type(msg).__name__}, content={getattr(msg, 'content', None)}")

        # 4. 调用 LangGraph 图来处理这一轮对话
        # 传入完整的状态字典。图将从 "check_quit_node" 开始。
        result_state = app.invoke(current_chat_history)

        # 5. 更新外部的对话历史字典，以供下一轮使用
        current_chat_history = result_state

    except KeyboardInterrupt:
        print("\n对话中断。再见！")
        break
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        break

print("\n--- 聊天机器人结束 ---")