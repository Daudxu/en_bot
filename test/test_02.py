import os
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # 导入 SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # 导入提示模板相关类

# --- 加载环境变量 ---
load_dotenv()

# --- 从环境变量获取 LLM 配置 ---
base_url = os.getenv("BASE_URL")
model_api_key = os.getenv("MODEL_API_KEY")
model_name = os.getenv("MODEL_NAME")

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

# --- 【新增】定义带有系统提示的 LLM Chain ---
# 定义系统提示词
SYSTEM_PROMPT = (
    "你是一个友好的AI助手，你的名字叫司马青，今年32岁。"
    "你会尽量回答用户的问题，提供有用的信息。"
    "请保持对话的连贯性，并尽量提供准确的信息。 "
    "如果被问到超出你能力范围的问题，请礼貌地告知。"
)

# 构建 ChatPromptTemplate
# SystemMessage: 定义系统级别指令
# MessagesPlaceholder: 表示对话历史的占位符，它会接收一个消息列表
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages") # 这里的 "messages" 对应 llm_chain.invoke(messages=...)
])

# 将提示模板和 LLM 模型组合成一个链
llm_chain = prompt | llm

def llm_node_chat(state: ChatState) -> ChatState:
    """
    节点：调用 LLM 生成回复。
    """
    print("\n--- 进入 llm_node ---")
    current_messages = state["messages"]
    print(f"LLM 节点收到 {len(current_messages)} 条消息。最新消息: {current_messages[-1].content[:100]}...")

    # *** 关键变化：现在调用的是 llm_chain，并将消息历史传入 'messages' 变量 ***
    ai_response = llm_chain.invoke({"messages": current_messages}) 
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


def check_quit_node(state: ChatState) -> ChatState:
    """
    一个常规节点，用于打印决策信息，但不直接做路由。
    路由由后续的 conditional_edges 决定。
    """
    print("\n--- 进入 check_quit_node (常规决策检查节点) ---")
    return {}


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
workflow.add_node("check_quit_node", check_quit_node)

# --- 4. 定义边的连接方式 ---
workflow.set_entry_point("check_quit_node")

workflow.add_conditional_edges(
    "check_quit_node", 
    route_decision,    
    {
        "llm_node": "llm_node",
        "END": END
    }
)

workflow.add_edge("llm_node", "display_node")

workflow.add_edge("display_node", END)


# --- 5. 编译并运行图 ---
app = workflow.compile()

print("\n--- 启动 LangGraph 聊天机器人 (使用 StateGraph 修正版 - 整合提示词) ---")
print("输入 '退出' 或 'exit' 结束对话。")

current_chat_history = {"messages": []}

while True:
    try:
        user_input_content = input("您: ")

        if user_input_content.lower() in ["退出", "exit"]:
            print("\n对话已通过外部指令结束。再见！")
            break

        current_chat_history["messages"].append(HumanMessage(content=user_input_content))

        # 调试打印
        print("\n[DEBUG] 当前 ChatState messages (准备传入 LangGraph):")
        for idx, msg in enumerate(current_chat_history["messages"]):
            print(f"  {idx}: type={type(msg).__name__}, content={getattr(msg, 'content', None)}")

        result_state = app.invoke(current_chat_history)

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