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

# --- 定义带有变量的系统提示词模板 ---
SYSTEM_PROMPT_TEMPLATE = '''你是一位专业的英语单词学习助手，当前学习单词为“{word}”。
【对话规则】
- 首次进入时，只输出：同学你好，针对单词“{word}”，还有什么想要了解的，我可以为你详细讲解哦~你也可以点击对话框上方的选项来进行提问。不要输出释义、用法、搭配等内容。
- 用户输入的内容如果不是“{word}”，无论是其他英文单词还是其他内容，都只回复：咱们还是专注于“{word}”这个单词吧，你在这个单词上还有什么疑问吗？
- 只有当用户输入“{word}”时，才输出该单词的简明中文释义，并以“你理解这个意思了吗？”结尾。例如：“这个单词的意思是‘男孩’，你理解这个意思了吗？”
- 用户输入“详细用法”时，只输出1~2种常见用法，举例说明，并以“你理解了吗？”结尾，不要输出多余拓展。
- 用户输入“固定搭配”时，只列举常见搭配，举例说明，并以“你记住这个搭配了吗？”结尾。
- 用户输入“词根词缀”时，只说明有无词根词缀，简要解释，并以“现在你理解了吗？”结尾。
- 用户输入“例句”时，只输出1个例句，并以“你能理解这个例句中‘{word}’的用法吗？”结尾。
- 用户输入“选择题”或“出一道选择题”时，只设计一道选择题，并以“请选择A、B或C。你能找出正确答案吗？”结尾。
- 用户输入A/B/C时，只判断正误并回复。

【输出要求】
- 只允许输出纯文本、结构化简明内容，禁止输出任何 markdown、表格、代码块、分点说明、mermaid、emoji、拓展知识、文化背景等。
- 每次回复只聚焦用户当前问题，不要重复输出全部知识点。
- 欢迎语只输出一次，后续不再重复。'''

# --- 1. 定义图的状态 (新增 'word' 字段来存储当前学习的单词) ---
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    word: str # 新增一个字段，用于在整个图中传递当前学习的单词

# --- 2. 定义节点函数 ---

# 构建 ChatPromptTemplate (这里 SystemMessage 的 content 是一个模板字符串)
# 注意：MessagesPlaceholder(variable_name="messages") 仍然用于插入历史对话
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT_TEMPLATE), # SystemMessage 现在接收模板字符串
    MessagesPlaceholder(variable_name="messages"),
])

# 将提示模板和 LLM 模型组合成一个链
llm_chain = prompt | llm

def llm_node_chat(state: ChatState) -> ChatState:
    """
    节点：调用 LLM 生成回复。
    """
    print("\n--- 进入 llm_node ---")
    current_messages = state["messages"]
    current_word = state["word"] # 从状态中获取当前学习的单词

    print(f"LLM 节点收到 {len(current_messages)} 条消息。最新消息: {current_messages[-1].content[:100]}...")
    print(f"当前学习单词: {current_word}")

    # 调用 llm_chain，并将消息历史和 word 变量传入
    ai_response = llm_chain.invoke({"messages": current_messages, "word": current_word}) 
    print(f"LLM 回复 (部分): {ai_response.content[:100]}...")
    
    # *** 关键新增部分：后处理 LLM 的回复，替换其中的 {word} 占位符 ***
    if isinstance(ai_response.content, str):
        processed_content = ai_response.content.replace("{word}", current_word)
        processed_ai_response = AIMessage(content=processed_content)
    else:
        processed_ai_response = ai_response # 保留原样，以防非字符串内容

    # 返回包含处理过的 AI 回复的状态更新
    return {"messages": [processed_ai_response]}


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

print("\n--- 启动 LangGraph 英语单词学习助手 (整合动态提示词) ---")
print("输入 '退出' 或 'exit' 结束对话。")

# --- 维护并传递状态的外部循环 ---
# 初始化 ChatState，这里需要设定一个初始的单词
initial_word = input("请输入您想学习的单词：")
current_chat_history = {"messages": [], "word": initial_word}

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
        print(f"[DEBUG] 当前学习单词: {current_chat_history['word']}")

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