import os
from typing import TypedDict, Annotated, List
import operator
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import HTMLResponse # 用于提供简单的前端页面

# --- LangGraph 配置和初始化 (与您之前的代码基本相同) ---
load_dotenv()
# 确保您的 .env 文件中 MODEL_API_KEY, BASE_URL, MODEL_NAME 都已正确设置
base_url = os.getenv("BASE_URL")
model_api_key = os.getenv("MODEL_API_KEY")
model_name = os.getenv("MODEL_NAME")

llm = ChatOpenAI(
    base_url=base_url,
    api_key=model_api_key,
    model=model_name,
    temperature=0.1,
    max_tokens=512,
    streaming=True
)

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

# --- 1. 定义图的状态 ---
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    word: str # 用于在整个图中传递当前学习的单词

# --- 2. 定义节点函数 ---
# 构建 ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT_TEMPLATE),
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
    current_word = state["word"]

    # 调用 llm_chain，并将消息历史和 word 变量传入
    ai_response = llm_chain.invoke({"messages": current_messages, "word": current_word}) 
    
    # 后处理 LLM 的回复，替换其中的 {word} 占位符
    if isinstance(ai_response.content, str):
        processed_content = ai_response.content.replace("{word}", current_word)
        processed_ai_response = AIMessage(content=processed_content)
    else:
        processed_ai_response = ai_response # 保留原样，以防非字符串内容

    # 返回包含处理过的 AI 回复的状态更新
    return {"messages": [processed_ai_response]}

def check_quit_node(state: ChatState) -> ChatState:
    """
    一个常规节点，用于打印决策信息，但不直接做路由。
    路由由后续的 conditional_edges 决定。
    """
    # 在 WebSocket 中，这个节点主要用于打印和逻辑判断，不直接修改状态
    # 实际的退出逻辑将由 WebSocket 连接管理
    return {}

def route_decision(state: ChatState) -> str:
    """
    纯粹的路由函数，只用于 conditional_edges，不作为节点。
    """
    current_messages = state["messages"]
    last_human_message = None
    for msg in reversed(current_messages):
        if isinstance(msg, HumanMessage):
            last_human_message = msg
            break

    # 在 WebSocket 环境中，我们倾向于让客户端决定何时断开。
    # 这里的 "END" 更多的是为了让 LangGraph 内部的单轮执行完成，
    # 而不是立即关闭 WebSocket 连接。
    if last_human_message and last_human_message.content.lower() in ["退出", "exit"]:
        print("路由：用户选择退出 (LangGraph 内部流程结束)。")
        return "END"
    else:
        print("路由：继续对话到 LLM。")
        return "llm_node"

# --- 3. 构建并编译图 ---
workflow = StateGraph(ChatState)
workflow.add_node("llm_node", llm_node_chat)
workflow.add_node("check_quit_node", check_quit_node)

workflow.set_entry_point("check_quit_node")
workflow.add_conditional_edges(
    "check_quit_node", 
    route_decision,    
    {
        "llm_node": "llm_node",
        "END": END
    }
)
# 注意：在 WebSocket 场景中，llm_node 处理完后，我们直接结束 LangGraph 的一轮处理
# 因为将结果发送回客户端的逻辑在 FastAPI 的 WebSocket 路由中。
workflow.add_edge("llm_node", END) 

# 编译 LangGraph 应用
langgraph_app = workflow.compile()

# --- FastAPI 应用初始化 ---
fastapi_app = FastAPI(
    title="LangGraph WebSocket Chatbot",
    description="一个使用 LangGraph 和 FastAPI WebSockets 构建的英语单词学习助手。"
)

# --- WebSocket 端点 ---
# 客户端通过 /ws/{word} 连接，例如 ws://localhost:8000/ws/apple
@fastapi_app.websocket("/ws/{word}")
async def websocket_endpoint(websocket: WebSocket, word: str):
    await websocket.accept() # 接受 WebSocket 连接
    print(f"新 WebSocket 连接建立，学习单词: {word}")

    # 为当前连接初始化 LangGraph 状态
    # `messages` 列表用于存储当前连接的对话历史
    chat_history_for_connection = {"messages": [], "word": word}

    # --- 首次自动触发欢迎语 ---
    try:
        # 为了触发 SYSTEM_PROMPT_TEMPLATE 中的“首次进入时”规则，
        # 我们向 LangGraph 传入一个空的 HumanMessage。
        # 这样 LLM 会识别到这是对话的开始，并按规则生成欢迎语。
        initial_invoke_state = {"messages": [HumanMessage(content="")], "word": word}
        
        # 调用 LangGraph 处理首次请求
        result_state_greeting = langgraph_app.invoke(initial_invoke_state)
        
        # 欢迎语是 LangGraph 返回状态中的最后一条 AI 消息
        greeting_message = result_state_greeting["messages"][-1].content
        await websocket.send_text(greeting_message) # 发送欢迎语给客户端
        
        # 将欢迎语也添加到当前连接的对话历史中，以便后续对话使用
        chat_history_for_connection = result_state_greeting

    except Exception as e:
        print(f"生成首次欢迎语时发生错误: {e}")
        await websocket.send_text("抱歉，初始化聊天时发生错误。请重试。")
        await websocket.close()
        return

    # --- 循环接收和发送消息 ---
    try:
        while True:
            # 接收客户端发送的文本消息
            user_input_content = await websocket.receive_text()
            print(f"收到来自客户端的消息 (单词 '{word}'): {user_input_content}")

            # 如果用户输入“退出”或“exit”，则关闭连接
            if user_input_content.lower() in ["退出", "exit"]:
                await websocket.send_text("再见！对话已结束。")
                break # 退出循环，从而关闭 WebSocket 连接

            # 将用户消息添加到当前连接的对话历史中
            chat_history_for_connection["messages"].append(HumanMessage(content=user_input_content))

            # 调用 LangGraph 进行处理
            result_state = langgraph_app.invoke(chat_history_for_connection)

            # 更新当前连接的对话历史，以便下一轮使用
            chat_history_for_connection = result_state

            # 从 LangGraph 返回的状态中获取 AI 的最新回复
            ai_response_message = chat_history_for_connection["messages"][-1]

            # 检查并发送 AI 回复给客户端
            if isinstance(ai_response_message, AIMessage):
                await websocket.send_text(ai_response_message.content)
            else:
                await websocket.send_text("抱歉，未能获取有效回复。")

    except WebSocketDisconnect:
        print(f"客户端 (单词 '{word}') 断开连接。")
    except Exception as e:
        print(f"处理消息时发生错误 (单词 '{word}'): {e}")
        import traceback
        traceback.print_exc() # 打印详细的错误堆栈
        await websocket.send_text(f"抱歉，服务器发生错误: {e}")
    finally:
        # 确保 WebSocket 连接最终被关闭
        await websocket.close()
        print(f"WebSocket 连接 (单词 '{word}') 已关闭。")

# --- 提供一个简单的 HTML 页面作为客户端 (可选) ---
# 这将允许您直接在浏览器中测试 WebSocket 连接。
# 您可以在浏览器中访问 http://localhost:8000/
html = """
<!DOCTYPE html>
<html>
<head>
    <title>LangGraph WebSocket dev test</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { max-width: 800px; margin: 0 auto; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #007bff; text-align: center; margin-bottom: 20px; }
        #controls { margin-bottom: 15px; display: flex; align-items: center; }
        #wordInput { flex-grow: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; margin-right: 10px; }
        button { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background-color: #0056b3; }
        button:disabled { background-color: #cccccc; cursor: not-allowed; }
        #chatbox { border: 1px solid #e0e0e0; background-color: #fdfdfd; padding: 15px; height: 400px; overflow-y: auto; margin-bottom: 15px; border-radius: 4px; display: flex; flex-direction: column; }
        .message-container { margin-bottom: 10px; display: flex; }
        .user-message .message-bubble { background-color: #e6f7ff; align-self: flex-end; }
        .ai-message .message-bubble { background-color: #e6ffe6; align-self: flex-start; }
        .message-bubble { max-width: 70%; padding: 10px 15px; border-radius: 18px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.1); word-wrap: break-word; }
        .user-message { justify-content: flex-end; }
        .ai-message { justify-content: flex-start; }
        .system-message { text-align: center; font-style: italic; color: #888; font-size: 0.9em; margin-bottom: 10px; }
        #messageInput { width: calc(100% - 90px); padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 16px; margin-right: 10px; }
        #input-area { display: flex; }
    </style>
</head>
<body>
    <div class="container">
        <h1>dev test</h1>
        <div id="controls">
            <label for="wordInput">请输入要word：</label>
            <input type="text" id="wordInput" value="apple" placeholder="输入单词，例如 'book'">
            <button id="connectButton" onclick="connectWebSocket()">连接</button>
        </div>
        <div id="chatbox"></div>
        <div id="input-area">
            <input type="text" id="messageInput" placeholder="输入消息..." disabled>
            <button id="sendButton" onclick="sendMessage()" disabled>发送</button>
        </div>
    </div>

    <script>
        let ws;
        const chatbox = document.getElementById('chatbox');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const wordInput = document.getElementById('wordInput');
        const connectButton = document.getElementById('connectButton');

        function appendMessage(sender, message) {
            const messageContainer = document.createElement('div');
            messageContainer.className = 'message-container ' + (sender === '您' ? 'user-message' : 'ai-message');
            
            const messageBubble = document.createElement('div');
            messageBubble.className = 'message-bubble';
            messageBubble.textContent = message;
            
            if (sender === '系统') {
                const systemP = document.createElement('p');
                systemP.className = 'system-message';
                systemP.textContent = message;
                chatbox.appendChild(systemP);
            } else {
                messageContainer.appendChild(messageBubble);
                chatbox.appendChild(messageContainer);
            }
            chatbox.scrollTop = chatbox.scrollHeight; // Scroll to bottom
        }

        function connectWebSocket() {
            if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
                ws.close(); // Close existing connection if any
            }
            const word = wordInput.value.trim();
            if (!word) {
                alert("请输入要学习的单词！");
                return;
            }
            
            // 构建 WebSocket URL，假设您的 FastAPI 运行在 localhost:8000
            const wsUrl = `ws://localhost:8000/ws/${encodeURIComponent(word)}`; 
            ws = new WebSocket(wsUrl); 
            
            chatbox.innerHTML = ''; // Clear chatbox on new connection
            appendMessage('系统', `正在连接到单词“${word}”...`);
            wordInput.disabled = true;
            connectButton.disabled = true;

            ws.onopen = (event) => {
                appendMessage('系统', '连接成功！请等待AI助手回复欢迎语。');
                messageInput.disabled = false;
                sendButton.disabled = false;
                messageInput.focus();
            };

            ws.onmessage = (event) => {
                appendMessage('AI', event.data);
            };

            ws.onclose = (event) => {
                appendMessage('系统', '连接已关闭。您可以输入新单词并重新连接。');
                messageInput.disabled = true;
                sendButton.disabled = true;
                wordInput.disabled = false; 
                connectButton.disabled = false;
            };

            ws.onerror = (event) => {
                appendMessage('系统', 'WebSocket 错误！');
                console.error("WebSocket Error:", event);
            };
        }

        function sendMessage() {
            const message = messageInput.value.trim();
            if (ws && ws.readyState === WebSocket.OPEN && message) {
                ws.send(message);
                appendMessage('您', message);
                messageInput.value = '';
                if (message.toLowerCase() === '退出' || message.toLowerCase() === 'exit') {
                    // 对于 '退出' 命令，客户端会发送，服务器收到后会关闭连接
                    // 页面上的 ws.onclose 会处理连接关闭后的 UI 状态
                }
            }
        }

        // 允许用户按 Enter 键发送消息
        messageInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });

        // 页面加载时自动连接一个默认单词（可选）
        window.onload = function() {
            connectWebSocket(); 
        };
    </script>
</body>
</html>
"""

# 定义一个 FastAPI 路由来提供上面的 HTML 页面
@fastapi_app.get("/", response_class=HTMLResponse)
async def get_root():
    return html