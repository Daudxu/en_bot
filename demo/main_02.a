import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# 1. 加载环境变量
load_dotenv()

# 2. 创建 FastAPI 应用
app = FastAPI()

# 3. 聊天历史管理，直接用变量存储每个 session 的历史（仅存字符串，不用 InMemoryChatMessageHistory）
chats_by_session_id = {}

def get_chat_history(session_id: str, max_history: int = 10):
    history = chats_by_session_id.get(session_id, [])
    # 控制历史长度
    if max_history > 0 and len(history) > max_history:
        history = history[-max_history:]
        chats_by_session_id[session_id] = history
    return history

def add_history(session_id: str, role: str, content: str, max_history: int = 10):
    history = chats_by_session_id.get(session_id, [])
    history.append({"role": role, "content": content})
    if max_history > 0 and len(history) > max_history:
        history = history[-max_history:]
    chats_by_session_id[session_id] = history

def remove_chat_history(session_id: str):
    if session_id in chats_by_session_id:
        del chats_by_session_id[session_id]

# 4. LLM 配置，支持 OpenAI 兼容模型
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

# 5. Prompt 模板，支持动态单词
SYSTEM_PROMPT_TEMPLATE = '''你是一个助手'''

# 6. 构建 LCEL Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])

# 7. 构建 LCEL Chain，自动注入历史
MAX_HISTORY = 10
chain = (
    RunnablePassthrough.assign(
        chat_history=lambda x: get_chat_history(x["session_id"], max_history=MAX_HISTORY)
    )
    | prompt.partial(word="apple")
    | llm
    | StrOutputParser()
)

# 8. FastAPI WebSocket 端点，支持多 session 聊天
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    word = "apple"
    try:
        init = await websocket.receive_text()
        if init.strip():
            word = init.strip()
    except Exception:
        pass
    # 首次进入，发送欢迎语
    result = chain.invoke({"input": "", "session_id": session_id})
    add_history(session_id, "user", "")
    add_history(session_id, "assistant", result)
    await websocket.send_text(result)
    await websocket.send_text("[END]")
    try:
        while True:
            user_input = await websocket.receive_text()
            if user_input.lower() in ["exit", "quit", "q"]:
                await websocket.close()
                remove_chat_history(session_id)
                return
            result = chain.invoke({"input": user_input, "session_id": session_id})
            add_history(session_id, "user", user_input)
            add_history(session_id, "assistant", result)
            await websocket.send_text(result)
            await websocket.send_text("[END]")
    except WebSocketDisconnect:
        remove_chat_history(session_id)
        pass