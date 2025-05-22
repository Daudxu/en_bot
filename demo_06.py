import uuid
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# 1. 加载环境变量
load_dotenv()

# 2. 创建 FastAPI 应用
app = FastAPI()

# 3. 聊天历史管理，按 session_id 存储多轮对话
chats_by_session_id = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    """获取或创建指定 session_id 的聊天历史"""
    chat_history = chats_by_session_id.get(session_id)
    if chat_history is None:
        chat_history = InMemoryChatMessageHistory()
        chats_by_session_id[session_id] = chat_history
    return chat_history

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
SYSTEM_PROMPT_TEMPLATE = '''你是一位专业的英语单词学习助手，当前学习单词为“{word}”。\n【对话规则】\n- 首次进入时，只输出：同学你好，针对单词“{word}”，还有什么想要了解的，我可以为你详细讲解哦~你也可以点击对话框上方的选项来进行提问。不要输出释义、用法、搭配等内容。\n- 用户输入的内容如果不是“{word}”，无论是其他英文单词还是其他内容，都只回复：咱们还是专注于“{word}”这个单词吧，你在这个单词上还有什么疑问吗？\n- 只有当用户输入“{word}”时，才输出该单词的简明中文释义，并以“你理解这个意思了吗？”结尾。例如：“这个单词的意思是‘男孩’，你理解这个意思了吗？”\n- 用户输入“详细用法”时，只输出1~2种常见用法，举例说明，并以“你理解了吗？”结尾，不要输出多余拓展。\n- 用户输入“固定搭配”时，只列举常见搭配，举例说明，并以“你记住这个搭配了吗？”结尾。\n- 用户输入“词根词缀”时，只说明有无词根词缀，简要解释，并以“现在你理解了吗？”结尾。\n- 用户输入“例句”时，只输出1个例句，并以“你能理解这个例句中‘{word}’的用法吗？”结尾。\n- 用户输入“选择题”或“出一道选择题”时，只设计一道选择题，并以“请选择A、B或C。你能找出正确答案吗？”结尾。\n- 用户输入A/B/C时，只判断正误并回复。\n\n【输出要求】\n- 只允许输出纯文本、结构化简明内容，禁止输出任何 markdown、表格、代码块、分点说明、mermaid、emoji、拓展知识、文化背景等。\n- 每次回复只聚焦用户当前问题，不要重复输出全部知识点。\n- 欢迎语只输出一次，后续不再重复。'''

# 6. 构建 LCEL Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEMPLATE),  # 系统提示词
    MessagesPlaceholder(variable_name="chat_history"),  # 聊天历史占位符
    ("user", "{input}"),  # 用户输入
])

# 7. 构建 LCEL Chain，自动注入历史
chain = (
    RunnablePassthrough.assign(chat_history=lambda x: get_chat_history(x["session_id"]).messages)
    | prompt.partial(word="apple")  # 默认单词，可动态传递
    | llm
    | StrOutputParser()
)

# 8. FastAPI WebSocket 端点，支持多 session 聊天
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())  # 为每个连接分配唯一 session_id
    print(f"新连接：{session_id}")
    # 支持前端传参切换单词
    word = "apple"
    try:
        init = await websocket.receive_text()
        if init.strip():
            word = init.strip()
    except Exception:
        pass
    # 首次进入，发送欢迎语
    result = chain.invoke({"input": "", "session_id": session_id})
    await websocket.send_text(result)
    await websocket.send_text("[END]")
    try:
        while True:
            user_input = await websocket.receive_text()
            if user_input.lower() in ["exit", "quit", "q"]:
                await websocket.close()
                return
            result = chain.invoke({"input": user_input, "session_id": session_id})
            await websocket.send_text(result)
            await websocket.send_text("[END]")
    except WebSocketDisconnect:
        pass