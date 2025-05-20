from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.Agents import AgentClass
from src.Storage import add_user
import json
import logging
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 创建 FastAPI 应用并添加元数据
app = FastAPI(
    title="Agent Bot API",
    description="智能代理机器人API服务，支持WebSocket连接和HTTP请求",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI 路径
    redoc_url="/redoc",  # ReDoc 路径
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化Agent
agent = AgentClass()

# 定义请求模型
class ChatRequest(BaseModel):
    input: str
    user_id: str = "default_user"

# 设置日志
def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("server.log")
        ]
    )
    return logging.getLogger("Server")

logger = setup_logging()

# 添加POST接口处理聊天请求
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # 将用户添加到存储中
        add_user(request.user_id, {"connected": True, "last_input": request.input})
        logger.info(f"添加用户 {request.user_id} 到存储 (来自HTTP请求)")
        
        # 使用Agent处理输入
        response = agent.run_agent(request.input)
        logger.info(f"Agent响应HTTP请求: {response}")
        
        # 返回响应
        return {
            "output": response.get("output", ""),
            "result": response.get("result", "")
        }
    except Exception as e:
        logger.error(f"处理HTTP请求时出错: {e}")
        return {"error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收客户端消息
            data = await websocket.receive_text()
            try:
                # 解析JSON消息
                message = json.loads(data)
                input_text = message.get("input", "")
                user_id = message.get("user_id", "default_user")
                
                # 将用户添加到存储中 - 参照DingWebHook的实现
                add_user(user_id, {"connected": True, "last_input": input_text})
                logger.info(f"添加用户 {user_id} 到存储")
                
                # 使用Agent处理输入
                response = agent.run_agent(input_text)
                logger.info(f"Agent响应: {response}")
                
                # 发送响应
                await websocket.send_text(json.dumps({
                    "output": response.get("output", ""),
                    "result": response.get("result", "")
                }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "无效的JSON格式"
                }))
            except Exception as e:
                logger.error(f"处理消息时出错: {e}")
                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))
    except WebSocketDisconnect:
        logger.info("WebSocket客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    try:
        logger.info("启动FastAPI服务器...")
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"启动服务器失败: {e}")
        logger.error("请检查:")
        logger.error("1. 必要的依赖已安装 (fastapi, uvicorn, websockets)")
        logger.error("2. 端口8000可用")
        logger.error("3. .env文件存在并包含必要的API密钥")
