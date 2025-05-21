#!/usr/bin/env python
from src.Agents import AgentClass
from src.Storage import add_user
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()
import os
import logging
import uuid


def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("main_connection.log", encoding="utf-8")  # 添加 encoding
        ]
    )
    return logging.getLogger("DingTalk")


def main():
    logger = setup_logging()
    logger.info("启动服务器流客户端")
    logger.info(f"应用ID: {os.getenv('DINGDING_ID')}")
    logger.info(f"使用凭证连接服务器")
    
    try:
        userid = str(uuid.uuid4())
        add_user("userid", userid)
        logger.info(f"用户{userid}已添加到存储中")
        agent = AgentClass()

        # 新增：首次进入时自动传一个单词，但 input 只传空字符串，确保只输出欢迎语
        initial_word = "boy"
        msg = agent.run_agent("", world=initial_word)
        print("助手：", end="", flush=True)
        if hasattr(msg, '__iter__') and not isinstance(msg, dict):
            for chunk in msg:
                print(chunk, end="", flush=True)
            print()
        else:
            print(msg.get("output", msg))
        current_word = initial_word
        first_input = True
        while True:
            user_input = input("你：")
            if user_input.lower() in ["exit", "quit", "q"]:
                print("结束会话。")
                break
            # 判断用户输入是否为英文单词（简单判断）
            if user_input.isalpha():
                current_word = user_input
            # 首次输入单词时，查释义，否则正常走 agent
            if first_input and user_input == initial_word:
                msg = agent.run_agent(user_input, world=current_word)
                first_input = False
            else:
                msg = agent.run_agent(user_input, world=current_word)
            print("助手：", end="", flush=True)
            if hasattr(msg, '__iter__') and not isinstance(msg, dict):
                for chunk in msg:
                    print(chunk, end="", flush=True)
                print()
            else:
                print(msg.get("output", msg))
    except Exception as e:
        logger.error(f"连接服务器时出错: {e}", exc_info=True)


# 确保当脚本直接运行时也能执行main函数
if __name__ == "__main__":
    main()