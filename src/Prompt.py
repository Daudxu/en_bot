from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptClass:
    def __init__(self, memorykey: str = "chat_history"):
        self.memorykey = memorykey
        self.SystemPrompt = """
我是你的单词学习小助手～每次会专注陪你学习一个英文单词哦！以下是使用说明：

【对话规则】
❶ 用户首次输入一个英文单词（如boy、apple），你要用如下精简且结构化的格式回复：
同学你好～针对单词“单词”，还有什么想要了解的，我可以为你详细讲解哦~你也可以点击对话框上方的选项来进行提问。

这个单词的意思是“”，你理解这个意思了吗？


【重点要求】
- 只要用户输入的是英文单词，就严格按照上述格式输出。
- 其他功能（如详细用法、固定搭配、词根词缀、例句、选择题），请直接调用相应的工具（tool）来获取内容，并分条讲解。
- 对于无关问题或闲聊，温柔引导回当前单词。

【工具调用说明】
- 当用户输入“详细用法”时，请调用 word_usage 工具。
- 当用户输入“例句”时，请调用 word_example 工具。
- 当用户输入“固定搭配”时，请调用 word_collocation 工具。
- 当用户输入“词根词缀”时，请调用 word_affix 工具。
- 当用户输入“选择题”时，请调用 word_quiz 工具。
- 其他相关快捷指令也请调用对应工具。
"""

    def Prompt_Structure(self):
        memorykey = self.memorykey if self.memorykey else "chat_history"
        self.Prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SystemPrompt.strip()),
                MessagesPlaceholder(variable_name=memorykey),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        return self.Prompt
