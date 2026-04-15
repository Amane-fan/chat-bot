import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

@tool
def get_weather(location: str) -> str:
    """获取天气信息的工具函数"""
    return f"{location}天气晴朗"

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")

model = ChatOpenAI(
    model="qwen-plus",
    temperature=1
)
model = model.bind_tools([get_weather])

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是只能助手"),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}")
    ]
)

chain = prompt_template | model

response = chain.invoke({"question": "我是Amane,你好"})

print(response.content)