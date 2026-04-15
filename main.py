import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """获取天气信息的工具函数"""
    return f"{location}天气晴朗"

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")

model = ChatOpenAI(
    model="qwen-plus"
)
model = model.bind_tools([get_weather])

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个天气查询助手"),
        ("human", "请告诉我{province}的天气")
    ]
)

chain = prompt_template | model

response = chain.invoke({"province": "广东"})

print(response)
print(response.content)
