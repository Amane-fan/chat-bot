from dataclasses import field
import os
from pydoc import describe
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from openai import base_url
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("DASHSCOPE_BASE_URL")

@tool
def get_weather(location: str) -> str:
    """获取某地的天气"""
    return f"{location}有龙卷风！"

model = ChatOpenAI(
    model="qwen-plus",
    temperature=1.0
)

agent = create_agent(
    model=model,
    tools=[get_weather]
)

result = agent.invoke({
    "messages": [
        {"role": "user", "content": "我在苏州"},
        {"role": "user", "content": "今天我这里的天气怎么样?"}
    ]
})

parser = StrOutputParser()

for message in result["messages"]:
    print(f"{type(message).__name__}: {message.content}")