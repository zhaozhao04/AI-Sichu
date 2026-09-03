from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.chat_models import init_chat_model
import os
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pathlib import Path

# 获取脚本所在目录，并创建数据库路径
# script_dir = Path(__file__).parent
# db_path = script_dir / "resources" / "personal_chief.db"
# db_path.parent.mkdir(parents=True, exist_ok=True)

load_dotenv()

web_search = TavilySearch(
    max_results=5,
    topic="general",
)

model = init_chat_model(
    model="qwen3.7-plus",
    model_provider="openai",
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# connection = sqlite3.connect("resources/personal_chief.db", check_same_thread=False)
# checkpointer = SqliteSaver(connection)
# checkpointer.setup()

system_prompt = """
你是一名私人厨师。收到用户提供的食材照片或清单后，请按以下流程操作：
1.识别和评估食材：若用户提供照片，首先辨识所有可见食材。基于食材的外观状态，评估其新鲜度与可用量，整理出一份“当前可用食材清单”。
2.智能食谱检索：优先调用 web_search 工具，以“可用食材清单”为核心关键词，查找可行菜谱。
3.多维度评估与排序：从营养价值和制作难度两个维度对检索到的候选食谱进行量化打分，并根据得分排序，制作简单且营养丰富的排名靠前。
4.结构化方案输出：把排序后的食谱整理为一份结构清晰的建议报告，要包含食谱信息、得分、推荐理由、食谱的参考图片，帮助用户快速做出决策。

请严格按照流程，优先调用 web_search 工具搜索食谱，搜索不到的情况下才能自己发挥。
"""

def build_agent(checkpointer=None):
    """构建私厨 agent。

    checkpointer: 传入则启用本地记忆（FastAPI 部署用）；
    不传则为无状态版本（langgraph.json / LangSmith 部署用，平台自带记忆）。
    """
    return create_agent(
        model=model,
        tools=[web_search],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


# langgraph.json 指向此对象；LangGraph 平台自带 checkpointer，勿在此挂本地 checkpointer
chief_agent = build_agent()


# LangChain的Agent底层是基于LangGraph实现的，而LangGraph提供了完整的后端部署功能，自带非常完善的API接口，无需我们额外处理。
# 同时，LangChain还提供了基于LangSmith的GUI控制台实现Agent的调试、监控、一键部署。
# 注意如果是langsmith去部署agent，那么不需要checkpointer去管理记忆，如果开启checkpointer反而会报错，langsmith只需要创建一个智能体就行了


# multimodal_message = HumanMessage(
#     content=[
#         {"type": "image",
#          "url": "https://img.freepik.com/free-photo/arrangement-different-foods-organized-fridge_23-2149099882.jpg"},
#         {"type": "text", "text": "帮我看看这些食材能做些什么？"}
#     ])

# config = {"configurable": {"thread_id": "1"}}

# response = agent.invoke({"messages": [multimodal_message]}, config)

# for message in response['messages']:
#     message.pretty_print()