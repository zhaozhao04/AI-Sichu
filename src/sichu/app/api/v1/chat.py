import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ...agents.chief import build_agent
from ...models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()

# 记忆数据库路径：src/sichu/resources/personal_chief.db
DB_PATH = Path(__file__).parents[3] / "resources" / "personal_chief.db"

# 在 lifespan 中初始化的带记忆 agent
_agent: Optional[Any] = None


@asynccontextmanager
async def lifespan(app):
    """启动时创建带 Sqlite 记忆的 agent，关闭时释放数据库连接"""
    global _agent
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        await checkpointer.setup()
        _agent = build_agent(checkpointer=checkpointer)
        logger.info("私厨 agent 已就绪，记忆数据库: %s", DB_PATH)
        yield
    _agent = None


# LangChain 消息类型 -> 前端角色
_ROLE_MAP = {"human": "user", "ai": "assistant"}


def _serialize_content(content: Any):
    """把消息内容序列化为前端可渲染的格式：str 或 [{type, text|url}] 列表"""
    if isinstance(content, str):
        return content
    blocks = []
    for block in content or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and block.get("text"):
            blocks.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image" and block.get("url"):
            blocks.append({"type": "image", "url": block["url"]})
    return blocks


async def _stream_tokens(
    thread_id: str, message: str, image_url: Optional[str]
) -> AsyncIterator[str]:
    """调用 agent 并把 token 逐块产出（前端直接按纯文本拼接，勿加 SSE 包装）"""
    if image_url:
        content = [
            {"type": "image", "url": image_url},
            {"type": "text", "text": message},
        ]
    else:
        content = message

    config = {"configurable": {"thread_id": thread_id}}
    try:
        async for chunk, _ in _agent.astream(
            {"messages": [HumanMessage(content=content)]},
            config,
            stream_mode="messages",
        ):
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield chunk.content
    except Exception as e:
        # 响应头已发出，无法改状态码，只能把错误写进流里
        logger.exception("对话流式输出失败, thread_id=%s", thread_id)
        yield f"\n[出错] {e}"


@router.post("/chat/stream")
async def chat_endpoint(request: ChatRequest):
    """流式对话"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent 未初始化")
    return StreamingResponse(
        _stream_tokens(request.thread_id, request.message, request.image_url),
        media_type="text/plain; charset=utf-8",
    )


@router.get("/chat/messages")
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent 未初始化")
    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = await _agent.aget_state(config)
    except ValueError:
        return {"messages": []}
    if state is None:
        return {"messages": []}

    messages = []
    for msg in state.values.get("messages", []):
        role = _ROLE_MAP.get(getattr(msg, "type", ""))
        if role is None:
            continue  # 跳过工具调用、系统消息等
        content = _serialize_content(msg.content)
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return {"messages": messages}


@router.delete("/chat/messages")
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="agent 未初始化")
    try:
        await _agent.checkpointer.adelete_thread(thread_id)
    except Exception:
        # thread 不存在等场景视为已清空
        logger.exception("清空历史消息失败, thread_id=%s", thread_id)
    return {"status": "ok"}
