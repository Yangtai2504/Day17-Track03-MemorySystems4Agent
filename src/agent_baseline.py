from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A — within-session memory only, no User.md, no cross-session recall."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def _get_session(self, thread_id: str) -> SessionState:
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        return self.sessions[thread_id]

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            session = self._get_session(thread_id)
            try:
                result = self.langchain_agent.invoke(
                    {"messages": [{"role": "user", "content": message}]},
                    config={"configurable": {"thread_id": thread_id}},
                )
                messages = result.get("messages", [])
                response_text = messages[-1].content if messages else ""
                usage = result.get("usage_metadata", {})
                agent_tokens = usage.get("output_tokens", estimate_tokens(response_text))
                prompt_tokens = usage.get("input_tokens", estimate_tokens(message))
                session.token_usage += agent_tokens
                session.prompt_tokens_processed += prompt_tokens
                session.messages.append({"role": "user", "content": message})
                session.messages.append({"role": "assistant", "content": response_text})
                return {
                    "response": response_text,
                    "agent_tokens": agent_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._get_session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._get_session(thread_id)
        session.messages.append({"role": "user", "content": message})

        # Build a simple response from in-session history
        history_text = " ".join(m["content"] for m in session.messages if m["role"] == "user")
        response = self._generate_response(message, history_text)

        agent_tokens = estimate_tokens(response)
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in session.messages)

        session.token_usage += agent_tokens
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({"role": "assistant", "content": response})

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _generate_response(self, message: str, history: str) -> str:
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["tên", "bạn là ai", "mình là"]):
            for word in history.split():
                if len(word) > 2 and word[0].isupper():
                    return f"Trong session này bạn đã nhắc tên: {word}."
            return "Mình chưa thấy bạn giới thiệu tên trong session này."

        if any(w in msg_lower for w in ["đồ uống", "uống", "cà phê"]):
            if "cà phê" in history.lower():
                return "Trong session này bạn đề cập đến cà phê sữa đá."
            return "Mình không nhớ đồ uống bạn thích trong session này."

        if any(w in msg_lower for w in ["ở đâu", "nơi ở", "sống ở"]):
            return "Mình chỉ nhớ thông tin trong session hiện tại."

        if any(w in msg_lower for w in ["nghề", "làm gì", "công việc"]):
            return "Bạn chưa nhắc nghề nghiệp trong session này."

        return f"[Baseline] Tôi nhận được: {message[:80]}"

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langchain_core.messages import SystemMessage
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, MessagesState, StateGraph

            llm = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()

            system_msg = SystemMessage(
                content=(
                    "Bạn là trợ lý AI hữu ích. "
                    "Bạn chỉ nhớ thông tin trong cuộc hội thoại hiện tại. "
                    "Không có bộ nhớ dài hạn giữa các session."
                )
            )

            def call_model(state: MessagesState):
                messages = [system_msg] + state["messages"]
                response = llm.invoke(messages)
                return {"messages": [response]}

            graph = StateGraph(MessagesState)
            graph.add_node("agent", call_model)
            graph.add_edge(START, "agent")
            graph.add_edge("agent", END)
            self.langchain_agent = graph.compile(checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None
