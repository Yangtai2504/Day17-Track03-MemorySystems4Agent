from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CONFIDENCE_THRESHOLD,
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B — 3-layer memory: short-term + User.md persistent + compact memory."""

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.confidence_threshold = confidence_threshold
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline:
            self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent and not self.force_offline:
            try:
                profile_text = self.profile_store.read_text(user_id)
                ctx = self.compact_memory.context(thread_id)
                summary = ctx.get("summary", "")
                recent = ctx.get("messages", [])

                system_content = (
                    "Bạn là trợ lý AI có bộ nhớ dài hạn.\n\n"
                    f"## Hồ sơ người dùng\n{profile_text}\n\n"
                )
                if summary:
                    system_content += f"## Tóm tắt hội thoại trước\n{summary}\n\n"
                if recent:
                    history_lines = "\n".join(
                        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: {m['content']}"
                        for m in recent
                    )
                    system_content += f"## Lịch sử gần đây\n{history_lines}\n\n"

                from langchain_core.messages import HumanMessage, SystemMessage
                result = self.langchain_agent.invoke(
                    {"messages": [HumanMessage(content=message)]},
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "system_content": system_content,
                        }
                    },
                )
                messages = result.get("messages", [])
                response_text = messages[-1].content if messages else ""
                usage = result.get("usage_metadata", {})
                agent_tokens = usage.get("output_tokens", estimate_tokens(response_text))
                prompt_tokens = usage.get("input_tokens", self._estimate_prompt_context_tokens(user_id, thread_id))

                # Update memory
                facts = extract_profile_updates(message, self.confidence_threshold)
                for key, value in facts.items():
                    self.profile_store.upsert_fact(user_id, key, value)
                self.compact_memory.append(thread_id, "user", message)
                self.compact_memory.append(thread_id, "assistant", response_text)

                self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
                self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

                return {
                    "response": response_text,
                    "agent_tokens": agent_tokens,
                    "prompt_tokens": prompt_tokens,
                }
            except Exception:
                pass
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        # 1. Extract and persist profile facts (confidence threshold + conflict handling applied inside)
        facts = extract_profile_updates(message, self.confidence_threshold)
        for key, value in facts.items():
            self.profile_store.upsert_fact(user_id, key, value)

        # 2. Append to compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 3. Estimate prompt context
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 4. Generate response
        response = self._offline_response(user_id, thread_id, message)

        # 5. Update memory and token counters
        self.compact_memory.append(thread_id, "assistant", response)
        agent_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + agent_tokens
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        return {
            "response": response,
            "agent_tokens": agent_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile_tokens = estimate_tokens(self.profile_store.read_text(user_id))
        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx.get("summary", "")))
        messages: list = ctx.get("messages", [])  # type: ignore[assignment]
        msg_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        return profile_tokens + summary_tokens + msg_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        msg_lower = message.lower()

        if any(w in msg_lower for w in ["tên", "bạn là ai", "mình là gì", "mình tên"]):
            name = facts.get("tên")
            if name:
                return f"Bạn tên là {name}."
            return "Mình chưa lưu tên của bạn."

        if any(w in msg_lower for w in ["đồ uống", "uống gì", "thích uống"]):
            drink = facts.get("đồ uống yêu thích")
            if drink:
                return f"Đồ uống yêu thích của bạn là {drink}."
            return "Mình chưa ghi nhận đồ uống yêu thích của bạn."

        if any(w in msg_lower for w in ["ở đâu", "nơi ở", "sống ở", "quê"]):
            loc = facts.get("nơi ở")
            if loc:
                return f"Bạn đang ở {loc}."
            return "Mình chưa lưu nơi ở của bạn."

        if any(w in msg_lower for w in ["nghề", "làm gì", "công việc", "làm nghề"]):
            job = facts.get("nghề nghiệp")
            if job:
                return f"Nghề nghiệp của bạn là {job}."
            return "Mình chưa ghi nhận nghề nghiệp của bạn."

        if any(w in msg_lower for w in ["style", "trả lời", "phong cách"]):
            style = facts.get("phong cách trả lời")
            if style:
                return f"Bạn thích tôi trả lời theo phong cách: {style}."
            return "Mình chưa ghi nhận phong cách trả lời bạn muốn."

        if facts:
            summary_parts = [f"**{k}**: {v}" for k, v in list(facts.items())[:3]]
            return f"[Advanced] Đã nhớ về bạn: {', '.join(summary_parts)}. Bạn vừa nhắn: {message[:60]}"

        return f"[Advanced] Đã nhận tin nhắn: {message[:80]}"

    def _maybe_build_langchain_agent(self) -> None:
        try:
            from langchain_core.messages import SystemMessage
            from langgraph.checkpoint.memory import InMemorySaver
            from langgraph.graph import END, START, MessagesState, StateGraph

            llm = build_chat_model(self.config.model)
            checkpointer = InMemorySaver()

            def call_model(state: MessagesState, config):
                system_content = config.get("configurable", {}).get(
                    "system_content",
                    "Bạn là trợ lý AI có bộ nhớ dài hạn.",
                )
                messages = [SystemMessage(content=system_content)] + state["messages"]
                response = llm.invoke(messages)
                return {"messages": [response]}

            graph = StateGraph(MessagesState)
            graph.add_node("agent", call_model)
            graph.add_edge(START, "agent")
            graph.add_edge("agent", END)
            self.langchain_agent = graph.compile(checkpointer=checkpointer)
        except Exception:
            self.langchain_agent = None
