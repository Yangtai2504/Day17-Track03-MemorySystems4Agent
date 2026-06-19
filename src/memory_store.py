from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# UserProfileStore
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = "# User Profile\n\n"


@dataclass
class UserProfileStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, user_id: str) -> Path:
        safe = re.sub(r"[^\w\-]", "_", user_id)
        return self.root_dir / f"{safe}.md"

    def read_text(self, user_id: str) -> str:
        p = self.path_for(user_id)
        return p.read_text(encoding="utf-8") if p.exists() else _DEFAULT_PROFILE

    def write_text(self, user_id: str, content: str) -> Path:
        p = self.path_for(user_id)
        p.write_text(content, encoding="utf-8")
        return p

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        p = self.path_for(user_id)
        return p.stat().st_size if p.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        text = self.read_text(user_id)
        result: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"\*\*(.+?)\*\*:\s*(.+)", line)
            if m:
                result[m.group(1).strip().lower()] = m.group(2).strip()
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        text = self.read_text(user_id)
        marker = f"**{key}**:"
        new_line = f"**{key}**: {value}"
        if marker in text:
            lines = text.splitlines()
            lines = [new_line if l.startswith(marker) else l for l in lines]
            self.write_text(user_id, "\n".join(lines) + "\n")
        else:
            if not text.endswith("\n"):
                text += "\n"
            self.write_text(user_id, text + new_line + "\n")


# ---------------------------------------------------------------------------
# Bonus 1 — Confidence Threshold
# ---------------------------------------------------------------------------

# Default threshold: only persist facts with confidence >= this value.
CONFIDENCE_THRESHOLD = 0.6

# Words that signal the message is hypothetical or a joke → lower confidence.
_JOKE_RE = re.compile(
    r"\b(đùa|chỉ là câu đùa|hay là chuyển|thử xem|giả sử|nếu như)\b",
    re.IGNORECASE,
)

# Words that signal genuine first-person statements → higher confidence.
_FIRST_PERSON_RE = re.compile(r"\b(mình|tôi|tao)\b", re.IGNORECASE)


def _compute_confidence(message: str, value: str, base: float) -> float:
    """Adjust base confidence using simple heuristics."""
    score = base

    # Longer, more detailed messages are more reliable.
    if len(message) > 40:
        score += 0.05

    # Very short extracted values are often noise.
    if len(value.strip()) < 3:
        score -= 0.35

    # Single-word values that look like question words are noise.
    if value.strip().lower() in {"gì", "ai", "đâu", "nào", "sao", "thế", "vậy"}:
        score -= 0.9

    # Joke / hypothetical context → not reliable.
    if _JOKE_RE.search(message):
        score -= 0.45

    # Explicit first-person subject → more reliable.
    if _FIRST_PERSON_RE.search(message):
        score += 0.05

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Bonus 2 — Conflict Handling
# ---------------------------------------------------------------------------

# Markers that signal the user is correcting a previously stated fact.
_CORRECTION_RE = re.compile(
    r"\b(thực ra|sửa lại|không phải|nhưng thực ra|cập nhật lại|"
    r"đúng hơn là|thay đổi thành|nhầm rồi|thực tế là|thực chất)\b",
    re.IGNORECASE,
)


def is_correction(message: str) -> bool:
    """Return True when the message contains an explicit correction marker."""
    return bool(_CORRECTION_RE.search(message))


# ---------------------------------------------------------------------------
# extract_profile_updates (combines both bonuses)
# ---------------------------------------------------------------------------

_QUESTION_RE = re.compile(
    r"^\s*(bạn|bạn có|có thể|làm ơn|hãy|cho mình|cho tôi)",
    re.IGNORECASE,
)

# Strip common Vietnamese sentence-final particles from extracted values.
_TRAILING_PARTICLES_RE = re.compile(
    r"\s+(?:rồi|nhé|nha|đây|này|luôn|thôi|vậy|ạ|nhỉ|nhở|hen|ha|ok|nữa)$",
    re.IGNORECASE,
)


def _clean_value(value: str) -> str:
    value = value.strip().rstrip(".,")
    return _TRAILING_PARTICLES_RE.sub("", value).strip()


# (fact_key, compiled_pattern, base_confidence)
_SCORED_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("tên", re.compile(
        r"(?:mình|tôi|tao)\s+tên(?:\s+là)?\s+([A-ZÀ-Ỹa-zà-ỹ][^\.,\n!?]{1,28})",
        re.IGNORECASE,
    ), 0.85),
    ("nơi ở", re.compile(
        r"(?:mình|tôi|tao)\s+(?:ở|sống ở|đang ở|tại|làm việc ở|đang làm việc ở|làm việc tại)\s+([^\.,\n!?]{2,30})",
        re.IGNORECASE,
    ), 0.75),
    ("nghề nghiệp", re.compile(
        r"(?:mình|tôi|tao)\s+(?:đang làm|làm)\s+([^\.,\n!?]{2,40})",
        re.IGNORECASE,
    ), 0.70),
    ("đồ uống yêu thích", re.compile(
        r"(?:đồ uống yêu thích(?:\s+(?:của\s+)?(?:mình|tôi))?)\s+(?:là|:)?\s*([^\.,\n!?]{2,30})",
        re.IGNORECASE,
    ), 0.80),
    ("sở thích", re.compile(
        r"(?:mình|tôi)\s+(?:thích|yêu thích)\s+([^\.,\n!?]{2,50})",
        re.IGNORECASE,
    ), 0.65),
    ("phong cách trả lời", re.compile(
        r"(?:trả lời|câu trả lời|hãy|style).{0,20}(?:ngắn gọn|rõ ý|bullet|có ví dụ|gọn)",
        re.IGNORECASE,
    ), 0.75),
]


def extract_profile_updates(
    message: str,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, str]:
    """Return stable profile facts extracted from *message*.

    Bonus 1 — Confidence threshold: only include facts whose confidence
    score meets *confidence_threshold*.

    Bonus 2 — Conflict handling: when correction markers are present,
    prefer the last (corrected) match and boost its confidence; also skip
    negated matches ("không ở Huế" should not store "Huế").
    """
    stripped = message.strip()

    # Skip pure questions.
    if stripped.endswith("?") or _QUESTION_RE.match(stripped):
        return {}

    correction = is_correction(message)
    # Correction messages get a confidence boost: the user is explicitly
    # stating the correct value, so we trust it more.
    correction_boost = 0.15 if correction else 0.0

    updates: dict[str, str] = {}

    for key, pattern, base_conf in _SCORED_PATTERNS:
        all_matches = list(pattern.finditer(message))
        if not all_matches:
            continue

        if pattern.groups == 0:
            # Pattern has no capture group (e.g. style pattern).
            updates[key] = "ngắn gọn, rõ ý, có ví dụ thực tế"
            continue

        # For correction messages take the LAST match (the corrected value
        # comes after the thing being corrected).  Otherwise take the first.
        # Note: negated forms like "mình không ở Huế" never reach here because
        # the patterns require the location keyword to immediately follow the
        # subject pronoun with only whitespace — "không" breaks that connection.
        chosen = all_matches[-1] if correction else all_matches[0]
        value = _clean_value(chosen.group(1))
        conf = _compute_confidence(message, value, base_conf + correction_boost)

        if conf >= confidence_threshold:
            updates[key] = value

    return updates


# ---------------------------------------------------------------------------
# CompactMemoryManager
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    kept = messages[-max_items:] if len(messages) > max_items else messages
    lines = []
    for m in kept:
        role = "Người dùng" if m["role"] == "user" else "Trợ lý"
        lines.append(f"{role}: {m['content'][:120]}")
    return "[Tóm tắt hội thoại cũ]\n" + "\n".join(lines)


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _init_thread(self, thread_id: str) -> None:
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

    def append(self, thread_id: str, role: str, content: str) -> None:
        self._init_thread(thread_id)
        s = self.state[thread_id]
        s["messages"].append({"role": role, "content": content})  # type: ignore[index]

        total_tokens = sum(
            estimate_tokens(m["content"]) for m in s["messages"]  # type: ignore[index]
        )
        if total_tokens > self.threshold_tokens:
            messages: list = s["messages"]  # type: ignore[assignment]
            old = messages[: -self.keep_messages]
            kept = messages[-self.keep_messages :]
            s["summary"] = summarize_messages(old)
            s["messages"] = kept
            s["compactions"] = int(s["compactions"]) + 1  # type: ignore[arg-type]

    def context(self, thread_id: str) -> dict[str, object]:
        self._init_thread(thread_id)
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        self._init_thread(thread_id)
        return int(self.state[thread_id]["compactions"])  # type: ignore[arg-type]
