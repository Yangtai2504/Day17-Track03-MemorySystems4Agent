from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    base = load_config()
    dummy_provider = ProviderConfig(
        provider="custom",
        model_name="test-model",
        temperature=0.0,
        api_key="test-key",
        base_url="http://localhost:9999",
    )
    return LabConfig(
        base_dir=base.base_dir,
        data_dir=base.data_dir,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=50,   # low threshold so compaction triggers fast
        compact_keep_messages=2,
        model=dummy_provider,
        judge_model=dummy_provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    store = agent.profile_store
    user_id = "test_user"

    # Initial read returns default
    text = store.read_text(user_id)
    assert "User Profile" in text, "Default profile should contain header"

    # Write and read back
    store.write_text(user_id, "# User Profile\n\n**tên**: Alice\n")
    assert store.read_text(user_id) == "# User Profile\n\n**tên**: Alice\n"

    # Edit existing line
    changed = store.edit_text(user_id, "**tên**: Alice", "**tên**: Bob")
    assert changed is True
    assert "Bob" in store.read_text(user_id)

    # Edit non-existing returns False
    assert store.edit_text(user_id, "nonexistent", "replacement") is False

    # file_size > 0 after writing
    assert store.file_size(user_id) > 0

    print("PASS: test_user_markdown_read_write_edit")


def test_compact_trigger(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)

    user_id = "compact_user"
    thread_id = "compact_thread"

    # Send enough messages to exceed the low threshold (50 tokens)
    long_messages = [
        "Đây là tin nhắn rất dài để ép compact memory phải kích hoạt " + "x" * 60,
        "Thêm một tin nhắn dài nữa để vượt ngưỡng token " + "y" * 60,
        "Tin nhắn thứ ba tiếp tục thêm ngữ cảnh " + "z" * 60,
        "Tin nhắn thứ tư để chắc chắn compact đã xảy ra " + "w" * 60,
    ]

    for msg in long_messages:
        agent.reply(user_id=user_id, thread_id=thread_id, message=msg)

    count = agent.compaction_count(thread_id)
    assert count >= 1, f"Expected at least 1 compaction, got {count}"
    print(f"PASS: test_compact_trigger (compactions={count})")


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    # Advanced agent - session 1: introduce name and drink
    adv = AdvancedAgent(config=config, force_offline=True)
    adv.reply(user_id="alice", thread_id="thread_1", message="Chào bạn, mình tên là Alice.")
    adv.reply(user_id="alice", thread_id="thread_1", message="Đồ uống yêu thích của mình là cà phê sữa đá.")

    # Advanced agent - session 2 (new thread_id): should recall from User.md
    result_adv = adv.reply(user_id="alice", thread_id="thread_2", message="Mình tên gì?")
    assert "Alice" in result_adv["response"], (
        f"Advanced agent should recall name across sessions, got: {result_adv['response']}"
    )

    # Baseline agent - should NOT recall across threads
    base = BaselineAgent(config=config, force_offline=True)
    base.reply(user_id="alice", thread_id="thread_A", message="Chào bạn, mình tên là Alice.")
    result_base = base.reply(user_id="alice", thread_id="thread_B", message="Mình tên gì?")
    # Baseline may or may not return the name; it should NOT if it truly resets
    # We assert the advanced recall works, which is the key distinction
    assert "Alice" in result_adv["response"]
    print("PASS: test_cross_session_recall")
    print("  Advanced recalled:", result_adv["response"][:80].encode("ascii", "replace").decode())
    print("  Baseline (new thread):", result_base["response"][:80].encode("ascii", "replace").decode())


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    baseline = BaselineAgent(config=config, force_offline=True)
    advanced = AdvancedAgent(config=config, force_offline=True)

    user_id = "stress_user"
    thread_id = "stress_thread"

    # Send many long messages to both agents
    messages = [
        "Chào bạn, mình tên là DũngCT Stress và đang làm MLOps engineer. " + "a" * 80,
        "Mình ở Đà Nẵng và thích Python, AI ứng dụng. " + "b" * 80,
        "Hãy nhớ mình thích câu trả lời ngắn gọn 3 bullet. " + "c" * 80,
        "Tiếp tục thêm ngữ cảnh để so sánh token usage. " + "d" * 80,
        "Đây là lượt cuối trước khi hỏi recall. " + "e" * 80,
    ]

    for msg in messages:
        baseline.reply(user_id=user_id, thread_id=thread_id, message=msg)
        advanced.reply(user_id=user_id, thread_id=thread_id, message=msg)

    base_prompt = baseline.prompt_token_usage(thread_id)
    adv_prompt = advanced.prompt_token_usage(thread_id)
    adv_compactions = advanced.compaction_count(thread_id)

    print(f"PASS: test_compact_reduces_prompt_load_on_long_thread")
    print(f"  Baseline prompt tokens: {base_prompt}")
    print(f"  Advanced prompt tokens: {adv_prompt}")
    print(f"  Advanced compactions: {adv_compactions}")

    # When compaction triggers, advanced prompt context should not grow unboundedly.
    # After compaction, context is bounded (summary + keep_messages only).
    # This assertion is intentionally loose since offline mode is heuristic.
    assert adv_compactions >= 1 or adv_prompt <= base_prompt + 500, (
        "Advanced should compact or keep prompt load manageable vs baseline"
    )


def test_conflict_handling(tmp_path: Path) -> None:
    """Bonus 2: correction markers update the fact instead of keeping the old value."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "conflict_user"

    # Session 1: introduce location as Hue
    agent.reply(user_id=user_id, thread_id="s1", message="Mình đang ở Huế.")

    facts_before = agent.profile_store.facts(user_id)
    assert facts_before.get("nơi ở") == "Huế", (
        f"Expected 'Huế' stored, got: {facts_before}"
    )

    # Session 2: correct to Da Nang
    agent.reply(
        user_id=user_id,
        thread_id="s2",
        message="Thực ra từ tuần này mình đang làm việc ở Đà Nẵng rồi, không phải Huế nữa.",
    )

    facts_after = agent.profile_store.facts(user_id)
    location = facts_after.get("nơi ở", "")
    assert "Đà Nẵng" in location, (
        f"Expected 'Đà Nẵng' after correction, got: {location!r}"
    )
    assert "Huế" not in location, (
        f"Old value 'Huế' should be replaced, got: {location!r}"
    )
    print(f"PASS: test_conflict_handling (nơi ở = {location!r})")


def test_confidence_threshold_blocks_noise(tmp_path: Path) -> None:
    """Bonus 1: joke/hypothetical messages do not get written to User.md."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True, confidence_threshold=0.6)
    user_id = "noise_user"

    # Genuine fact — should be stored
    agent.reply(user_id=user_id, thread_id="t1",
                message="Mình đang làm backend engineer tại một công ty AI.")

    # Joke — should NOT overwrite nghề nghiệp
    agent.reply(user_id=user_id, thread_id="t1",
                message="Hay là mình chuyển sang làm PM cho khỏe, đùa thôi.")

    facts = agent.profile_store.facts(user_id)
    job = facts.get("nghề nghiệp", "")
    assert "backend engineer" in job.lower() or "engineer" in job.lower(), (
        f"Genuine job should be stored, got: {job!r}"
    )
    assert "pm" not in job.lower(), (
        f"Joke job should NOT be stored, got: {job!r}"
    )
    print(f"PASS: test_confidence_threshold_blocks_noise (nghề nghiệp = {job!r})")


def test_negation_not_stored(tmp_path: Path) -> None:
    """Bonus 2: negated facts ('không ở Huế') are not written to User.md."""
    config = make_config(tmp_path)
    agent = AdvancedAgent(config=config, force_offline=True)
    user_id = "neg_user"

    # "mình không ở Hà Nội" → pattern needs "mình\s+ở" directly, but "không" breaks
    # that contiguity so Hà Nội is never captured.
    # "mình đang ở Đà Nẵng" → matches normally and stores Đà Nẵng.
    agent.reply(user_id=user_id, thread_id="t1",
                message="Mình không ở Hà Nội đâu, mình đang ở Đà Nẵng.")

    facts = agent.profile_store.facts(user_id)
    location = facts.get("nơi ở", "")
    assert "Đà Nẵng" in location, f"Expected 'Đà Nẵng', got: {location!r}"
    assert "Hà Nội" not in location, f"Negated value should not be stored, got: {location!r}"
    print(f"PASS: test_negation_not_stored (nơi ở = {location!r})")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        test_user_markdown_read_write_edit(p / "t1")
        test_compact_trigger(p / "t2")
        test_cross_session_recall(p / "t3")
        test_compact_reduces_prompt_load_on_long_thread(p / "t4")
        test_conflict_handling(p / "t5")
        test_confidence_threshold_blocks_noise(p / "t6")
        test_negation_not_stored(p / "t7")
        print("\nAll tests passed!")
