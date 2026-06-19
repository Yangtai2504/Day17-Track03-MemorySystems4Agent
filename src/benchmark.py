from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for e in expected if e.lower() in answer.lower())
    ratio = hits / len(expected)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.5:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    if not answer or answer.startswith("[") and len(answer) < 30:
        return 0.2
    score = 0.5
    if len(answer) > 20:
        score += 0.2
    if expected:
        hits = sum(1 for e in expected if e.lower() in answer.lower())
        score += 0.3 * (hits / len(expected))
    return min(score, 1.0)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_memory_bytes = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conv in conversations:
        user_id = conv["user_id"]
        thread_id = conv["id"]

        # Feed conversation turns
        for turn in conv.get("turns", []):
            result = agent.reply(user_id=user_id, thread_id=thread_id, message=turn)
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)

        # Ask recall questions in a fresh thread
        recall_thread_id = f"{thread_id}_recall"
        for rq in conv.get("recall_questions", []):
            q = rq["question"]
            expected = rq.get("expected_contains", [])
            result = agent.reply(user_id=user_id, thread_id=recall_thread_id, message=q)
            answer = result.get("response", "")
            total_agent_tokens += result.get("agent_tokens", 0)
            total_prompt_tokens += result.get("prompt_tokens", 0)
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

        # Memory file size (advanced only)
        if hasattr(agent, "memory_file_size"):
            total_memory_bytes += agent.memory_file_size(user_id)

        # Compaction count
        total_compactions += agent.compaction_count(thread_id)

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=total_memory_bytes,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    data = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.3f}",
            f"{r.response_quality:.3f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    return tabulate(data, headers=headers, tablefmt="github")


def _print_analysis(standard_rows: list[BenchmarkRow], stress_rows: list[BenchmarkRow]) -> None:
    print("\n## Phân tích kết quả\n")

    b_std = next((r for r in standard_rows if "Baseline" in r.agent_name), None)
    a_std = next((r for r in standard_rows if "Advanced" in r.agent_name), None)
    b_stress = next((r for r in stress_rows if "Baseline" in r.agent_name), None)
    a_stress = next((r for r in stress_rows if "Advanced" in r.agent_name), None)

    if b_std and a_std:
        print(f"**Standard benchmark:**")
        print(f"- Advanced recall {a_std.recall_score:.3f} vs Baseline {b_std.recall_score:.3f} "
              f"(+{a_std.recall_score - b_std.recall_score:.3f})")
        if a_std.prompt_tokens_processed > b_std.prompt_tokens_processed:
            print(f"- Advanced dùng nhiều prompt token hơn ở hội thoại ngắn "
                  f"({a_std.prompt_tokens_processed} vs {b_std.prompt_tokens_processed}) "
                  f"— do overhead của User.md và summary.")
        print(f"- Advanced compactions: {a_std.compactions}, Memory growth: {a_std.memory_growth_bytes} bytes")

    if b_stress and a_stress:
        print(f"\n**Stress benchmark (long context):**")
        print(f"- Advanced compactions: {a_stress.compactions} lần — compact kéo prompt token xuống")
        diff = b_stress.prompt_tokens_processed - a_stress.prompt_tokens_processed
        print(f"- Prompt token: Baseline={b_stress.prompt_tokens_processed}, Advanced={a_stress.prompt_tokens_processed} "
              f"(Advanced tiết kiệm {diff} tokens so với Baseline)")
        print(f"- Advanced recall: {a_stress.recall_score:.3f} vs Baseline: {b_stress.recall_score:.3f}")
        print(f"\n**Kết luận:**")
        print(f"- User.md persistent memory giúp Advanced cross-session recall cao hơn Baseline")
        print(f"- Ở hội thoại ngắn, Advanced tốn hơn do overhead (User.md + summary injected mỗi lượt)")
        print(f"- Ở hội thoại dài, compact memory tối ưu chủ yếu ở prompt_tokens_processed")
        print(f"- Rủi ro: memory file phình to nếu agent ghi quá nhiều fact; lưu sai fact nếu regex extract kém chính xác")


def _build_report(
    std_rows: list[BenchmarkRow],
    stress_rows: list[BenchmarkRow],
) -> str:
    lines: list[str] = []
    lines.append("# Benchmark Report — Day 17 Track 3: Memory Systems for AI Agent\n")

    lines.append("## Standard Benchmark (`data/conversations.json`)\n")
    lines.append(format_rows(std_rows))
    lines.append("")

    lines.append("## Long-Context Stress Benchmark (`data/advanced_long_context.json`)\n")
    lines.append(format_rows(stress_rows))
    lines.append("")

    b_std = next((r for r in std_rows if "Baseline" in r.agent_name), None)
    a_std = next((r for r in std_rows if "Advanced" in r.agent_name), None)
    b_stress = next((r for r in stress_rows if "Baseline" in r.agent_name), None)
    a_stress = next((r for r in stress_rows if "Advanced" in r.agent_name), None)

    lines.append("## Nhận xét tự động\n")

    if b_std and a_std:
        recall_delta = a_std.recall_score - b_std.recall_score
        prompt_ratio = a_std.prompt_tokens_processed / max(b_std.prompt_tokens_processed, 1)
        lines.append("**Standard benchmark:**")
        lines.append(
            f"- Cross-session recall: Advanced `{a_std.recall_score:.3f}` vs Baseline `{b_std.recall_score:.3f}` "
            f"(+{recall_delta:.3f}) — User.md cho phép nhớ facts qua thread mới."
        )
        lines.append(
            f"- Prompt tokens: Advanced `{a_std.prompt_tokens_processed}` vs Baseline `{b_std.prompt_tokens_processed}` "
            f"(×{prompt_ratio:.1f}) — overhead của việc inject User.md + summary mỗi lượt."
        )
        lines.append(f"- Compactions: Advanced `{a_std.compactions}` — thread ngắn chưa đủ dài để kích hoạt compact.")
        lines.append("")

    if b_stress and a_stress:
        prompt_saved = b_stress.prompt_tokens_processed - a_stress.prompt_tokens_processed
        lines.append("**Stress benchmark (long context):**")
        lines.append(
            f"- Compactions: Advanced `{a_stress.compactions}` lần — compact hoạt động khi thread vượt ngưỡng."
        )
        lines.append(
            f"- Prompt tokens tiết kiệm: `{prompt_saved}` tokens "
            f"(Baseline `{b_stress.prompt_tokens_processed}` → Advanced `{a_stress.prompt_tokens_processed}`)."
        )
        lines.append(
            f"- Agent tokens: Baseline `{b_stress.agent_tokens_only}` vs Advanced `{a_stress.agent_tokens_only}` "
            f"— compact không ảnh hưởng nhiều đến agent tokens, chủ yếu tối ưu prompt context."
        )
        lines.append(
            f"- Recall: Advanced `{a_stress.recall_score:.3f}` vs Baseline `{b_stress.recall_score:.3f}` "
            f"— User.md giúp giữ facts dù thread bị compact."
        )

    return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    outputs_dir = config.base_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"

    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    print("=" * 70)
    print("## Standard Benchmark (conversations.json)")
    print("=" * 70)

    baseline_std = BaselineAgent(config=config, force_offline=True)
    advanced_std = AdvancedAgent(config=config, force_offline=True)

    std_rows = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print(format_rows(std_rows))

    print("\n" + "=" * 70)
    print("## Long-Context Stress Benchmark (advanced_long_context.json)")
    print("=" * 70)

    baseline_stress = BaselineAgent(config=config, force_offline=True)
    advanced_stress = AdvancedAgent(config=config, force_offline=True)

    stress_rows = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print(format_rows(stress_rows))

    _print_analysis(std_rows, stress_rows)

    report = _build_report(std_rows, stress_rows)
    report_path = outputs_dir / "benchmark_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
