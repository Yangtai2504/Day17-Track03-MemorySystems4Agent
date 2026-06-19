# Benchmark Report — Day 17 Track 3: Memory Systems for AI Agent

## Standard Benchmark (`data/conversations.json`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                1948 |                     16348 |                  0     |              0.7   |                       0 |             0 |
| Advanced |                4140 |                     34622 |                  0.179 |              0.649 |                    3615 |             0 |

## Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    |   Agent tokens only |   Prompt tokens processed |   Cross-session recall |   Response quality |   Memory growth (bytes) |   Compactions |
|----------|---------------------|---------------------------|------------------------|--------------------|-------------------------|---------------|
| Baseline |                 294 |                     22620 |                  0     |              0.7   |                       0 |             0 |
| Advanced |                 360 |                     11059 |                  0.167 |              0.775 |                     314 |             4 |

## Nhận xét tự động

**Standard benchmark:**
- Cross-session recall: Advanced `0.179` vs Baseline `0.000` (+0.179) — User.md cho phép nhớ facts qua thread mới.
- Prompt tokens: Advanced `34622` vs Baseline `16348` (×2.1) — overhead của việc inject User.md + summary mỗi lượt.
- Compactions: Advanced `0` — thread ngắn chưa đủ dài để kích hoạt compact.

**Stress benchmark (long context):**
- Compactions: Advanced `4` lần — compact hoạt động khi thread vượt ngưỡng.
- Prompt tokens tiết kiệm: `11561` tokens (Baseline `22620` → Advanced `11059`).
- Agent tokens: Baseline `294` vs Advanced `360` — compact không ảnh hưởng nhiều đến agent tokens, chủ yếu tối ưu prompt context.
- Recall: Advanced `0.167` vs Baseline `0.000` — User.md giúp giữ facts dù thread bị compact.