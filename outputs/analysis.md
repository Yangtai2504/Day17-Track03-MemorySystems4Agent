# Phân tích kết quả — Day 17 Track 3: Memory Systems for AI Agent

## Tổng quan

Bài lab so sánh hai agent trên cùng bộ dữ liệu tiếng Việt:

- **Baseline Agent**: chỉ có short-term memory trong một thread duy nhất.
- **Advanced Agent**: ba lớp memory — short-term (in-thread), persistent (`User.md`), và compact memory (nén lịch sử dài).

Mục tiêu không phải là "agent nhớ nhiều hơn thì tốt hơn", mà là hiểu rõ **trade-off** giữa recall, token cost, và độ phức tạp của hệ thống.

---

## Kết quả benchmark

### Standard Benchmark (`data/conversations.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|------------------------|-------------|
| Baseline | 1948              | 16 348                  | 0.000                | 0.700            | 0                      | 0           |
| Advanced | 4140              | 34 622                  | 0.214                | 0.660            | 3 633                  | 0           |

### Long-Context Stress Benchmark (`data/advanced_long_context.json`)

| Agent    | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|----------|-------------------|-------------------------|----------------------|------------------|------------------------|-------------|
| Baseline | 294               | 22 620                  | 0.000                | 0.700            | 0                      | 0           |
| Advanced | 360               | 11 065                  | 0.167                | 0.775            | 319                    | 4           |

---

## Câu chuyện 5 bước

### Bước 1 — Baseline không nhớ dài hạn

Baseline đạt cross-session recall = **0.000** ở cả hai benchmark. Khi hỏi "Mình tên gì?" trong một thread mới, agent không thể trả lời vì toàn bộ lịch sử bị bỏ lại ở thread cũ. Đây là hành vi đúng thiết kế — Baseline là mốc so sánh công bằng, không phải một agent bị lỗi.

### Bước 2 — Advanced thêm `User.md` nên recall tăng

Advanced đạt recall = **0.214** ở standard benchmark nhờ `User.md`. Mỗi khi người dùng đề cập tên, nơi ở, nghề nghiệp, hay style trả lời, agent trích xuất fact và ghi vào `User.md` tương ứng với `user_id`. Khi mở thread mới, agent đọc lại `User.md` và inject vào context trước khi trả lời.

Đây là lý do recall tăng mà **không cần giữ nguyên lịch sử hội thoại** — thông tin được chuẩn hóa thành profile thay vì nhét nguyên raw text.

### Bước 3 — Hội thoại dài làm prompt cost của Baseline tăng mạnh

Ở stress benchmark, Baseline xử lý **22 620 prompt tokens** trong khi toàn bộ agent tokens chỉ là **294**. Lý do: mỗi lượt reply, Baseline phải kéo theo toàn bộ lịch sử thread tích lũy từ đầu. Với thread 15 turns dài, prompt context tăng tuyến tính theo số lượt — một vấn đề nghiêm trọng trong production.

### Bước 4 — Compact memory kéo prompt cost của Advanced xuống

Advanced thực hiện **4 compactions** trong stress benchmark và chỉ xử lý **11 065 prompt tokens** — tiết kiệm **11 555 tokens** so với Baseline (giảm ~51%). Cơ chế:

1. Khi tổng token của thread vượt ngưỡng (`COMPACT_THRESHOLD_TOKENS`), `CompactMemoryManager` lấy các messages cũ và nén thành một đoạn `summary`.
2. Chỉ giữ lại `COMPACT_KEEP_MESSAGES` messages gần nhất dưới dạng đầy đủ.
3. Các lượt sau, context được xây dựng từ: `User.md` + `summary` + recent messages — thay vì toàn bộ lịch sử thô.

**Tại sao compact chủ yếu tối ưu `prompt tokens processed` chứ không phải `agent tokens only`?**

`Agent tokens only` là số token agent *sinh ra* — phụ thuộc vào độ dài câu trả lời, không phụ thuộc nhiều vào độ dài context đầu vào. Ngược lại, `prompt tokens processed` là lượng context agent phải *đọc qua* mỗi lượt. Compact giảm phần này bằng cách thay thế nhiều messages dài bằng một summary ngắn. Agent tokens gần như không thay đổi vì output vẫn là câu trả lời tương đương về độ dài.

Điều này có nghĩa: **compact memory không giảm chi phí sinh text, nó giảm chi phí đọc context**. Đây là phần tốn kém nhất khi thread trở nên dài.

### Bước 5 — Hệ thống mạnh hơn nhưng phức tạp hơn và cần guardrail tốt hơn

Advanced vượt Baseline về recall và prompt efficiency ở thread dài, nhưng đi kèm các rủi ro mới.

---

## Tại sao Advanced có thể tốn hơn Baseline ở hội thoại ngắn

Nhìn vào standard benchmark: Advanced xử lý **34 622 prompt tokens** — gấp **2.1 lần** Baseline (16 348). Lý do là overhead cố định mỗi lượt:

- `User.md` được đọc và inject vào system prompt mỗi turn, ngay cả khi người dùng chỉ nói "Chào".
- Khi có compact summary, summary đó cũng được inject, cộng thêm với recent messages.

Ở hội thoại ngắn (ít turns), overhead này lớn hơn lợi ích mà compact mang lại vì compact chưa kịp kích hoạt. Đây không phải bug — đây là trade-off có chủ đích: Advanced đánh đổi **chi phí per-turn cao hơn** để đổi lấy **recall tốt hơn và prompt cost thấp hơn ở thread dài**.

> Trong production, điểm hòa vốn (breakeven) nằm ở khoảng số turns đủ để compact bù lại overhead của User.md injection. Với ngưỡng 800 tokens và 4 messages giữ lại, điểm này xảy ra sau khoảng 6–10 turns tùy độ dài message.

---

## Ba lớp memory: phân biệt rõ vai trò

| Lớp | Nơi lưu | Phạm vi | Mục đích |
|-----|---------|---------|---------|
| **Short-term** | `SessionState.messages` (in-memory) | Trong một thread | Giữ context hội thoại hiện tại, không persist khi process tắt |
| **Persistent** | `User.md` (file disk) | Xuyên session, xuyên thread | Lưu facts ổn định về người dùng: tên, nơi ở, nghề, style |
| **Compact** | `CompactMemoryManager.state` (in-memory) + summary text | Trong một thread, nhưng nén lại | Giảm chi phí prompt khi thread dài, giữ gist thay vì raw text |

**Baseline chỉ có lớp 1.** Khi thread mới bắt đầu, SessionState trống hoàn toàn.

**Advanced có cả ba lớp.** Lớp 2 (User.md) hoạt động xuyên thread vì ghi ra file disk và đọc lại theo `user_id`. Lớp 3 (compact) hoạt động trong thread nhưng giới hạn kích thước context.

---

## Rủi ro và guardrail cần thiết

### Rủi ro 1: Memory file phình to

Nếu agent ghi mọi thứ người dùng nói vào `User.md`, file sẽ lớn dần không kiểm soát. Mỗi lượt inject toàn bộ file vào context sẽ xóa đi lợi ích của compact memory.

**Guardrail**: Chỉ ghi facts ổn định (tên, nơi ở, nghề, style). Không ghi thông tin tạm thời (tin tức đang đọc, việc đang làm hôm nay). Thực tế bài lab dùng regex patterns giới hạn các category được phép ghi.

### Rủi ro 2: Lưu sai fact từ câu hỏi hoặc nhiễu

Nếu người dùng hỏi "Bạn có biết Hà Nội không?", regex kém có thể extract "Hà Nội" thành nơi ở của người dùng. Tương tự, câu đùa "Hay là mình chuyển sang làm PM cho khỏe" có thể bị ghi nhận là nghề nghiệp mới.

**Guardrail hiện có**: `extract_profile_updates` skip mọi message kết thúc bằng `?` hoặc bắt đầu bằng từ hỏi. Pattern tên yêu cầu capital letter để tránh bắt common words.

**Guardrail cần thêm** (bonus): confidence threshold và conflict detection — xem phần tiếp theo.

### Rủi ro 3: Compact làm mất thông tin quan trọng

Nếu summary quá ngắn, các fact quan trọng từ messages cũ bị mất. Agent trông như còn nhớ (vì có summary) nhưng thực ra đã mất chi tiết. Đây là lỗi khó phát hiện vì agent vẫn trả lời được, chỉ là trả lời thiếu chính xác.

**Guardrail**: Facts quan trọng nên được đưa vào `User.md` trước khi compact xảy ra, không nên chỉ tồn tại trong messages. Compact chỉ nên làm mất thông tin *tạm thời*, không làm mất *profile facts*.

---

## Bonus: Conflict Handling và Confidence Threshold

*(Xem chi tiết implementation trong `src/memory_store.py`)*

### Conflict Handling

**Vấn đề giải quyết**: Người dùng thường chỉnh sửa thông tin đã cung cấp trước đó ("thực ra mình không ở Huế mà đang làm ở Đà Nẵng"). Nếu không phát hiện correction, `User.md` sẽ giữ cả fact cũ lẫn fact mới — gây mâu thuẫn và trả lời sai.

**Cách hoạt động**: Phát hiện các marker ngôn ngữ như "thực ra", "sửa lại", "không phải ... mà là", "nhưng thực ra". Khi gặp, agent dùng `upsert_fact` thay thế fact cũ thay vì cộng thêm.

**Cải thiện**: Recall chính xác hơn trong stress benchmark — agent trả lời "Đà Nẵng" thay vì "Huế" hoặc cả hai. Không tốn thêm token vì vẫn dùng cùng `upsert_fact`.

**Rủi ro mới**: Nếu conflict detection quá nhạy, agent có thể xóa fact đúng khi người dùng chỉ nói hypothetical ("nếu mình ở Hà Nội thì sao"). Cần scope marker detection hẹp và test kỹ edge case.

### Confidence Threshold

**Vấn đề giải quyết**: Regex extraction không chắc chắn 100%. Một số matches là false positive — đặc biệt với messages ngắn hoặc cấu trúc không rõ ràng. Ghi nhầm fact vào `User.md` tệ hơn không ghi vì khó phát hiện và sửa.

**Cách hoạt động**: Mỗi pattern được gán confidence score (0.0–1.0) dựa trên độ dài match, tính cụ thể của pattern, và context của message. Chỉ ghi vào `User.md` khi score ≥ ngưỡng (mặc định 0.6).

**Cải thiện**: Giảm false positive, `User.md` chứa ít entries hơn nhưng chính xác hơn. File size nhỏ hơn → overhead inject nhỏ hơn → tiết kiệm prompt tokens về dài hạn.

**Rủi ro mới**: Ngưỡng quá cao → bỏ sót facts đúng (false negative). Agent có recall thấp hơn với người dùng nói ngắn gọn. Cần calibrate ngưỡng theo style người dùng, không dùng một giá trị cố định cho mọi trường hợp.
