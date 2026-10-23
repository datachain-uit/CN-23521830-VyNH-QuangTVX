# Chuẩn bị A2 A4 và C03 matched trên CQ V4

Đây là code và cấu hình chuẩn bị train/dev, không phải kết quả utility hoặc phê duyệt M0/M1. Không chọn winner tự động, không mở test và không sửa ba cấu hình ứng viên đang dùng.

| ID | Đường chạy | Điều giữ nguyên và giới hạn |
|---|---|---|
| `a2_skq_pc_class_only` | Gọi engine SKQ-PC gốc với phase hằng `all_phases`, tương đương chia tầng chỉ theo class | Kế thừa source RFF, geometry, herding, QP, allocation policy/floor và LRQ nếu có. Chỉ bảo toàn class mass, không hứa bảo toàn phase mass. Không fallback khi mất phase metadata. |
| `a4_skq_pc_uniform_weights` | Dùng nguyên source indices; mỗi dòng đã chọn trong tầng nhận `N_h / m_h` | Không gọi QP lại nhưng source selection đã chạy QP. Ghi cả source time và reweight time; đây là ablation utility của trọng số, chưa phải benchmark selector chạy từ đầu không QP. |
| `c03_matched_skq_pc` | Random không hoàn lại theo đúng quota thực tế của source ở từng phase×class | Khớp cả khi source dùng sqrt/floor. Không đoán proportional tương đương. Ghi source dependency riêng; không cộng chi phí nguồn lặp lại khi tổng hợp cùng một workflow. |

Mỗi method cần khai báo `source_method` cụ thể trong method_options và cấu hình của source trong cùng bảng. Ba nguồn được nhận: `skq_pc`, `skq_pc_lrq_sq`, `skq_pc_sqrt`. Không có winner mặc định và không nhận override tham số riêng cho ablation. Runner chặn mọi stage khác s1_screen hoặc test_locked=false trong phạm vi chuẩn bị này; sau freeze cần tích hợp một đường test được review riêng.

Ba template có tên `cq_v4_dev_ablations_from_{source_method}_v1.json`; không sửa đè panel cũ. Mỗi template chạy source trước rồi A2/A4/C03 matched, ba seed 11/29/47, learner XGB natural, rho=0,5%. Việc có ba template không yêu cầu chạy cả ba: sau khi đối chiếu panel và cô chấp thuận, dùng template đúng nguồn đã chốt.

Template đứng riêng có experiment/protocol identity khác panel, nên source được chạy lại trong template. Không giả định tự động reuse artifact khác experiment hoặc bỏ qua fingerprint. Dự toán phải tính thêm ba source selection/fit khi chạy template riêng. Chỉ reuse được nếu có một cơ chế bàn giao artifact rõ ràng, kiểm identity và được kiểm thử riêng.

A2 dùng cùng floor nhưng giảm số tầng; đó là đúng tác động cần kiểm tra, không ép quota A2 theo phase để làm mất ý nghĩa ablation. A4 giữ đúng source rows/quotas và không thay chính sách natural của learner. Source LRQ của A2 tiếp tục dùng query OOF train-only qua runner gốc; AutoCoreset vẫn là đối chứng điều kiện nếu bản chốt giữ LRQ.

Đã sửa cách cộng cost trong runner cho SKQ-PC LRQ (và A2 kế thừa LRQ): OOF seconds trước đó có trong metadata nhưng bị bỏ qua tổng vì method không có structure parent. Tổng mới gồm skq_only + query, có query_cache_hit/query_cost_shared để workflow chỉ tính dependency một lần. Đây là sửa số đo cho lượt chạy mới, không đổi sample/weight/label và không ghi đè cost trong archive lịch sử.

Chỉ gọi lệnh dưới đây sau khi nguồn/tham số/scope dev đã được duyệt và dữ liệu processed CQ V4 đã chuẩn bị:

```text
python -m skq_exp.cli run --config configs/cq_v4_dev_ablations_from_skq_pc_sqrt_v1.json --resume
```

Ví dụ tên sqrt không phải quyết định chọn sqrt. Các test dùng dữ liệu giả kiểm exact budget, class/phase mass đúng scope, same indices A4, quotas C03, source options/LRQ inheritance, guard test, và đường runner lưu metrics bốn phase. Chưa có kết quả ablation CQ V4 thật trong lần chuẩn bị này.
