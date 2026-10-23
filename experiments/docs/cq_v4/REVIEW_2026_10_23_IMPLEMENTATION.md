# Thực hiện giao thức Review 23 tháng 10

Phần bổ sung này dùng khung V4 đã khóa và feature CD3 theo QĐ28. Không thay
selector SKQ-PC, nhãn, cohort hay split. Config 22/10 được giữ để tái lập lịch
sử, không dùng để quyết định learner/rho theo giao thức mới. Không mở test.

## Phần đã cài

| Phần | Implementation | Phạm vi |
|---|---|---|
| Mã cô gửi | `data/cq_patch_1023.py`, `data/cq_tempo_features.py` trong package skq_exp | Patch trích Phụ lục A; feature helper lấy file kèm review; giữ công thức gốc |
| CD3 raw→ready | `data/cq_cd3_views.py`, `scripts/prepare_cq_cd3_dev.py` | Kiểm spine với V4 gốc, train-only QA23/V1-Median, 74 numeric trước drop, metadata riêng, dev-only và fresh output |
| L0/L1/S | `experiments/cq_review_1023.py`, runner/config | L1 ghép theo hàng thành một eval_set; S chỉ train/eval P4; không trộn S/V vào chọn L0/L1 |
| LR | `data/cq_lr_views.py` | One-hot gender fit toàn Pool B train trước selection; rare/unseen→OTHER; NULL riêng; numeric V1 không đổi |
| Y25 | runner/evaluator/adapter | Parquet p_E/G/A float32, row_id gốc + runtime_row_id, phase, y và khóa nối; storage guard không được bỏ prediction rồi vẫn báo thành công |
| D13 diagnostics | evaluator, runtime helper, endpoint script | best_iteration, fitted/used rounds và số cây, curve ở 50/100/200/400 round/best nếu có, calibration, subgroup user, teacher nếu có mapping, gain của mã chỉ V |
| QĐ25/QĐ26 | `experiments/cq_decisions_1023.py` và script headroom mới | LL nhỏ tốt hơn; đủ 3 model seed/5 selector seed/4 budget; margin và rho chỉ đề xuất |
| CI và đóng góp course | `scripts/compare_cq_review2310.py` | Ghép row ID hai run; 2.000 family bootstrap cùng lượt qua bốn phase; top ba course, không đổi estimand thành family-equal |
| Chặng sau | `scripts/materialize_cq_review2310_configs.py` | Sinh grid 6 + min_child 2, rồi A/B/headroom/D04 sau khi có lựa chọn đo thực; không sinh panel/test |

Mọi đường dẫn module trên tính từ `src/skq_exp/` nếu không ghi `scripts/`.

## Cấu hình ban đầu

14 config mới mang prefix `cq_v4_review2310_`: V20, V50, L0, L1, S,
A-reference và 8 LR. XGB khai báo `device=cuda`; LR/preparation/aggregation dùng
CPU. T4×2 không có nghĩa code đang phân tán training trên hai GPU. Khi đổi
device cần lưu config/runtime thực, không sửa thông tin của lượt cũ.

- V20/V50 dùng `data/processed_review2310_v4` mới, cần ingest lại **cùng** bundle
  V4 dev-only để giữ canonical row_id; không đổi feature hay fit lại preprocessing.
- CD3 dùng `data/processed_cd3`; artifact mới vào `artifacts/review2310`.
- A-reference là mốc chuẩn bị để so với S, phải xác nhận cách dùng comparator
  trong báo cáo; không lấy A lịch sử V4 so với S CD3 rồi tự đánh PASS.
- LR natural là nhóm lựa chọn, balanced là sensitivity với A8.

## Thứ tự chạy

1. Kiểm dữ liệu V4 dev-only và quyền truy cập; chuẩn bị runtime V4 mới bằng
   config V20. Có thể chạy V20 và V50 trước khi nhận raw CD3. Hai lượt có cùng
   feature/params/model seed 42, chỉ khác patience 20/50.
2. Khi có CD3: `prepare_cq_cd3_dev.py --raw-root ... --v4-bundle ...
   --output-bundle ... --processed-root ...`. Raw dùng layout `raw_features/`
   đã thống nhất; original V4 bundle dùng để đối chiếu row/cohort/nhãn/cutoff.
   Cả hai input phải dev-only và có SHA256SUMS. Output phải là đường dẫn mới.
3. Chạy L0/L1/S và A comparator phù hợp. Xuất endpoint bằng
   `compute_cq_v4_dev_endpoints.py --review-1023` trên đúng bundle/view/run mới.
4. So L1−L0, V−L0 và A−B bằng script compare trên artifact đã giải nén; S chỉ
   có P4 nên không được đưa vào bootstrap giả bốn phase. Kiểm điều kiện dừng S
   theo review và ghi subgroup còn thiếu, không bỏ qua vì điểm trung bình đẹp.
5. Chọn L0/L1 bằng `select_learner_view(..., order=[L0,L1], tol=0.002)` của cô.
   Khi có config chọn và evidence thật, materialize stage `grid`; chọn theo LL,
   hòa 0,002 ưu tiên fit nhanh. Stage `min-child` thêm 1/10, tổng tối đa 8.
6. Chốt tham số theo rule, lưu bảng lựa chọn; materialize `followups` để có
   FullB seeds 42/43/44, FullA, C02 bốn budget × năm seed và ba D04 shuffle seed.
   D04 giữ model seed 42; shuffle seed nằm trong protocol hash riêng.
7. Chạy LR tám config, chọn natural theo LL/thời gian; balanced+A8 chỉ phụ.
   Headroom dùng script QĐ25/QĐ26 mới, không dùng script legacy dựa F1.
8. Bổ sung CI/E-AUC D04, timing QĐ16 trên CD3 tại rho đề xuất, tài nguyên thực
   và hồ sơ để cô duyệt. Sau duyệt mới panel/ablation, sau freeze mới test.

Với mỗi run cần giữ `metrics_all.json`, `cost.json`, `predictions.parquet/npz`,
`metrics_per_label.csv`, `confusion_matrix.csv`, manifest và retention. Curve
nằm trong `metrics_all.json`; round không được train tới được ghi not_available,
không gắn nhãn 400 cho model ngắn hơn. Một round CQ multiclass có ba cây, các
trường fitted/used/trees được phân biệt.

## Ranh giới nghiệm thu

Code và kiểm thử không thay D06 thủ công hoặc xác nhận cô. Teacher subgroup
thiếu mapping được ghi thiếu, không suy ra từ mã 0. Script headroom scalar
không tự tạo CI; phải ghép CI từ prediction đủ seed theo giao thức, không chỉ
CI một cặp seed rồi gọi là CI toàn headroom. Freeze/panel legacy chưa được
nâng thành đã duyệt cho giao thức 23/10.

Kết quả kiểm thử thực nằm trong hồ sơ `2410/08_Kiem_thu/` trên Drive chung.
Chưa có utility CQ/CD3 mới trong lần chuẩn bị này. Các thay đổi mới phải được
commit/push có chọn lọc trước khi Kaggle clone; không đưa raw, predictions,
archive hay token lên GitHub.
