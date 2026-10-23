# Thực nghiệm CQ V4

Tất cả lệnh dưới đây chạy trong `experiments/`. Cấu trúc `src/`, `configs/`, `scripts/`, `tests/`, tên package `skq_exp` và API được giữ từ repo cá nhân. Cơ chế selector gốc không đổi; phần learner/view/endpoint được bổ sung theo review 23/10, tách khỏi config và kết quả lịch sử.

## Cập nhật hiện hành theo review 23 tháng 10

Xem [phạm vi cài đặt và thứ tự chạy](docs/cq_v4/REVIEW_2026_10_23_IMPLEMENTATION.md).
UL chính là **log loss trung bình P1–P4**, nhỏ tốt hơn; Macro-F1 là phụ. Main
view là CD3 trên spine/nhãn/split V4, QA23 và V1-Median train-only. V4 view cũ
chỉ dùng diagnostic V20/V50 và sensitivity, không tham gia chọn L0/L1.

Config mới mang prefix `cq_v4_review2310_`. Bước chạy được trước khi nhận CD3:

```text
python -m skq_exp.cli prepare-external --config configs/cq_v4_review2310_v20_v1.json --input-dir <runtime_dev_only_V4> --dataset course_quality_v4
python -m skq_exp.cli run --config configs/cq_v4_review2310_v20_v1.json
python -m skq_exp.cli run --config configs/cq_v4_review2310_v50_v1.json
```

Không trỏ nhầm V4 ready vào config CD3. CD3 runtime và artifact 23/10 dùng
thư mục mới, giữ canonical row ID cho Parquet; không dùng `--overwrite` lên
runtime lịch sử. Panel và test chưa được phép chạy chỉ vì config đã có.

Các mục hướng dẫn 22/10 bên dưới được giữ để tái lập lịch sử; không dùng
headroom/selection dựa Macro-F1 cũ cho quyết định mới.

## 1. Môi trường và phần cứng

Dùng môi trường đã ghi trong notebook/archive khi cần tái lập. Không nâng/hạ thư viện giữa một lượt chạy.

```text
python -m pip install -e ".[boosting,dev]"
python -m skq_exp.cli doctor
```

CQ V4 đọc Parquet, nên môi trường cần có engine `pyarrow` tương thích với bundle; giữ phiên bản đã ghi trong môi trường thực nghiệm. Khi chạy đủ baseline trên Kaggle/Linux, dùng extra `kaggle-panel` và fetch repo nguồn tương ứng. Không cài JAX/KIP chỉ để chạy CQ V4; hướng dẫn cài JAX legacy đã bỏ khỏi luồng hiện hành.

| Việc | Thiết bị theo cấu hình hiện có |
|---|---|
| Audit, bootstrap, LR, nhiều selector | CPU |
| FullTrain/headroom/panel XGB CQ V4 | Config đã khóa `device=cuda`; dùng runtime GPU tương ứng |
| Baseline cần dependency riêng | Theo adapter/diagnostics, không suy ra dùng GPU từ việc Kaggle cấp T4×2 |

Không tự sửa `device`, số cây hoặc seed để đổi môi trường mà vẫn gọi là cùng config.

## 2. Đầu vào và kiểm tra dữ liệu

Dùng bundle `runtime_dev_only/` của nhóm dữ liệu, có Pool B train, dev P1–P4 và FullTrain-A train. Không dùng root private chứa test làm input của bước dev.

```text
python scripts/audit_cq_v4_dev.py --input-dir <runtime_dev_only> --output artifacts/cq_v4_audit/audit.json
python -m skq_exp.cli prepare-external --config configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json --input-dir <runtime_dev_only> --dataset course_quality_v4
```

Các đường dẫn `data/processed`, `artifacts`, `external/repos` tính từ thư mục hiện tại. Chúng không được commit. Không lấy toàn bộ cột số làm X hoặc fit lại scaler trên dev.

## 3. Cấu hình CQ V4 lịch sử theo review 22 tháng 10

| Công việc | Config/đầu vào |
|---|---|
| FullTrain-A | `cq_v4_m0_fulltrain_a_dev_natural_xgb_v1.json` |
| FullTrain-B | `cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json` |
| Headroom | Bốn file `cq_v4_m0_headroom_c02_rho{005,01,02,05}_natural_v1.json`, năm selector seed |
| LR-B | Tám file `cq_v4_m0_lr_tune_*.json`; natural chính, balanced sensitivity |
| Panel dev | `cq_v4_panel_dev_rho005_seeds11_29_47_v1.json`; chờ dữ liệu/tham số được duyệt |
| QĐ16 | Script calibration nhận rõ dataset V4 và headroom decision, không lấy default V1 |

Ví dụ FullTrain-B sau ingest:

```text
python -m skq_exp.cli run --config configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json --resume
python scripts/compute_cq_v4_dev_endpoints.py --input-dir <runtime_dev_only> --artifacts artifacts --experiment-id cq_v4_m0_fulltrain_b_dev_natural_xgb_v1 --output artifacts/cq_v4_fulltrain_b_endpoints.json
```

Mỗi thiết kế fit một model, dùng lại trên bốn dev phase; không coi bốn phase là bốn model độc lập. FullTrain-A dùng snapshot lambda=1 của cùng cohort train, không phải P4 và không ghép vào Pool B.

Headroom dùng `compute_cq_v4_headroom.py` với bốn option `--c02-config`. LR dùng `select_cq_v1_lr_config.py` với tối đa tám option `--config`; tên script cũ nhưng là code dùng chung đã kiểm với V4, không tự đổi tên/import.

## 4. Baseline và hiệu chuẩn QĐ16

Fetch đúng repo trong lock file. Ví dụ dependency cho bộ regression:

```text
python scripts/fetch_official_repos.py --repo tabular_data_distillation
```

Trước khi chạy baseline khác, fetch đúng key nguồn của method; `--repo` có thể lặp. Các key hiện hành gồm `coretab`, `bdis`, `moderate_ds`, `ccs`, `dataoob`, `goodpoints`. AutoCoreset là đối chứng điều kiện khi giữ LRQ; dependency và thời gian nhánh này phải tính riêng.

```text
python scripts/run_cq_v1_resource_calibration.py --dataset course_quality_v4 --processed-root data/processed --external-root external/repos --headroom-decision docs/cq_v4/m0_headroom_decision_2026_10_22.json --output artifacts/cq_v4_resource_calibration --max-threads 4 --replicates 3 --resume
```

File headroom đi kèm là bằng chứng dev đã tính, không phải văn bản cô đã duyệt. Với lượt dữ liệu mới phải dùng decision/hash tương ứng và xin duyệt thay đổi, không tự dùng lại rho của dữ liệu khác.

Tên script calibration còn V1 là để tương thích; `--dataset course_quality_v4` là bắt buộc trong lệnh V4. Trần 10.800 giây, ba kích thước và ba lặp giữ theo code đã khóa; 800k là workload có hoàn lại nếu lớn hơn train và phải khai báo. TIMEOUT/OOM/dependency lỗi không được thay bằng proxy hoặc xóa khỏi ledger.

## 5. Endpoint và panel

- Theo QĐ25 ngày 23/10: UL chính là mean enrollment log loss P1–P4; Macro-F1 và course-equal là bổ sung. Script endpoint cần `--review-1023` để xuất trường LL chính và LL sau A8; trường `ul_*` trong phần PCC legacy vẫn là F1, không dùng nhầm.
- EC: MAE của Q_hat theo PCC trên course có n≥30; kèm signed error và A8.
- Báo precision/recall/F1/PR-AUC từng lớp, từng phase; không chỉ mean F1.
- Main policy natural; balanced sensitivity không thay main vì điểm dev đẹp hơn.
- Ba ứng viên hiện hành: `skq_pc`, `skq_pc_lrq_sq`, `skq_pc_sqrt`. R không còn là ứng viên thứ tư.
- C03 chính phải cùng allocation/floor với winner; C03 proportional chỉ mô tả nếu không khớp winner.
- BDIS native giữ realized size; thêm C02/SKQ chốt cùng size, không trim/pad rồi vẫn gắn nhãn native.
- A2/A4/C03 matched đã có đường chạy chuẩn bị và kiểm thử; xem `docs/cq_v4/ABLATIONS_PREPARATION.md`. A4 giữ source rows và chi phí QP nguồn, chỉ đo tác động trọng số; không gọi bước reweight là tăng tốc selector không QP. A8 không refit. Chưa có utility panel/ablation hoặc freeze đã được duyệt.
- Giữ test khóa đến freeze M1 được duyệt; không lấy config confirm legacy để mở CQ V4 test.

## 6. Kiểm thử và file giữ lại có chủ đích

```text
python -m pytest tests -q
```

Giữ toàn bộ 200 tests hồi quy ban đầu và thêm tests ablation, không xóa test để rút gọn repo hoặc làm kết quả qua. Vì test hiện đọc một số config V1/Adult/CQ_med và gọi script dùng chung, 17 config ngoài V4 và các module tương ứng được giữ đúng đường dẫn cũ làm fixture/dependency. **Không dùng chúng cho lượt CQ V4 mới.** Danh sách cụ thể và lý do ở `../CURATION.json`.

Một số module source còn chứa method legacy vì registry/factory và regression đang import; bỏ module đó sẽ thành refactor chứ không còn dọn file an toàn. Chỉ danh sách method trong config V4 là panel hiện hành.

Đã bỏ hướng dẫn GitHub/Kaggle tổng quát, report generator/ZIP builder cũ, template freeze sai phiên bản và config không nằm trong closure CQ V4 hoặc regression. Bản nhập ban đầu vẫn nằm trong lịch sử Git, không rewrite hay mất khả năng truy vết.
