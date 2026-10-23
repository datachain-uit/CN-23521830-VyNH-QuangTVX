# SKQ Experiment — execution repository có gate kiểm toán

Project này hiện thực hóa ma trận thực nghiệm cho **Structured Kernel Quadrature
(SKQ)** trên dữ liệu bảng. Raw run trong repo này chỉ là execution artifact; chỉ
run qua gate, audit và export sang evidence store của paper mới được dùng làm số
liệu bài báo.

## 1. Câu hỏi nghiên cứu

Ta có tập train `D={(x_i,y_i)}` và budget `m=round(r×|D|)`. Mục tiêu là tạo một
tập train nhỏ hơn nhưng vẫn giữ:

1. utility trên nhiều downstream learner;
2. phân phối lớp và vùng cấu trúc quan trọng;
3. chi phí chọn, train, lưu trữ có lợi so với FullTrain;
4. khả năng tái lập bằng seed, commit và artifact bất biến.

Phương pháp đề xuất dùng `parent structure × class` để chia nhóm, cấp exact quota,
coarsen micro-strata theo class quota, biểu diễn RBF bằng Random Fourier Features
(RFF), chọn điểm bằng kernel herding và
tìm trọng số không âm tổng bằng khối lượng nhóm qua simplex-QP. P04/P05 ghép thêm
OOF query loss để bảo toàn vùng khó của một hoặc nhiều learner mà không dùng
dev/test. Protocol v3 bổ sung P06–P09 để kiểm tra riêng đóng góp của coverage
Gonzalez, QP và query loss; đây là biến thể đề xuất để screening, không phải
baseline SOTA hay implementation từ một paper khác.
Protocol v4 giữ nguyên P06/P08 và thêm P10/P11 dạng sharded merge-reduce cho
dữ liệu lớn: mọi dòng train vẫn được đọc, nhưng parent và RFF/QP chỉ giữ một
shard/nhóm trong bộ nhớ; không dùng proxy dataset.

Test không được dùng để chọn method hoặc siêu tham số. Config pilot hiện đánh giá
trên **dev**; kết quả đó chỉ phục vụ kiểm tra code và screening.

## 2. Panel đã được code

Config hiện hành
[`pilot_adult_full_seed11_v2.json`](configs/pilot_adult_full_seed11_v2.json) có
22 dòng, tương ứng 110 ô method × learner ở một seed và lên kế hoạch đủ 110 ô.
Control Stratified Random được mở cho đủ năm learner bằng override nằm trong
config/protocol hash. Config v1 được giữ lại để đọc đúng artifact cũ có bốn ô
`NA_CONTRACT`; thiếu dependency, vượt RAM hoặc vượt time gate vẫn được ghi
`BLOCKED/PREDICTED_OOM/PREDICTED_TIMEOUT`, không bị xóa khỏi mẫu số. Tên ngắn
`pilot_adult_seed11.json` là alias của full-v2.

Protocol mở rộng
[`pilot_adult_full_seed11_v3.json`](configs/pilot_adult_full_seed11_v3.json) giữ
nguyên 22 dòng v2 và thêm P06–P09, thành 26 dòng/130 ô ở một seed. Tách config v3
giúp artifact v2 cũ vẫn tái lập đúng protocol hash; không được trộn kết quả hai
protocol như thể chúng là cùng một lượt chạy.
Config
[`pilot_adult_full_seed11_v4.json`](configs/pilot_adult_full_seed11_v4.json)
giữ nguyên toàn bộ v3 và thêm P10/P11, thành 28 dòng/140 ô ở một seed. P10/P11
là proposed adapter để kiểm tra khả năng mở rộng, không phải native CoreTab/BDIS.
Config [`pilot_course_quality_med_seed11_v3.json`](configs/pilot_course_quality_med_seed11_v3.json)
mở cùng portfolio 26 dòng cho CourseQuality ở vòng dev screen. Chỉ họ winner đã
freeze mới được chuyển sang cả bốn temporal test; không dùng test để chọn giữa
parent CoreTab và Gonzalez. Nhãn evidence `integration_debug_split_overlap` phải
được giữ vì split do dataset cung cấp có nguồn train/validation trùng như đã
audit, nên chưa phải confirmatory.

| Nhóm | ID | Implementation và cách hiểu |
|---|---|---|
| Reference | `c00_full_train` | Train toàn bộ; không phải selector |
| Control | `c02_stratified_random` | Exact 5%; full-v2 chạy đủ LR/RF/XGB/CAT/MLP, confirmatory vẫn có thể khóa riêng |
| Native | `n01_coretab_dt_subset` | Gọi trực tiếp CoreTab-DT commit khóa; giữ native size |
| Native | `n02_coretab_xgb_subset` | Gọi trực tiếp CoreTab-XGB commit khóa; giữ native size |
| Native | `n_bdis_native` | Gọi BDIS upstream; bắt buộc `faiss`, không sklearn fallback |
| Native | `n_autocoreset_native` | Nhập artifact do driver environment riêng tạo |
| Adaptation | `a_craig_feature_space` | Facility-location/lazy-greedy feature-space; R1, không claim native |
| Official-source adapter | `n_gcoreset_benchmark` | Gọi trực tiếp `distill_coreset` từ benchmark Tabular Data Distillation; adapter chỉ seed và khôi phục index |
| Official-source adapter | `n_leverage_benchmark` | Gọi trực tiếp `distill_coreset_leverage_scores`; adapter inject import PCA/seed |
| Synthetic | `s_kip_tdbench` | Gọi KIP trong TDBench; lưu `generated_train.npz` |
| Synthetic | `s_mtt_tdbench` | Gọi trajectory matching trong TDBench |
| Synthetic | `s_gm_tdbench` | Gradient Matching (ICLR 2021) qua TDBench; patch gradient được khai báo |
| Synthetic | `s_datm_tdbench` | Difficulty-Aligned Trajectory Matching (ICLR 2024) qua TDBench; patch được khai báo |
| Synthetic | `s_tame_official` | Gọi `tame_synthesize` từ TAME; GPU/`max_rows` gate rõ ràng |
| Proposed | `p01_skq_coretab_dt` | SKQ dùng structure do N01 xuất |
| Proposed | `p02_skq_coretab_xgb` | SKQ dùng structure do N02 xuất |
| Proposed | `p03_skq_bdis_filtered` | SKQ trên candidate/structure BDIS; chỉ mở khi BDIS gate qua |
| Proposed | `p04_skq_lrq_sq` | P02 structure + OOF loss của LR |
| Proposed | `p05_skq_lrq_mq` | P02 structure + OOF loss LR/RF/XGB |
| Proposed screen | `p06_gonzalez_qp` | Candidate Gonzalez đúng 5%; Voronoi theo 128 Gonzalez anchor + simplex-QP, giữ nguyên tập index để cô lập đóng góp weighting |
| Proposed screen | `p07_skq_gonzalez` | Gonzalez tạo pool 10%, SKQ nén về exact 5% bằng RFF/herding/QP |
| Proposed screen | `p08_skq_gonzalez_lrq_sq` | P07 + OOF query loss của LR |
| Proposed screen | `p09_skq_gonzalez_lrq_mq` | P07 + OOF query loss LR/RF/XGB |
| Proposed scale-out | `p10_skq_mr_coretab_xgb` | CoreTab-XGB theo shard, gộp leaf nhỏ, SKQ streaming và exact budget toàn cục; không proxy |
| Proposed scale-out | `p11_skq_mr_bdis` | BDIS candidate theo shard, SKQ streaming; cap theo union candidate, không pad/trùng |
| Ablation | `d02_parent_structured_random` | Giữ structure/QP, thay herding bằng random trong nhóm |
| Ablation | `d04_global_rff_quadrature` | Bỏ parent structure, vẫn giữ class/RFF/herding/QP |
| Ablation | `d05_equal_group_weight` | Giữ index của P05, thay QP bằng trọng số đều trong nhóm |

GM và DATM là baseline deep chính thống về ý tưởng/paper nhưng lượt `full-v2`
dùng profile `pilot_resource_bounded`, không tự nhận là full-paper reproduction.
Artifact lưu toàn bộ tham số chạy và danh sách source patch. Cấu hình tham chiếu
đầy đủ của TDBench lớn hơn đáng kể và chỉ nên chạy ở stage scale riêng.

Không thêm baseline LLM vào ma trận số hiện tại. GReaT/LLM cần DataFrame thô với
tên cột, kiểu categorical và tokenizer/model checkpoint; runner hiện nhận
`X_train.npy` đã mã hóa. Ép LLM học các cột số ẩn danh sẽ sai cơ chế paper và phụ
thuộc tải model ngoài khi Kaggle Internet có thể bị tắt.

`p00_structured_kquad` là engine dùng chung, không phải dòng kết quả. Trong
`s1_screen`, P04/P05 và P08/P09 được chạy trên **dev** với parent đã khai báo trước và
artifact phải ghi `pre_freeze_dev_screen`, `confirmatory_eligible=false`. Đây chỉ
là sàng lọc thăm dò, không phải tuyên bố parent đã thắng. Khi chạy ngoài vòng screen,
đặc biệt `s2_confirm` trên test, các biến thể LRQ vẫn trả `GATE_LOCKED` nếu chưa có freeze
manifest hoặc parent không khớp winner đã freeze.

## 3. Native, benchmark, synthetic khác nhau thế nào

- **Native**: chạy implementation của tác giả đúng commit. Kích thước thực tế
  được giữ nguyên; không trim/pad để giả thành 5%.
- **Official-source adapter**: chạy trực tiếp function của repo tác giả đã khóa
  commit; adapter chỉ chuyển input/output và vá boundary import/seed/index.
- **Synthetic**: sinh X/y mới nên không có `selected_indices.npy`; nó dùng contract
  và storage-matched table riêng.
- **Proposed/ablation**: code trong package này, classwise coarsening, exact-budget,
  mass-preserving weights và invariant tests.

AutoCoreset cần boundary riêng vì upstream phụ thuộc API cũ. BDIS cần Faiss. KIP
cần JAX/neural-tangents; MTT/GM/DATM cần PyTorch. Nếu môi trường không đạt,
ledger nói rõ lý do và tuyệt đối không chạy một thuật toán khác dưới cùng tên.

CoreTab upstream ở commit khóa chỉ có native subset contract cho nhãn nhị phân.
Với dataset đa lớp, N01/N02 được ghi `NA_CONTRACT`; adapter chỉ xuất leaf structure
đa lớp cho P01/P02 và ghi rõ `structure_only=true`, không tạo điểm native CoreTab
giả bằng một OVA tùy ý.

## 4. Cấu trúc project

```text
skq_experiment/
├── configs/                         # protocol/stage, không chứa thuật toán
├── data/raw/                        # raw data, Git ignore
├── data/processed/                  # split + train-only preprocessing, Git ignore
├── external/
│   ├── official_repos.lock.json     # URL + commit khóa
│   └── repos/                       # checkout repo tác giả, Git ignore
├── artifacts/                       # selection, prediction, metric, ledger
├── requirements/                    # base, Kaggle và native-isolated
├── scripts/                         # fetch repo, driver AutoCoreset, Kaggle entry
├── src/skq_exp/
│   ├── data/                        # download, split, preprocessing và manifest
│   ├── methods/
│   │   ├── native/                  # boundary repo tác giả
│   │   ├── benchmark/               # reproduction benchmark có khai báo
│   │   ├── synthetic/               # contract KIP/MTT/GM/DATM/TAME sinh X/y
│   │   └── proposed/                # RFF, allocation, herding, QP, query loss
│   ├── training/                    # 5 learner, weight contract và metrics
│   ├── experiments/                 # DAG, gate, runner và failure ledger
│   └── reports/                     # bảng long, trạng thái và break-even
└── tests/                            # contract/unit/end-to-end tests
```

Mỗi file Python có phần ghi chú đầu file bằng tiếng Việt, mô tả đầu vào, việc nó
làm và lý do tồn tại. Thuật toán không được viết trong notebook hoặc shell cell.

## 5. Cài và kiểm tra

PowerShell:

```powershell
cd C:\source\paper\skq_experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[kaggle-full,data,dev]"
python scripts\fetch_official_repos.py
pytest -q
skq doctor
skq smoke --config configs\s0_smoke.json
```

`fetch_official_repos.py` kiểm tra đúng bảy commit trong lock file. Không commit
các checkout này lên GitHub.

## 6. Chạy Adult một seed

Tải và chuẩn bị dữ liệu:

```powershell
skq fetch-data --config configs\pilot_adult_full_seed11_v2.json --dataset adult_uci2_v1
skq prepare --config configs\pilot_adult_full_seed11_v2.json --dataset adult_uci2_v1
```

Kiểm tra nhanh trước (FullTrain-LR + D04-LR):

```powershell
skq run --config configs\pilot_adult_quick_seed11.json `
  --selector-seed 11 --max-ram-gb 8 --timeout-seconds 1800 --max-threads 4
```

Chạy toàn bộ candidate matrix trên Adult, seed 11:

```powershell
skq plan --config configs\pilot_adult_full_seed11_v2.json
skq run --config configs\pilot_adult_full_seed11_v2.json `
  --selector-seed 11 --model-seed 42 `
  --max-ram-gb 8 --timeout-seconds 3600 --max-threads 4 `
  --max-estimated-operations 30000000000
```

Đây mới là lệnh **full-v2**. Nó đi qua cả 22 method và 5 learner; không phải lệnh
2-method quick check. `skq plan` phải báo 21 phương pháp nén/sinh, 110 ô tổng và
110 ô được lên kế hoạch chạy. P04/P05 chạy ở dev dưới nhãn pre-freeze; chỉ
confirmatory/test mới giữ `GATE_LOCKED` cho tới khi base winner đã freeze.
Dependency bị thiếu không được coi là kết quả cuối. Chạy riêng
một ô để debug:

```powershell
skq run --config configs\pilot_adult_full_seed11_v2.json `
  --method p02_skq_coretab_xgb --learner lr --selector-seed 11
```

Các method phụ thuộc structure phải chạy source trước nếu chạy riêng: N01 trước
P01; N02 trước P02/P04/P05/D02; BDIS trước P03. Full config đã sắp đúng thứ tự và
structure artifact cũng có thể được nạp lại từ ổ đĩa.

Sau khi seed 11 ổn, thêm seed mà không sửa code:

```powershell
skq run --config configs\pilot_adult_full_seed11_v2.json `
  --selector-seed 11 --selector-seed 29 --selector-seed 47
```

## 7. Điều gì được lưu và có nặng máy không

Mặc định `save_models=false`. Một run chọn dòng lưu:

```text
manifest.json
selected_indices.npy
sample_weights.npy
diagnostics.json
timings.json
predictions.npz
metrics_overall.json
metrics_per_label.csv
metrics_all.json
confusion_matrix.csv
cost.json
```

Synthetic thay hai file selection bằng `generated_train.npz`. C00 chỉ lưu
full-reference manifest, không tạo `selected_indices.npy` giả. Model chỉ được lưu
khi thêm `--save-model`; đây mới là loại artifact dễ phình lớn. Prediction được
nén và Adult nhỏ, nên phần mặc định chủ yếu là metric/index chứ không phải hàng
chục checkpoint.

Guard bảo vệ máy gồm:

- preflight RAM và số phép tính;
- hard resource check giữa các batch/vòng của SKQ và Gonzalez;
- CRAIG tính cả ma trận pairwise O(n²) trong dự báo RAM;
- KIP tính cả kernel target-support, gradient và optimizer trong dự báo RAM;
- KIP/MTT/GM/DATM không có trần budget tùy ý; OOM/timeout thật được ghi tách biệt;
- Trên Kaggle, `scripts/run_kaggle_split_env.py` chạy panel trước, sau đó
  pin đồng bộ JAX/CUDA 0.4.38 và chạy riêng KIP trong process mới; hai
  pha vẫn dùng cùng protocol hash và được đóng gói chung;
- trạng thái dừng được ghi vào manifest/ledger.

Repo ngoài không phải method nào cũng có điểm kiểm tra giữa vòng; với chúng,
preflight là lớp bảo vệ chính. Requested size và realized size đều được lưu để
không đánh đồng giới hạn khả thi của source với lỗi tài nguyên.

## 7a. Chạy thử CourseQuality MED

Snapshot Kaggle `hoangzyyng/cq-med` đã có train, validation và bốn temporal test
snapshot nên không được đưa qua splitter của các dataset UCI. Lệnh sau giữ
`val_med.csv` làm dev và lưu riêng đủ bốn phase:

```bash
skq prepare-external \
  --config configs/pilot_course_quality_med_quick_seed11.json \
  --dataset course_quality_med_v1 \
  --input-dir /kaggle/input/datasets/hoangzyyng/cq-med
skq run --config configs/pilot_course_quality_med_quick_seed11.json
```

Quick config chỉ chạy FullTrain-LR và StratifiedRandom-LR để kiểm tra schema trên
toàn bộ 2.637.700 dòng train. Nó fit mỗi learner đúng một lần rồi đánh giá cùng
model trên đủ `test_phase1` đến `test_phase4`; mỗi phase có run/metric riêng và
`temporal_summary.json` báo mean/worst. Config
`pilot_course_quality_med_temporal_seed11_v4.json` chuyển đầy đủ portfolio Adult
hiện hành gồm 28 method × 5 learner sang CQ. Các method bậc hai/deep có thể bị
resource gate trên tập lớn này; đó là trạng thái thực, không được thay bằng
fallback.

Audit snapshot hiện có overlap user-course giữa train/dev/test. Quick temporal
run mang `evidence_role=integration_debug_split_overlap_temporal`; metric dùng để
kiểm tra pipeline, chưa được đưa vào bảng confirmatory. Full config hiện vẫn là
`s1_screen` nên chỉ đọc dev để chọn/freeze phương pháp. Sau khi tạo
`artifacts/freeze/freeze_manifest.json` từ screen của chính CourseQuality, chạy
`pilot_course_quality_med_temporal_seed11_v4.json` để đánh giá ma trận đầy đủ
trên cả bốn test. Config temporal yêu cầu freeze manifest; P04/P05 và P08/P09
không được mở test trước bước này. Khi stage là `s2_confirm` hoặc `s4_temporal`,
runner bắt buộc CourseQuality có và đánh giá đủ bốn test.

Nếu dùng cấu hình đã chốt từ Adult thay vì chọn lại trên CQ, tạo freeze chuyển
giao **trước khi mở CQ test** bằng `skq freeze` và ghi
`selection_basis=transferred_full_adult_v4_portfolio_predeclared_before_cq_test`
và khóa cả `n02_coretab_xgb_subset` lẫn `gonzalez_pool_2x` bằng các option
`--allowed-parent-source`. Đây vẫn chỉ là evidence integration vì snapshot CQ
hiện có overlap. Lệnh Kaggle full dùng
`scripts/run_kaggle_split_env.py --prepare-autocoreset`: 28 method × 5 learner ×
4 test tạo 560 ledger row. Method vượt tài nguyên được giữ dưới trạng thái
`predicted_oom`/`predicted_timeout`; không đổi thuật toán hoặc hạ budget 5%.

Các dataset ngoài dùng chung lệnh `prepare-external`. Mỗi dataset có một module
trong `src/skq_exp/data/adapters/`, kế thừa contract base hoặc khung CSV pre-split
và được đăng ký tường minh trong adapter registry. Mọi dataset ngoài chỉ dùng
`prepare-external`; không duy trì một lệnh hoặc module riêng lặp lại cho từng
dataset. Hướng dẫn và skeleton nằm trong
[`docs/DATASET_ADAPTERS.md`](docs/DATASET_ADAPTERS.md).

## 8. Metrics

Mỗi prediction tạo hơn 25 metric tổng thể: accuracy, balanced accuracy,
precision/recall/F1/Jaccard macro-micro-weighted, MCC, Cohen kappa, quadratic
kappa, worst-class F1/recall, ROC-AUC và PR-AUC macro/weighted, log-loss, Brier và
ECE-15. Mỗi label có support, prevalence, TP/FP/TN/FN, precision, recall,
specificity, NPV, F1/F2, FPR/FNR, MCC, kappa, ROC-AUC và PR-AUC.

`cost.json` tách selection/generation, RFF/herding/QP, fit, predict, total, CPU,
RSS lower bound, VRAM, I/O, throughput, compression ratio và storage artifact.
`skq aggregate` ghép FullTrain để tính break-even reuse khi fit coreset thực sự
nhanh hơn FullTrain.

```powershell
skq aggregate --artifact-root artifacts
```

Đầu ra chính:

- `artifacts/aggregate/runs_long.csv`;
- `artifacts/aggregate/status_summary.csv`;
- `artifacts/<experiment_id>/run_ledger.json`.

## 9. Trạng thái kết quả pilot cũ

Archive Adult seed 11 trước commit sửa contract chỉ là
`integration_debug_adult_seed11`. Các run P01/P02/P04/P05 cũ đã bị vô hiệu hóa vì
allocation trước đây bỏ mass của zero-quota strata. Phải chạy lại bằng schema v3;
không được dùng các metric cũ để chọn winner hoặc viết bảng.

Structure schema v3 bắt buộc có `row_ids`, `stratum_ids`, `candidate_mask` cùng
dataset/split fingerprint. Artifact schema cũ bị từ chối khi replay.

## 10. Quy tắc trước khi viết bảng bài báo

1. Chọn representative/base winner chỉ bằng dev trên đủ dataset/seeds.
2. Freeze method, hyperparameter, query portfolio, contrasts và config hash.
3. Chỉ sau đó tạo `artifacts/freeze/freeze_manifest.json` và mở `s2_confirm`.
4. Confirmatory chính: budget 5%, XGBoost, Macro-F1, MCC, worst-class và Holm.
5. Transfer: LR/RF/XGB/CatBoost/MLP với cùng selection artifact.
6. Native-size và exact-5% phải nằm ở cột/figure khác nhau.
7. Synthetic báo cả row-count-matched và storage-matched.
8. Failure/OOM/timeout/NA phải ở lại bảng.

Trong pilot, MLP chạy số epoch cố định và không dùng dev để early-stop. Khi bổ
sung tuning chính thức, inner-validation phải được tách từ train trước selector,
freeze hyperparameter trên FullTrain rồi dùng lại cho mọi method.

Hướng dẫn đưa code lên GitHub và chạy Kaggle nằm trong
[`GUIDELINE_GITHUB_KAGGLE.md`](GUIDELINE_GITHUB_KAGGLE.md).

## 11. Nguồn quyết định khoa học

- `C:\source\paper\lý thuyết\GOI_THUC_NGHIEM_NEN_V1\docs\19_MASTER_RESEARCH_AND_IMPLEMENTATION_SPEC.md`
- `C:\source\paper\lý thuyết\nhận xét\implementation_plan.md`
- `C:\source\paper\outputs\experiment_matrix_20261006\MA_TRAN_THUC_NGHIEM_SKQ_FULL.xlsx`

Nếu code và protocol đã freeze mâu thuẫn, dừng run và ghi deviation; không sửa
thiết lập sau khi nhìn test.
