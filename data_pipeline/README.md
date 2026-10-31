# Pipeline dữ liệu CQ V4

Mã nguồn của nhóm dữ liệu để chuyển raw MOOCubeX thành bảng chất lượng, nhãn, đặc trưng theo phase, FullTrain-A và bundle bàn giao V4. Thư mục này giữ cấu trúc các script/notebook đã phát triển; không chứa dataset, model, cache hoặc kết quả thực nghiệm.

Nguồn dữ liệu: [MOOCCubeX của THU KEG](https://github.com/THU-KEG/MOOCCubeX); Yu và cộng sự, “MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs”, CIKM 2021, tr. 4643–4652. Repo công khai chỉ bàn giao mã và tài liệu tổng hợp theo QĐ40. Xem [hồ sơ xử lý Review 31/10](GITHUB_REVIEW_2026_10_31.md) để biết trạng thái gỡ PR #5/LFS và niêm phong test.

Trước freeze M1 được duyệt, chỉ dùng gói train/dev/FULL_A và metadata đã lọc train/dev. Các lệnh dựng test hoặc kiểm bundle đầy đủ bên dưới mô tả khả năng của source, chỉ thực hiện khi có quyết định của giảng viên theo QĐ42. Bản data card công khai đã hiệu đính ngày 31/10; builder 28 tái lập bản đo 30/10 và không được chạy lại để ghi đè các hiệu đính này trong quá trình xử lý GitHub.

Phần chạy mô hình/selector ở [`../experiments/`](../experiments/) được giữ riêng. Việc nhập source này không thay đổi thuật toán, công thức, split hoặc kết quả đã có của hai nhóm.

## Lấy code và môi trường

Lần đầu:

```bash
git clone --branch feat/cd3-v4-2-data-handoff https://github.com/datachain-uit/CN-23521830-VyNH-QuangTVX.git
cd CN-23521830-VyNH-QuangTVX/data_pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4.txt
```

Các lần cập nhật sau, tại repo đã clone:

```bash
git pull --ff-only origin feat/cd3-v4-2-data-handoff
```

`requirements-v4.txt` khóa môi trường cho builder V4 và kiểm thử giao thức. Các bước cũ/sentiment có dependency riêng; tham khảo import của từng script và [`afelo_sentiment_cardiff/requirements-sentiment.txt`](afelo_sentiment_cardiff/requirements-sentiment.txt). Không chạy toàn bộ pipeline nếu chỉ cần cập nhật code.

Chạy các lệnh bên dưới từ `data_pipeline/`. Builder V4 xác định root theo vị trí file; các script/notebook cũ có thể dùng thư mục làm việc hiện tại. Tên `working_13-10` còn xuất hiện trong một số ví dụ lịch sử; khi dùng repo này, root pipeline là `data_pipeline/`.

## Các bước nguồn

| Bước | Mã nguồn | Vai trò |
|---|---|---|
| 01–02 | Notebook có tiền tố `01_`, `02_` | Chuyển raw thành Parquet, tạo dimension/bridge |
| 03–05 | Script/notebook EDA và `05_build_review_time_phases.py` | Khảo sát dữ liệu, enrollment, cửa sổ thời gian |
| 06–10 | Builder chất lượng, sentiment, LOQV, phase và data card | Pipeline ban đầu |
| 11–14 | Audit thời gian/chapter và mô phỏng | Kiểm chứng định nghĩa trước các bản tiếp theo |
| 15–16 | `15_build_final_quality_v2.py`, `16_build_phase_features_v2.py` | Bản V2 |
| 18–20 | `18_build_final_quality_v3.py`, `19_build_phase_features_v3.py`, `20_build_fullA.py` | Bản V3 và đặc trưng toàn cửa sổ |
| 21–23 | Builder V4 và `cq_v4_common.py` | Assignment theo family, phase/FullTrain-A và bàn giao |

Các script audit bổ sung nằm trong `review_7_4_audit/`, `review_v3_compliance/`, `formula_feature_v4_audit/` và `d06_v4_audit/`. Git chỉ chứa source của các audit, không chứa báo cáo hoặc ví dụ dòng dữ liệu của chúng. Notebook EDA số 17 có ví dụ người học trong nội dung nên chỉ công bố bản Python.

## Input và tái tạo V4

Dữ liệu được cấp riêng theo quyền truy cập của nhóm. Cần đặt đúng các thư mục tương đối dưới root pipeline, hoặc dùng tham số đường dẫn mà từng script hỗ trợ. Bản clone GitHub không thay thế dữ liệu đầu vào.

- Raw/mapping: `input/moocubex_parquet/`, `input/moocubex_model/` và các bảng event/mapping đã chuẩn bị.
- Builder 21 cần `final_quality_v3/` cùng dữ liệu/mapping mà source tham chiếu.
- Builder 22 cần `final_quality_v4/`, `phase_features_v3/` và `fullA/`; các nguồn V3/fullA không được tái tạo bằng dữ liệu giả.
- Builder 23 cần các đầu ra V4, mã kiểm thử và thông tin bàn giao do hai bước trước tạo.

Khi đã có đủ input/cache của pipeline hiện hành:

```bash
python 21_build_final_quality_v4.py --threads 2 --memory 1200MB
python 22_build_phase_features_v4.py --threads 2 --memory 1200MB
python 23_finalize_data_card_v4.py
```

Không chạy lại các bước 01–20 chỉ để đọc bundle hoặc pull code. Những bước đó có đầu vào/dependency riêng và có thể tính lại dữ liệu.

## Feature CD3 V4.2 theo QĐ28

`25_build_phase_features_v4_2.py` tính lại feature CQ từ event gốc theo
`cq_tempo_features.py`, trên spine V4 đã đóng băng. Đầu ra mặc định là
`phase_features_v4_2/`, gồm Pool B train, dev P1–P4 và FULL_A train. Builder
không mở snapshot test và không huấn luyện mô hình.

```bash
python -m pip install -r requirements-v4_2.txt
python -m pytest -q test_cq_tempo_features.py tests/test_cq_v4_2_builder.py
python 25_build_phase_features_v4_2.py --threads 2 --memory 1200MB --batch-size 1500
python 26_verify_phase_features_v4_2.py
```

Data card của bộ CD3 đang dùng là [DATA_CARD_V4_2.md](DATA_CARD_V4_2.md), có bản
đi kèm trong `phase_features_v4_2/`. Sau khi bổ sung test bằng bước 27, tái tạo
tài liệu và evidence từ bundle/cache hiện hành bằng
`python 28_build_data_card_v4_2.py`, rồi chạy lại verifier 26.
Nếu đã nhận xác nhận D06 thực tế, truyền ngày nhận bằng `--d06-confirmed-on`;
tham số này ghi nhận thông tin, không ký thay phiếu kiểm thủ công. Bước 28 đo
Y37, kiểm Gctx và ghi nguồn thời gian của 12 cột, giữ nguyên hash của 20 file
raw/ready cùng sáu artifact preprocessing/feature/QA23.

Để bổ sung test P1–P4 vào bộ V4.2 đã có:

```bash
python 27_build_test_features_v4_2.py --threads 2 --memory 1200MB --batch-size 1500
python 26_verify_phase_features_v4_2.py
```

Bước 27 đọc spine test V4 và event gốc, khôi phục nguyên trạng các tham số từ
`preprocessing.json`, danh sách feature và QA23 đã chốt trên train. Bước này chỉ
transform test, không gọi fit/QA23 hoặc huấn luyện mô hình. Hash của raw/ready
train/dev/FULL_A và các artifact fit được đối chiếu trước/sau. Đầu ra test là
`raw_features/pool_b/test_phase1.parquet` đến `test_phase4.parquet` cùng các file
ready tương ứng trong `model_ready/`. Manifest, loader và checksum được cập nhật
sau khi dữ liệu test qua kiểm tra; manifest dev-only ban đầu được lưu trong
`provenance/pre_test_extension/`.

Để xuất metadata `teacher_subgroup` chính thức theo Y31:

```bash
python 29_build_teacher_subgroup_v4_2.py
python 28_build_data_card_v4_2.py
python 26_verify_phase_features_v4_2.py
```

Bước 29 tạo `metadata/teacher_subgroup.parquet`, một dòng/course, từ relation
`course_teacher` sau trim ID, bỏ NULL/rỗng và khử trùng. Reference là union
teacher ID của các khóa train: `seen` khi tất cả đã gặp, `new` khi tất cả chưa
gặp, `mixed` khi có cả hai, `missing` khi mapping không có ID dùng được. Không
đòi hỏi profile teacher ghép được. Reference được đóng băng trong
`metadata/train_teacher_reference.json`; chạy lại không tự mở rộng reference.
Spec, mapping chuẩn hóa, audit, version/hash nằm trong bundle và manifest.

Loader cung cấp `teacher_subgroups()` để đọc bảng metadata, hoặc
`load(..., include_teacher_subgroup=True)` để ghép `teacher_subgroup`,
`n_teachers`, `n_seen_teachers`, `n_new_teachers`. Mặc định loader giữ contract
cũ; bốn cột bổ sung là metadata và giữ ngoài learner/selector allowlist.
Raw/ready Parquet, nhãn, split và preprocessing được đối chiếu checksum trước/sau.
Mapping nguồn không có lịch sử as-of; nhóm này dùng cho phân tích hồi cứu.

Nguồn bắt buộc: `phase_features_v4/supervised_raw/P1..P4/{train,dev}.parquet`,
`phase_features_v4/fulltrain_a/train_raw.parquet`, raw Parquet của MOOCubeX,
cache Cardiff cũ và phần cache bổ sung trong `final_quality_v2/sentiment/`.
Không suy ra feature CD3 bằng đổi tên aggregate V4. Catalog bài tập lấy
`exercise_id`, `context_id` và điểm tối đa từ entity problem; `context_id` là
cả mảng ngữ cảnh, không thay bằng chapter hoặc explode từng concept. Comment
giữ cả text rỗng và các ID có cùng nội dung, nối course-comment và resource rồi
khử trùng theo comment/course. Video giữ đồng hồ V4 và toàn bộ segment được
chọn theo thời điểm bắt đầu, kể cả phần đuôi vượt cutoff, đúng helper đã giao.

- `raw_features/`: 85 feature CD3, 3 mask audit và `phase_fraction`, cùng metadata/target V4.
- `model_ready/`: learner view V1-Median; selector lấy subset theo allowlist.
- `preprocessing.json`: median/IQR, log columns, gender maps, thứ tự fit và hash dòng train.
- `audit/`: QA23 theo phase, union cột tạm loại, cardinality, event bounds, NULL/absence, sentiment coverage và so sánh D06 tự động.
- `run_manifest.json`, `SHA256SUMS.txt`, `_SUCCESS.json`: provenance và trạng thái kiểm tra.
- `load_cq_v4_2.py`: loader sinh cùng bundle, hỗ trợ `load(split='train', phase=None, view='learner', raw=False)`; sau bước 27 có thể gọi `load(split='test', phase='P1', view='learner')`.

Trước QA23, learner có 76 cột, selector có 74 cột. Cột chưa được duyệt có
|rho| ≥ 0,5 ở bất kỳ phase train nào bị tạm loại khỏi cả hai view; raw vẫn giữ
đủ cột. Fit V1/gender một lần trên train P1–P4, cột tĩnh dùng một dòng/enrollment;
dev, test và FULL_A chỉ transform. LR one-hot gender do nhóm thực nghiệm xử lý tiếp.
So sánh D06 bằng máy không thay chữ ký kiểm thủ công hoặc phê duyệt của giảng viên.
Verifier đọc lại file đã xuất, kiểm checksum, đối chiếu raw→V1 bằng SQL và tính
lại median/IQR train; báo cáo riêng nằm ở `outputs/phase_features_v4_2_validation.json`.

## Giao diện bàn giao V4

- Grain phase: `(user_id, course_id, phase)`; có `enrollment_id`, `row_id`, `course_family_id` và assignment cố định.
- Target chính: `q_alpha_1` / `y_alpha_1`; hai target alpha=0.1 vẫn được lưu trong dữ liệu gốc. Target và metadata không thuộc feature allowlist.
- Learner dùng 57 numeric + 5 categorical; selector dùng 56 numeric, loại `phase_fraction` và các code danh mục khỏi khoảng cách.
- P1/P2/P3/P4 có fraction 0.25/0.50/0.75/0.90. FullTrain-A chỉ có train, cửa sổ λ=1, một dòng/enrollment; không ghép A vào Pool B.
- Median/mean/scale được fit chung trên train P1–P4; dev/test và FullTrain-A chỉ transform. Encoder lấy vocabulary train, unknown/missing=0.
- `phase_features_v4/feature_columns.json`, `preprocessing.json`, `run_manifest.json` và `SHA256SUMS.txt` trong dữ liệu riêng là contract feature order/version/hash. Không suy ra feature order từ tất cả cột của Parquet.
- `phase_features_v4/runtime_dev_only/` chứa Pool B train, dev theo phase, FullTrain-A và loader `load_cq_v4.py`; không chứa test. Test vẫn tuân thủ quy trình freeze của nhóm.

Các file manifest, loader được sinh và artifact runtime đi cùng bundle dữ liệu riêng; không được nhập vào Git thay cho dữ liệu đã kiểm chứng. Việc build/check dữ liệu không đồng nghĩa đã huấn luyện hoặc nghiệm thu mô hình.

## Kiểm thử và công bố source

Kiểm thử giao thức trên fixture tổng hợp, không cần dữ liệu người học:

```bash
python -m pytest tests/test_cq_v4_protocol.py
python tests/test_phase_feature_cutoff.py
```

Lệnh thứ hai kiểm tra thêm log tương lai không thay đổi đặc trưng của phase trước; bằng chứng được ghi vào thư mục output đã ignore.

Trước commit, chạy các bước trong [`GITHUB_CODE_ONLY.md`](GITHUB_CODE_ONLY.md). Không đưa raw/derived data, bảng ID, predictions, model, archive, thông tin đăng nhập hoặc notebook có output lên GitHub. Làm việc trên nhánh `data-pipeline` và mở pull request vào `main` theo [`HANDOFF.md`](HANDOFF.md).
