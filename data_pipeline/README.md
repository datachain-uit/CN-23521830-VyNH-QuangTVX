# Pipeline dữ liệu CQ V4

Mã nguồn của nhóm dữ liệu để chuyển raw MOOCubeX thành bảng chất lượng, nhãn, đặc trưng theo phase, FullTrain-A và bundle bàn giao V4. Thư mục này giữ cấu trúc các script/notebook đã phát triển; không chứa dataset, model, cache hoặc kết quả thực nghiệm.

Phần chạy mô hình/selector ở [`../experiments/`](../experiments/) được giữ riêng. Việc nhập source này không thay đổi thuật toán, công thức, split hoặc kết quả đã có của hai nhóm.

## Lấy code và môi trường

Lần đầu:

```bash
git clone --branch data-pipeline https://github.com/datachain-uit/CN-23521830-VyNH-QuangTVX.git
cd CN-23521830-VyNH-QuangTVX/data_pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4.txt
```

Các lần cập nhật sau, tại repo đã clone:

```bash
git pull --ff-only origin data-pipeline
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
