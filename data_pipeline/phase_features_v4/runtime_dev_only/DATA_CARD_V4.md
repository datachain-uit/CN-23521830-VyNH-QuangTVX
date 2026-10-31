# Data card CQ V4

**Ngày tạo:** 2026-10-22. **Trạng thái:** dữ liệu final/phase/FullTrain-A đã build và kiểm tra kỹ thuật; còn các kiểm tra mô hình trước khi nghiệm thu M0-V4.

**Phạm vi người dùng chốt:** dùng bộ feature hiện có của V3; không cần đối chiếu TEMPO-DQ. Tài liệu này phân biệt dữ liệu đã xây với công việc thực nghiệm chưa chạy.

## 1. Định danh và quần thể

Dataset version: **CQ V4**. Final: [final_quality_v4](). Phase: [phase_features_v4](). Runtime an toàn cho dev: [runtime_dev_only](.).

Khóa gốc: `(user_id,course_id)`; snapshot `(user_id,course_id,phase)`. `enrollment_id` là SHA256 JSON của khóa cặp; `row_id` là SHA256 JSON enrollment_id + phase. Metadata ID không thuộc X.

Final giữ **10.455.017** cặp đủ thời gian sau lọc V3, gồm **178.598** cặp có nhãn và **10.276.419** cặp chưa có nhãn. Tỷ lệ có nhãn **1,7083%**. Cặp chưa có nhãn không thành lớp A, giữ split/enrollment_id/row_id NULL trong kho đầy đủ. Family chỉ được xây trên 1.808 course có nhãn; course ngoài cohort có family NULL.

Mỗi phase giữ 10.455.017 dòng raw đầy đủ và 178.598 dòng supervised. Tổng bốn phase **714.392 snapshot supervised**. Những dòng của cùng cặp không phải quan sát độc lập.

## 2. Bảng những gì đã làm và thay đổi so với V3

| Hạng mục | Trạng thái/thay đổi | Đã làm | Giới hạn/phần còn thiếu |
|---|---|---|---|
| Cohort và bộ lọc | Giữ nguyên | 178.598 cặp có nhãn / 1.808 course; loại only-unknown chapter như V3. Không oversample hoặc xóa lớp E. | Không thiếu ở bước build |
| q/y, thành phần và reference P95 | Giữ nguyên | Mọi giá trị gốc được đối chiếu với V3; chỉ metadata split/family thay đổi. Không inference sentiment lại. | Độ giá trị giáo dục/sentiment chưa được thẩm định độc lập |
| Target chính | Đã chốt | q_alpha_1 / y_alpha_1; alpha 0,1 chỉ sensitivity. | Không có lựa chọn alpha theo kết quả dev |
| Family mapping | Mới | Toàn cohort, graph ccid Jaccard≥0,9, minimum 5; connected components; không lọc theo tên. | Family là proxy tương đồng nội dung, không phải định danh lần mở chính thức |
| Độ nhạy family | Đã làm | Ngưỡng 0,8 lưu riêng; không thay assignment chính 0,9. | Không chọn ngưỡng theo điểm mô hình |
| Split | Thay đổi | Test 20% họ mở muộn nhất; train/dev seed42 theo quartile kích thước × has_E; không giao family/course/cặp. | Temporal overlap còn cao; cần NOLFO sau |
| Raw feature P1–P4 | Giữ nguyên | 57 numeric + 5 categorical; toàn bộ raw numeric/gender/cutoff khớp V3. Các code danh mục fit lại. | Không có yêu cầu đối chiếu TEMPO-DQ theo chỉ dẫn người dùng |
| Hai feature view | Đã xuất | Selector: 56 numeric bỏ phase_fraction và mọi categorical ID. Learner: 57 numeric gồm phase_fraction + 5 categorical, tổng 62. | Categorical codes là category, không phải biến liên tục; learner cần xử lý phù hợp |
| Preprocessing | Thay đổi | Một median/scaler chung trên 527624 snapshot train P1–P4; encoder chỉ fit train V4. Không missing indicator. | Không dùng ready/scaler V3 hoặc fit scaler riêng phase |
| Phase và row ID | Bổ sung | Metadata family/enrollment/row ID; phase_fraction vẫn có bốn giá trị khác nhau sau chuẩn hóa chung. | Các snapshot của cùng cặp có phụ thuộc |
| Chuỗi tuần | Giữ giá trị | Giá trị behavior_sequence khớp V3; cập nhật split và thêm family/row ID. | Chuỗi tuần thưa, không tự chèn tuần trống |
| FullTrain-A | Đã tạo theo contract V4 | Tái sử dụng raw λ=1 từ fullA; lấy train V4, remap code và dùng preprocessor Pool B. Thêm phase_fraction=1. | Chưa huấn luyện FullTrain-A; A không thuộc Pool B |
| Qc/nc, μy và oracle | Đã tính | Qc/nc trên labeled enrollment; μy từ train, một dòng/cặp; oracle floor dev và signed oracle errors được lưu. | Đây là oracle của phép tổng hợp, không phải kết quả dự đoán mô hình |
| Thống kê card | Đã tính | ICC(1), nc, concentration, missingness, class/family E, user/teacher cũ mới, temporal overlap. | Số liệu không chứng minh khả năng tổng quát hóa hoặc công bằng |
| Cutoff | Đã kiểm tra tự động | Công thức toàn supervised; 20 cặp/80 snapshot đối chiếu độc lập từ video/problem/comment/reply. | Chưa có người duyệt thủ công hồ sơ 20 cặp |
| Runtime dev-only | Đã đóng gói | Chỉ train/dev và A train; loader từ chối test; preprocessing cố định. Test chỉ có trong root private. | Không mount toàn root phase_features_v4 vào notebook dev |
| Manifest/hash/loader | Đã có | Mỗi gói có manifest, SHA256SUMS và loader; danh sách/feature order có hash. | Tái tạo cần giữ input V3/fullA và phiên bản môi trường |
| Kiểm tra 1–12 | Một phần theo phạm vi dữ liệu | Các check dữ liệu đã PASS; kiểm tra nguồn 20 cặp đã có hồ sơ. | Probe ID/mask, shuffled label, majority, trọng số và baseline cần pipeline thực nghiệm |
| Đối chiếu TEMPO-DQ | Không yêu cầu | Dùng nguyên feature do người dùng cung cấp; không coi thiếu mapping TEMPO-DQ là blocker. | Không tuyên bố tái lập feature TEMPO-DQ |


## 3. Family và chia tập

Ngưỡng 0,9 tạo **1740 họ**, **68 cạnh**, **68 họ nhiều course**. Phân bố trạng thái: `{"jaccard_component": 136, "singleton_lt5_ccid": 22, "singleton_no_ccid": 2, "singleton_no_match": 1648}`.

Tập ccid lấy từ resource video của catalog, nối tất cả quan hệ ccid khác rỗng và khử trùng. Không tự chọn một ccid tối thiểu cho video đa ánh xạ. Video-set đã được đối chiếu với catalog raw độc lập.

Test: ceil(0,2 × số họ), sắp `(family_start, family_id)`; ngày mở họ là min(course_start). Phần còn lại: qcut của rank kích thước, 4 nhóm × has_E; dev lấy round(0,125 × số họ trong tầng), thứ tự ngẫu nhiên seed42. Python ties-to-even được giữ; không di chuyển họ theo kết quả mô hình. Tỷ lệ áp dụng cho số họ, không cho số enrollment.

| Split | Family | Course | Cặp có nhãn | User | E | G | A | Snapshot P1–P4 |
|---|---|---|---|---|---|---|---|---|
| train | 1219 | 1278 | 131906 | 126016 | 522 | 36006 | 95378 | 527624 |
| dev | 173 | 179 | 22717 | 22350 | 263 | 6606 | 15848 | 90868 |
| test | 348 | 351 | 23975 | 23445 | 263 | 4225 | 19487 | 95900 |


| Split | Family có E |
|---|---|
| dev | 20 |
| test | 58 |
| train | 144 |


So với bản course-random V3: train: 128,407 → 131,906 cặp; dev: 16,301 → 22,717 cặp; test: 33,890 → 23,975 cặp.

Ngưỡng 0,8 tạo 1731 họ; 18 course thay đổi tập thành viên họ. Báo cáo: [family sensitivity](COURSE_FAMILY_SENSITIVITY.md).

**Temporal overlap:** 924/1219 họ train = **75.7998%**, so với family test mở sớm nhất tại 2020-08-29 11:09:03 theo timestamp pipeline. `requires_nolfo=True`. Đã lưu danh sách overlap; chưa dựng/chạy đối chứng NOLFO. Split chính được giữ.

## 4. Target và thời gian

Target chính q_alpha_1/y_alpha_1, giữ nguyên từ V3. q = mean LOQV các chương quan sát; LOQV_alpha = sqrt(sum_available(w × (1-index)^2)/sum_available(w)), w=(1,alpha,1). Mean bỏ NULL và giữ 0. E:q<0,2; G:0,2≤q<0,4; A:q≥0,4. Alpha 1 khớp khoảng cách Euclid chuẩn hóa ba chiều khi đủ ba chỉ số; khi thiếu chiều, chỉ dùng chiều khả dụng như V3.

COELO=mean khả dụng TS/RV/AV/IF, tối thiểu hai thành phần; AFELO=mean khả dụng FV/FA; ACELO dùng số problem đúng có điều chỉnh attempts trên mẫu số catalog. P95, signed FA, model sentiment và chapter -1 giữ phiên bản V3. Xem [formula policy](formula_policy.json) và `DATA_CARD_V3.md` (tài liệu nguồn, giữ riêng) cho công thức thành phần, revision sentiment, missing/0 và điều kiện tracking.

Enrollment về 00:00:00; course_end là cuối ngày hoạt động cuối quan sát, không phải lịch đóng chính thức. Cutoff P1–P4 = enrollment + λ × (course_end−enrollment), λ=0,25/0,50/0,75/0,90. FullTrain-A có λ=1. Timestamp không tự cộng offset; session SQL UTC, ngày tài liệu Asia/Ho_Chi_Minh.

## 5. Feature và preprocessing

Learner **62 cột** = 57 numeric + 5 categorical; selector **56 numeric**. Giữ feature của người dùng; phase_fraction là feature giao thức của learner, được giữ trong metadata phase để phân tầng selector. Các mã learner/teacher_set/school_set/field_set fit từ train mới; gender giữ mã nguồn. Unknown/missing category=0, riêng gender không hợp lệ=-1.

Fit median/mean/std trên **527,624 snapshot train** gộp bốn phase. Raw giữ NULL; ready=(coalesce(raw,median_train)−mean_train)/std_train. Cột hằng dùng scale=1; không thêm missing indicators. Dev, test lưu riêng và FullTrain-A chỉ transform, không fit. FullTrain-A raw giữ nguyên 56 numeric đã có, thêm window fraction=1 để đồng nhất schema learner.

Allowlist và định nghĩa: [selector](selector_feature_columns.json), [learner](learner_feature_columns.json), [dictionary](feature_dictionary.json), [preprocessing](preprocessing.json). Danh sách numeric/categorical không chứa q/y, chỉ số cuối cửa sổ, course_end, cutoff, split, family hay ID metadata.

Tỷ lệ thiếu X theo cột/phase/split có trong [missingness.json](missingness.json), trường `n` là mẫu số. Kho sequences giữ giá trị tuần V3, thêm metadata V4.

## 6. Thống kê cấp khóa học

Qc=mean q_alpha_1 trên cặp có nhãn; nc là số cặp đó. Đây là quần thể được QĐ21 chấp nhận, không đại diện toàn bộ người đăng ký. μy là hằng số train; nmin=30, delta_C=0,01, delta=0,005; class weights tự nhiên.

| Hằng số | Giá trị |
|---|---|
| E | 0.108847583929 |
| G | 0.351612432810 |
| A | 0.656836766532 |


| Split | Course | nc min | Q1 | Median | Q3 | Max | Course nc≥30 | Family có course nc≥30 |
|---|---|---|---|---|---|---|---|---|
| dev | 179 | 1 | 4.0 | 14.0 | 60.0 | 3040 | 67 | 67 |
| test | 351 | 1 | 2.0 | 9.0 | 40.0 | 4263 | 109 | 109 |
| train | 1278 | 1 | 4.0 | 15.0 | 58.75 | 16111 | 455 | 451 |


| Split | ICC(1) |
|---|---|
| train | 0.45952451 |
| dev | 0.50246277 |
| test | 0.40700910 |
| all | 0.45837919 |


| Split | Course dùng EC | Oracle floor | Signed oracle error |
|---|---|---|---|
| dev | 67 | 0.08280915 | -0.03194576 |
| test | 109 | 0.06421559 | 0.02437618 |
| train | 455 | 0.08690529 | -0.02780042 |


| Split | Course | Tỷ trọng lớn nhất | Số course hiệu dụng |
|---|---|---|---|
| dev | 179 | 13.3820% | 16.6678 |
| test | 351 | 17.7810% | 21.6273 |
| train | 1278 | 12.2140% | 48.4792 |


Các chỉ số oracle dùng đúng prior lớp của course nhân μy train; chúng không phải kết quả FullTrain-B. SD của πc và concentration theo family cũng lưu trong [course_statistics.json](course_statistics.json).

User cũ/mới tính trên lượt ghi danh test, so với user đã có trong train: `{"n_test_enrollments": 23975, "new_user_enrollments": 22328, "old_user_enrollments": 1647}`.

Giảng viên cũ/mới: phân loại mỗi lượt test thành tất cả giảng viên cũ, tất cả mới, mixed, hoặc thiếu metadata; không suy ra từ teacher_set_code=0. Số liệu: `{"all_teachers_new": 20773, "all_teachers_old": 567, "mixed_old_new": 2384, "n_test_enrollments": 23975, "no_teacher_metadata": 251}`.

## 7. Kiểm tra và việc còn thiếu

Quality build **83/83**, phase/FullTrain-A/loader **188/188**, kiểm tra độc lập bổ sung **14/14**; tổng **285** check dữ liệu PASS. Có 7 unit test giao thức family/split đã chạy đạt.

Đối chiếu độc lập 20 cặp × 4 phase đã tái đọc video/problem/comment/reply và reply nhận, kiểm tra số event cùng min/max timestamp khớp raw snapshot/audit. Hồ sơ dòng lưu riêng trong `phase_features_v4/audit/`; card chỉ ghi số tổng hợp. Việc người đánh giá duyệt thủ công hồ sơ này còn chờ.

| Việc chưa thực hiện | Đơn vị thực hiện | Trạng thái |
|---|---|---|
| ID-only and missing-mask probes on V4 | experiment | NOT_RUN |
| Shuffled-label / majority model checks on V4 | experiment | NOT_RUN |
| Selector weight contract and baseline reproduction | experiment | NOT_RUN |
| FullTrain-A/B training, headroom, QD16 and panel | experiment | NOT_RUN |
| NOLFO comparison because temporal_overlap>5% | experiment, thesis | REQUIRED_LATER |
| Human review of 20-pair source comparison and scientific label validation | research | PENDING |


**Không có phần đối chiếu TEMPO-DQ còn thiếu:** hạng mục này được bỏ theo yêu cầu người dùng. Không suy ra tính hợp lệ nhãn, fairness hoặc chất lượng mô hình từ các check kỹ thuật. Nhãn vẫn có thiên lệch chọn cohort, dữ liệu thiếu modality, P95 hiệu chuẩn trên quần thể V2 và mốc thời gian hồi cứu.

Trạng thái bàn giao: **DATA_BUILD_PASS / dữ liệu sẵn cho nhóm thực nghiệm**, nhưng **chưa READY_FOR_M0_V4** vì còn các probe/contract mô hình và review độc lập. [handoff_status.json](handoff_status.json) ghi từng việc chờ.

## 8. Cách đọc và tái tạo

Notebook dev chỉ dùng `phase_features_v4/runtime_dev_only/`. Root private còn test data và toàn kho raw nên không dùng nguyên root đó làm bundle M0.

```python
import importlib.util
from pathlib import Path
root = Path(".").resolve()
spec = importlib.util.spec_from_file_location("cq_v4_loader", root / "load_cq_v4.py")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
train = loader.load_phase(phase=None, split="train", root=root)
dev_p1 = loader.load_phase(phase="P1", split="dev", root=root)
full_a = loader.load_fullA(root=root)
x_cols = loader.feature_columns(view="learner", root=root)
# train.project(",".join(x_cols)) là X; q/y và metadata đọc riêng.
```

Từ `working_13-10/`, dùng Python 3.12.13 có DuckDB/NumPy/Pandas. Môi trường build: `{"duckdb": "1.5.5", "numpy": "2.4.6", "pandas": "3.0.3"}`. Phiên bản được khóa trong `requirements-v4.txt`.

```bash
python 21_build_final_quality_v4.py --threads 2 --memory 1200MB
python 22_build_phase_features_v4.py --threads 2 --memory 1200MB
python 23_finalize_data_card_v4.py
```

Builder ghi riêng V4, không sửa input V3/fullA. Bước 21 khóa assignment trước bước 22 fit X. Tên phiên bản/ngày tạo cố định trong splits; thời gian chạy nằm ở manifest riêng. Mỗi root và runtime có SHA256SUMS. Tái tạo cần dữ liệu private đã có; code/manifest không chứa dữ liệu thay thế.

Dữ liệu dòng và ID giả danh giữ ngoài Git; `.gitignore` chặn các output V4. Điều khoản phát hành/đồng thuận/kiểm tra công bằng độc lập chưa được xác nhận; gói này không tự cấp giấy phép công bố dữ liệu.
