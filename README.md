# CQ V4 — Thực nghiệm nén dữ liệu theo giai đoạn và lớp

Repo chung của nhóm cho giao thức CQ V4 theo review ngày 12, 13, 22 và 23/10/2026. Phạm vi hiện tại là kiểm tra dữ liệu, D13 và learner/headroom trên train/dev. Bộ feature chính chuyển sang CD3 trên khung V4, UL chính chuyển sang log loss theo QĐ25. **Panel cần được duyệt; test chỉ mở sau freeze manifest M1 được giảng viên duyệt.**

## Cấu trúc

```text
data_pipeline/                 # Giữ chỗ cho mã nguồn nhóm dữ liệu
experiments/
  src/skq_exp/                 # Adapter, selector, training, metrics và runner
  configs/                    # Config CQ V4 và fixture cần cho regression
  scripts/                    # Audit, endpoint, headroom, hiệu chuẩn và runner phụ trợ
  tests/                      # Kiểm thử giao thức và hồi quy
  external/official_repos.lock.json
  requirements/
  pyproject.toml
tools/verify_source_snapshot.py
```

Bên trong `experiments/` giữ cấu trúc package của repo cá nhân đang chạy. Không di chuyển/đổi tên module, không sửa thuật toán hoặc config CQ V4 để tổ chức lại repo.

## Nhánh làm việc

| Nhánh | Vai trò |
|---|---|
| `main` | Bản code tích hợp để nhóm và giảng viên đối chiếu |
| `data-pipeline` | Nhánh nhóm dữ liệu tự quản lý và cập nhật mã nguồn; xem thư mục data_pipeline trên nhánh này |

Đoạn yêu cầu cung cấp repo chung chưa chỉ định nhánh nộp. Y19 của Review22 yêu cầu hợp nhánh `cq-v1-protocol-fixes` vào `main` qua pull request, không squash, trong ngữ cảnh repo nguồn. Không xem việc nhập code sang repo này là đã hoàn tất PR/tag của repo nguồn hoặc là bằng chứng cô yêu cầu một nhánh cụ thể cho repo mới.

## Quy trình CQ V4 hiện hành

1. Nhận bundle `runtime_dev_only/` từ nhóm dữ liệu; audit hash, schema, cohort và split theo `course_family_id`.
2. Tính raw CD3, QA23 và V1-Median train-only; chạy D13 L0/L1/S, V20/V50 diagnostic và lưới learner đã đăng ký. LR gender one-hot, balanced chỉ sensitivity.
3. FullTrain-B cấu hình đã chọn với model seeds 42/43/44; headroom LL tại 0,5/1/2/5%, C02 với năm seed 11/29/47/71/101; báo SD/SE, family CI và margin/rho đề xuất theo QĐ25/QĐ26.
4. QĐ16 cho baseline và ba ứng viên `skq_pc`, `skq_pc_lrq_sq`, `skq_pc_sqrt`; ghi fidelity, time/RSS và failure thật.
5. Trình V4, rho*, nmin/delta_C và khác biệt dữ liệu để cô chốt trước panel.
6. Panel dev ba seed 11/29/47, XGB natural; C03 matched, A2/A4/A8 và size-matched native theo review.
7. Ngày 08/11 trình QĐ5 và freeze manifest. Chưa được tự mở test hoặc đổi split/metric theo điểm ứng viên.

Hướng dẫn chạy và config hiện hành nằm trong [experiments/README.md](experiments/README.md); xem [bổ sung review 23/10](experiments/docs/cq_v4/REVIEW_2026_10_23_IMPLEMENTATION.md). A2/A4 và C03 matched đã có đường chạy chuẩn bị; xem [phạm vi ablation](experiments/docs/cq_v4/ABLATIONS_PREPARATION.md). Config/kiểm thử không đồng nghĩa đã có utility panel hoặc đã được duyệt. Các thay đổi có truy vết trong [IMPLEMENTATION_CHANGES.json](IMPLEMENTATION_CHANGES.json).

## Giữ nguyên code nguồn

Nguồn: [UIT23521830/skq-experiment](https://github.com/UIT23521830/skq-experiment), nhánh `cq-v1-protocol-fixes`, commit `ba2cbcd0f66b0d32c3215518c9741d676f072579`.

Bản nhập đầu tiên giữ 222 file. Các file đã bỏ được ghi ở [CURATION.json](CURATION.json). File còn trong `SOURCE_SNAPSHOT_SHA256SUMS.txt` phải khớp nguồn; file được mở rộng có khai báo/hash riêng trong `IMPLEMENTATION_CHANGES.json`, không tiếp tục nhận là byte-identical. Bổ sung 23/10 thay view/learner/evidence theo review, không sửa engine selector SKQ-PC, config lịch sử hoặc kết quả cũ. README không thuộc nhóm file bất biến.

```text
python tools/verify_source_snapshot.py
```

Notebook đang chạy từ repo cá nhân giữ nguyên URL/commit/input. Không đổi nguồn giữa lượt chạy hoặc thay commit lịch sử trong archive. Notebook mới dùng repo chung phải chạy từ thư mục `experiments/`.

## Phân định dữ liệu và kết quả

- `data_pipeline/` chỉ giữ chỗ; nhóm dữ liệu tự đưa code lên nhánh riêng. Adapter trong `experiments/src/skq_exp/data/` không thay cho builder từ raw log.
- Drive chung lưu báo cáo theo ngày, dữ liệu và archive theo quyền truy cập. Link Drive: **chờ nhóm bổ sung**.
- Không commit dữ liệu, predictions, selected indices, model, token, archive hoặc notebook có output.
- Repo ngoài chỉ fetch đúng commit khóa, không vendor vào Git.

## Nguồn dữ liệu và xử lý Review ngày 31 tháng 10 năm 2026

Dữ liệu nghiên cứu lấy từ [MOOCCubeX của THU KEG](https://github.com/THU-KEG/MOOCCubeX). Trích dẫn: Yu và cộng sự, “MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs”, CIKM 2021, tr. 4643–4652; BibTeX trong `experiments/references.bib`.

Repo công khai chỉ bàn giao mã và tài liệu tổng hợp theo QĐ40. PR #5 đã đóng không merge, nhánh dữ liệu đã xóa; ref/cache PR và object LFS còn chờ GitHub Support. Mã CD3/V4.2 bàn giao qua PR #4, chờ giảng viên xem. [Hồ sơ xử lý](data_pipeline/GITHUB_REVIEW_2026_10_31.md) ghi trạng thái thực tế, Traffic và các việc còn cần xác nhận.

Trước freeze M1 được giảng viên duyệt, chỉ dùng gói train/dev/FULL_A và metadata đã lọc train/dev. Test thuộc phạm vi niêm phong QĐ42; không công bố dữ liệu dòng, nhãn, dự đoán, chỉ số dòng, con trỏ hoặc luật Git LFS trên bất kỳ nhánh, tag hay PR nào.
