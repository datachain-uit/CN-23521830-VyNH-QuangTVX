# CQ V4 — Thực nghiệm nén dữ liệu theo giai đoạn và lớp

Repo chung của nhóm cho giao thức CQ V4 theo review ngày 12, 13 và 22/10/2026. Phạm vi hiện tại là kiểm tra dữ liệu, chạy tham chiếu và chuẩn bị so sánh coreset trên train/dev. **Test chỉ mở sau freeze manifest M1 được giảng viên duyệt.**

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
| `data-pipeline` | Nhánh dành cho nhóm dữ liệu tự cập nhật mã nguồn; hiện chỉ có tài liệu bàn giao, chưa có builder |

Đoạn yêu cầu cung cấp repo chung chưa chỉ định nhánh nộp. Y19 của Review22 yêu cầu hợp nhánh `cq-v1-protocol-fixes` vào `main` qua pull request, không squash, trong ngữ cảnh repo nguồn. Không xem việc nhập code sang repo này là đã hoàn tất PR/tag của repo nguồn hoặc là bằng chứng cô yêu cầu một nhánh cụ thể cho repo mới.

## Quy trình CQ V4 hiện hành

1. Nhận bundle `runtime_dev_only/` từ nhóm dữ liệu; audit hash, schema, cohort và split theo `course_family_id`.
2. FullTrain-A/B XGB natural; LR FullTrain-B là tham chiếu, balanced chỉ sensitivity.
3. Headroom trên dev tại 0,5/1/2/5%, C02 với năm seed 11/29/47/71/101; báo SD/SE và family CI.
4. QĐ16 cho baseline và ba ứng viên `skq_pc`, `skq_pc_lrq_sq`, `skq_pc_sqrt`; ghi fidelity, time/RSS và failure thật.
5. Trình V4, rho*, nmin/delta_C và khác biệt dữ liệu để cô chốt trước panel.
6. Panel dev ba seed 11/29/47, XGB natural; C03 matched, A2/A4/A8 và size-matched native theo review.
7. Ngày 08/11 trình QĐ5 và freeze manifest. Chưa được tự mở test hoặc đổi split/metric theo điểm ứng viên.

Hướng dẫn chạy và các config hiện hành nằm trong [experiments/README.md](experiments/README.md). A2/A4 và C03 matched đã có đường chạy chuẩn bị cho train/dev; xem [phạm vi ablation](experiments/docs/cq_v4/ABLATIONS_PREPARATION.md). Config và kiểm thử không đồng nghĩa đã có kết quả panel hoặc đã được duyệt. Các bổ sung sau snapshot gốc được ghi ở [IMPLEMENTATION_CHANGES.json](IMPLEMENTATION_CHANGES.json).

## Giữ nguyên code nguồn

Nguồn: [UIT23521830/skq-experiment](https://github.com/UIT23521830/skq-experiment), nhánh `cq-v1-protocol-fixes`, commit `ba2cbcd0f66b0d32c3215518c9741d676f072579`.

Bản nhập đầu tiên giữ 222 file. Bản hiện tại bỏ hướng dẫn GitHub/Kaggle cũ, config và script không dùng; ghi rõ danh sách ở [CURATION.json](CURATION.json). Các file còn nằm trong `SOURCE_SNAPSHOT_SHA256SUMS.txt` phải khớp hash nguồn. Runner/registry được mở rộng cho ablation, có hash riêng trong `IMPLEMENTATION_CHANGES.json`; engine SKQ-PC gốc, các config và test cũ không đổi. README được viết lại, không thuộc nhóm file bất biến.

```text
python tools/verify_source_snapshot.py
```

Notebook đang chạy từ repo cá nhân giữ nguyên URL/commit/input. Không đổi nguồn giữa lượt chạy hoặc thay commit lịch sử trong archive. Notebook mới dùng repo chung phải chạy từ thư mục `experiments/`.

## Phân định dữ liệu và kết quả

- `data_pipeline/` chỉ giữ chỗ; nhóm dữ liệu tự đưa code lên nhánh riêng. Adapter trong `experiments/src/skq_exp/data/` không thay cho builder từ raw log.
- Drive chung lưu báo cáo theo ngày, dữ liệu và archive theo quyền truy cập. Link Drive: **chờ nhóm bổ sung**.
- Không commit dữ liệu, predictions, selected indices, model, token, archive hoặc notebook có output.
- Repo ngoài chỉ fetch đúng commit khóa, không vendor vào Git.
