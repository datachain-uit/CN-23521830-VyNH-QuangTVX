# Hướng dẫn dùng snapshot CQ V4

Thư mục này giữ code tại commit nguồn `ba2cbcd0f66b0d32c3215518c9741d676f072579`. Tệp README gốc và các script/config legacy được giữ để truy vết; không dùng trạng thái cũ trong tài liệu đó thay trạng thái thí nghiệm hiện tại.

## Cài đặt và thư mục làm việc

```text
cd experiments
python -m pip install -e ".[boosting,dev]"
```

Trên Kaggle, khi cần đủ dependency baseline, dùng extra `kaggle-panel` từ thư mục này. Không mặc định GPU làm selector CPU chạy trên GPU; tham chiếu thiết bị thật trong config/diagnostics.

Các kiểm thử baseline cần mã nguồn ngoài đúng lock. Ví dụ cho bộ unit test hiện tại:

```text
python scripts/fetch_official_repos.py --repo tabular_data_distillation
python -m pytest tests -q
```

Các baseline khác cần fetch repo tương ứng và dependency đúng môi trường trước khi chạy. Không đổi thuật toán/fallback để che lỗi thiếu dependency. Không đẩy `external/repos/` lên GitHub.

## Đầu vào riêng tư

Nhận `runtime_dev_only/` do nhóm dữ liệu cung cấp, có Pool B train, bốn phase dev và FullTrain-A train. Không chép feature/label test vào gói dev. Đường dẫn input ở Kaggle tùy dataset được gắn; truyền `--input-dir` tường minh, không sửa cứng code để tìm dữ liệu.

Ví dụ chuẩn bị FullTrain-B, không tự chạy panel hoặc mở test:

```text
python -m skq_exp.cli prepare-external --config configs/cq_v4_m0_fulltrain_b_dev_natural_xgb_v1.json --input-dir <runtime_dev_only> --dataset course_quality_v4
```

Config giữ nguyên từ repo cá nhân: XGB A/B, headroom bốn rho, tám config LR và panel dev ba seed. Việc nhập config không phải phê duyệt thay tham số hoặc xác nhận panel đã chạy. Theo trạng thái bàn giao, A2/A4 mới có đặc tả, chưa được mô tả là runtime đã hoàn tất.

## Notebook đang chạy và notebook mới

- Notebook đang chạy từ repo cá nhân: giữ nguyên repo/commit/input; không thay giữa lượt chạy.
- Notebook mới từ repo chung: clone repo, pin commit repo chung được bàn giao, rồi đặt working directory và đường dẫn cài đặt thành `<checkout>/experiments`.
- Không thay commit lịch sử trong archive cũ bằng commit repo chung. Lưu cả source commit và import commit khi so sánh.
- Config hoặc script chuẩn bị tài liệu legacy có thể cần tài liệu private không nằm trong snapshot; không coi mọi script tạo báo cáo cũ là entrypoint thực nghiệm.

## Không thay đổi cơ chế trong lần chuyển repo

`src/`, `configs/`, `tests/`, dependency lock và script nguồn phải khớp hash ở `../SOURCE_SNAPSHOT_SHA256SUMS.txt`. Bổ sung thuật toán/config trong tương lai phải qua commit riêng và đánh giá lại identity của run; không sửa rồi gọi kết quả là tái lập cùng snapshot.
