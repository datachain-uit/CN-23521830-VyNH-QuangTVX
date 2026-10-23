# Khóa luận CQ và thực nghiệm nén dữ liệu

Repo chung của nhóm, tổ chức phần dữ liệu và thực nghiệm riêng để dễ bàn giao.

| Phần | Nơi làm việc | Phạm vi |
|---|---|---|
| Thực nghiệm | `main`, thư mục [experiments/](experiments/) | Code, config, test và lock dependency từ nhánh `cq-v1-protocol-fixes` của repo thực nghiệm |
| Dữ liệu | Nhánh `codex/data-pipeline`, thư mục [data_pipeline/](data_pipeline/) | Nhánh bàn giao cho nhóm dữ liệu tự đưa mã nguồn lên; hiện chỉ có hướng dẫn, chưa có builder |
| Báo cáo, archive và dữ liệu | Folder Drive riêng do nhóm cung cấp | Không đưa dữ liệu hoặc kết quả mức dòng lên repo này |

## Nguồn code và tính nhất quán

Code thực nghiệm được nhập nguyên trạng từ:

- Repo: https://github.com/UIT23521830/skq-experiment
- Nhánh nguồn: `cq-v1-protocol-fixes`
- Commit nguồn: `ba2cbcd0f66b0d32c3215518c9741d676f072579`
- Người nhập snapshot: `UIT23521830`; việc nhập code không thay đổi quyền tác giả của mã nguồn gốc.

Đây là snapshot file, không sao chép hoặc viết lại lịch sử repo nguồn. `src/`, `configs/`, `tests/` và các script nhập vào không được sửa thuật toán/config trong lần chuyển này. Tài liệu mới ở repo chung chỉ hướng dẫn bố cục và truy vết. Những thay đổi chưa commit ở repo cá nhân không tự được nhập.

[SOURCE_IMPORT.json](SOURCE_IMPORT.json) và [SOURCE_SNAPSHOT_SHA256SUMS.txt](SOURCE_SNAPSHOT_SHA256SUMS.txt) ghi nguồn và hash từng file. Kiểm tra từ thư mục gốc:

```text
python tools/verify_source_snapshot.py
```

## Chạy thực nghiệm

Đọc [hướng dẫn CQ V4](experiments/RUN_CQ_V4.md). Mọi lệnh package chạy từ `experiments/`, không chạy từ thư mục gốc rồi tự đổi đường dẫn config. Với notebook mới, thư mục cài đặt editable là `<checkout>/experiments`.

Notebook đang chạy từ repo cá nhân vẫn giữ URL/commit và nguồn dữ liệu cũ để đối chiếu; không đổi commit giữa một lượt chạy. Kết quả lịch sử tiếp tục ghi commit nguồn, không thay bằng commit nhập repo chung.

## Quy tắc bàn giao

- Nhóm dữ liệu tự cập nhật code trên nhánh riêng; không nhận loader hoặc audit của nhóm thực nghiệm làm toàn bộ pipeline tạo dữ liệu.
- Không commit dữ liệu, model, predictions, selected indices, token, notebook có output hoặc archive kết quả.
- Các ví dụ/config legacy giữ để tái lập, không phải danh sách run đã được cô duyệt. Test CQ V4 thực vẫn khóa; test unit dùng dữ liệu tổng hợp không đồng nghĩa mở test nghiên cứu.
- A2/A4 hiện còn cần triển khai runtime; panel utility chưa được chạy trong lần nhập repo này. Có code/config không đồng nghĩa đủ GO hoặc đã freeze.
- Không vendor repo bên ngoài: dependency được tải đúng commit trong `experiments/external/official_repos.lock.json` và bị Git ignore.

Link Drive chung: **chờ nhóm bổ sung**. Việc tải mã nguồn lên GitHub không đồng nghĩa đã tải báo cáo lên Drive.
