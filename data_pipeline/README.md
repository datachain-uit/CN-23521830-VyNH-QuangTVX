# Phần dữ liệu

Thư mục này được giữ chỗ cho mã nguồn của nhóm dữ liệu. **Hiện chưa có code tạo dữ liệu được bàn giao vào repo này.** Không có dữ liệu, loader hoặc builder được người thực nghiệm tự đưa lên thay nhóm dữ liệu.

Nhánh dành cho nhóm dữ liệu: `codex/data-pipeline`. Nhóm dữ liệu tự đưa mã nguồn lên nhánh này và mở pull request khi sẵn sàng. Không tạo lại công thức/feature/split từ suy đoán để lấp phần đang thiếu.

Mã adapter đọc dữ liệu cho thực nghiệm đã tồn tại ở `experiments/src/skq_exp/data/`; đó không phải toàn bộ pipeline tạo V4 từ log nguồn. Không di chuyển hoặc nhân bản module này trong lần tổ chức repo vì sẽ làm thay import/package đã kiểm chứng.

Các file dữ liệu và archive giữ ở kho riêng theo quyền truy cập của nhóm; repo chỉ lưu mã nguồn, cấu hình mẫu không chứa bí mật và tài liệu giao tiếp đầu vào/đầu ra.
