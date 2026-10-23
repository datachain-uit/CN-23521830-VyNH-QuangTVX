# Nhánh bàn giao cho nhóm dữ liệu

Nhánh: `codex/data-pipeline`. Đây là nhánh chuẩn bị để nhóm dữ liệu tự đưa mã nguồn lên. **Hiện chưa có builder, loader hoặc notebook của nhóm dữ liệu được nhập.** Commit tạo nhánh chỉ thêm tài liệu bàn giao, không nhận phần công việc dữ liệu là do người thực nghiệm viết.

## Khi nhóm dữ liệu bắt đầu

1. Checkout nhánh này và tạo thư mục con trong `data_pipeline/` theo mã nguồn thực tế đã có; không sửa `experiments/` để lấp code còn thiếu.
2. Đưa code tạo label/feature/split, config mẫu không chứa bí mật và kiểm thử tương ứng lên; giữ ghi chú đầu file, phiên bản công thức và nguồn/phạm vi tác giả.
3. Viết README cách chạy từng bước, dependency, input/output schema và vị trí dữ liệu riêng tư. Không hardcode token, tài khoản hoặc đường dẫn máy cá nhân.
4. Mô tả giao diện bàn giao V4 cho nhóm thực nghiệm, hash/version, feature order, train-only preprocessing và cách tách runtime dev-only. Mọi thay đổi công thức/split/target phải được xét theo review và ghi deviation, không đổi theo kết quả model.
5. Không đưa dữ liệu, bảng ID, predictions, archive, model hoặc notebook có output lên GitHub. Chỉ gửi các file đó qua nơi lưu trữ có quyền truy cập được phép.
6. Mở pull request vào `main` khi code/test/tài liệu sẵn sàng; không ghi đè hoặc force-push `main`.

## Phân định với code thực nghiệm

`experiments/src/skq_exp/data/` chứa adapter và giao thức đang dùng bởi snapshot thực nghiệm; đó không thay cho builder từ raw log của nhóm dữ liệu. Việc tích hợp builder vào giao diện adapter sẽ được kiểm tra qua một thay đổi riêng, không tự sửa core/config rồi coi là cùng phiên bản.

Nhánh này kế thừa code thực nghiệm từ `main` vì các nhánh Git cùng chia sẻ nền dự án. Phần đóng góp dữ liệu mới chỉ nằm trong `data_pipeline/` cho đến khi hai bên thống nhất thay đổi khác.
