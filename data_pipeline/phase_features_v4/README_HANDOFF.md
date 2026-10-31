# CQ V4 bàn giao dữ liệu

Trạng thái: DATA_BUILD_PASS; còn probe và kiểm tra mô hình theo handoff_status.json.
Đọc DATA_CARD_V4.md ở thư mục working_13-10 để xem đầy đủ thống kê/thay đổi.
Card trong runtime đã đổi đường dẫn sang các schema/preprocessor/loader đi kèm;
các nguồn private không phân phối được ghi bằng tên thay vì link hỏng.

Runtime này chỉ chứa train/dev Pool B và FullTrain-A train. Test feature/label
không có trong bundle. splits.json chỉ mô tả assignment theo family/course.
Loader trả relation có q,y và metadata; chỉ lấy X theo feature_columns(view).
Không fit lại preprocessor hoặc tự lấy mọi cột số làm X.

FullTrain-A có phase=FULL_A, phase_fraction=1 trước shared scaling.
Nó dùng đúng cohort train và không được ghép vào Pool B P1-P4.
SHA256SUMS.txt kiểm tra mọi file của gói, trừ chính checksum list.

Root private này có cả test; chỉ gửi runtime_dev_only/ cho notebook dev.
