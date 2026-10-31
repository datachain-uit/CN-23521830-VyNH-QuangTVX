# Đưa source lên GitHub và giữ dữ liệu cục bộ

Mã nguồn pipeline nằm trong `data_pipeline/` của repo chung. Các lệnh dưới đây chạy tại thư mục đó. Bàn giao CD3/V4.2 dùng nhánh `feat/cd3-v4-2-data-handoff`, PR #4 vào `main`, chờ giảng viên xem theo QĐ40c. Cập nhật: 31/10/2026.

`.gitignore` dùng allowlist: chỉ source Python/notebook/shell, dependency specification và tài liệu hướng dẫn được chọn mới hiện thành ứng viên Git. Parquet, CSV, JSON đầu ra, archive, dataset, cache, checkpoint và các báo cáo chứa ví dụ dòng dữ liệu được giữ cục bộ. Các thư mục output vẫn đọc được khi chạy code.

Các tài liệu tổng hợp đã được mở riêng trong `.gitignore` và checker:

- `DATA_CARD_V4_2.md`: data card CD3/V4.2, gồm quy tắc teacher_subgroup.
- `BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md`: giải trình công thức mà bước 28 đọc khi tái tạo card.
- `GITHUB_REVIEW_2026_10_31.md`: bàn giao mã, chính sách công bố và kết quả kiểm thử source. Hồ sơ sự cố chi tiết ở riêng.

Các data card khác, báo cáo EDA và review Word vẫn ở cục bộ. Bundle `phase_features_v4_2/`, sidecar teacher_subgroup, reference ID, JSON audit và dữ liệu dòng được tạo lại bằng source 25–29; không mở toàn bộ thư mục report hoặc JSON.

Notebook `17_eda_final_quality_v2.ipynb` được ignore riêng vì có ví dụ người học thật trong source/Markdown. Dùng `17_eda_final_quality_v2.py` cho bản code công khai. Notebook khác được giữ source và xóa output; bản trước khi xóa nằm trong `.local_notebook_backups/`, đã bị ignore. Không force-add thư mục backup.

Sau mỗi lần chạy notebook, làm sạch output trước khi staging:

```bash
python3 scripts/clear_notebook_outputs.py
python3 scripts/check_github_payload.py --working-tree
git status --short
git add .
python3 scripts/check_github_payload.py --staged
git diff --cached --stat
```

Chỉ commit khi checker staged trả PASS và danh sách staged đúng phạm vi. Checker đọc nội dung thật trong index, chặn cả các định dạng dữ liệu bị force-add, notebook có output, learner ID viết trực tiếp và một số mẫu credential phổ biến. Ngoại lệ ID giả lập dạng `U_<n>` chỉ gồm hậu tố 1/2 trong `test_cq_tempo_features.py` và 1/4 trong `tests/test_cq_v4_2_builder.py`; các ID này được fixture tự tạo, không lấy từ dataset. Các ID khác hoặc cùng ID ở file khác vẫn bị kiểm tra. Không in giá trị credential hoặc nội dung dữ liệu ra màn hình.

Kiểm tra remote/branch bằng Git tại máy của bạn trước khi push. Không ghi token vào code/notebook hoặc URL remote trong tài liệu. Checker hỗ trợ pipeline nằm trong thư mục con của repo: nội dung staged được đọc theo đường dẫn của `data_pipeline/`, không nhầm với file cùng tên ở root repo.

`.gitignore` không loại file đã được theo dõi trong lịch sử Git. Không dùng `git add -f` để đưa dữ liệu đã ignore lên. Allowlist và kiểm tra nội dung source áp dụng cho pipeline; guard đuôi dữ liệu, con trỏ LFS và luật `filter=lfs` áp dụng toàn repo, kể cả `.gitattributes` ở gốc và trong `experiments/`. Checker không tự cài hook hay chạy tự động khi push.

Kiểm hồi quy trên commit dữ liệu đã rút (phải trả exit 1 và `BLOCKED`, không tải LFS):

```bash
python3 scripts/check_github_payload.py --tree 8c6a52c15f749182b00240d32266cae6cc242d03
python3 -m pytest -q -p no:cacheprovider tests/test_cq_patch_1031.py tests/test_github_payload.py
```
