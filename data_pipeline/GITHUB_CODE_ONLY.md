# Đưa source lên GitHub và giữ dữ liệu cục bộ

Mã nguồn pipeline nằm trong `data_pipeline/` của repo chung. Các lệnh dưới đây chạy tại thư mục đó, trên nhánh `data-pipeline`; mở pull request vào `main` sau khi kiểm tra. Cập nhật: 23/10/2026.

`.gitignore` dùng allowlist: chỉ source Python/notebook/shell, dependency specification và tài liệu hướng dẫn được chọn mới hiện thành ứng viên Git. Parquet, CSV, JSON đầu ra, archive, dataset, cache, checkpoint và các báo cáo chứa ví dụ dòng dữ liệu được giữ cục bộ. Các thư mục output vẫn đọc được khi chạy code.

Các file `DATA_CARD*.md`, báo cáo EDA và review Word chưa nằm trong allowlist code-only. Nếu cần công bố tài liệu, chọn từng file sau khi kiểm tra nội dung; không mở toàn bộ thư mục report hoặc toàn bộ JSON.

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

Chỉ commit khi checker staged trả PASS và danh sách staged đúng phạm vi. Checker đọc nội dung thật trong index, chặn cả các định dạng dữ liệu bị force-add, notebook có output, learner ID viết trực tiếp và một số mẫu credential phổ biến. Không in giá trị credential hoặc nội dung dữ liệu ra màn hình.

Kiểm tra remote/branch bằng Git tại máy của bạn trước khi push. Không ghi token vào code/notebook hoặc URL remote trong tài liệu. Checker hỗ trợ pipeline nằm trong thư mục con của repo: nội dung staged được đọc theo đường dẫn của `data_pipeline/`, không nhầm với file cùng tên ở root repo.

`.gitignore` không loại file đã được theo dõi trong lịch sử Git. Không dùng `git add -f` để đưa dữ liệu đã ignore lên. Checker chỉ kiểm tra phần pipeline, không kiểm tra toàn bộ `experiments/`; nó không tự cài hook hay chạy tự động khi push.
