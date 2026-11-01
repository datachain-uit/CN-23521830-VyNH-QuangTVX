# Bàn giao mã theo Review ngày 31 tháng 10 năm 2026

Repo công khai bàn giao mã và tài liệu tổng hợp theo QĐ40. PR #5 đã đóng không merge và nhánh `feat/publish-datasets-lfs` đã xóa trên GitHub. Mã CD3/V4.2 tiếp tục bàn giao qua PR #4, chờ giảng viên xem trước khi hợp nhất.

Xóa nhánh không tự xóa ref/cache PR và object LFS trên máy chủ. Không coi trạng thái PR đã đóng là bằng chứng purge. Quy trình của GitHub: [gỡ Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/removing-files-from-git-large-file-storage), [gỡ ref và cache PR](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Mã và quy tắc công bố

Giữ builder 25–29, helper, test và mã audit hiện có trên main. `cq_patch_1031.py` cung cấp các hàm trong Phụ lục A của Review: kiểm payload, tạo/kiểm manifest niêm phong và so hash artifact được bảo vệ. Việc có hàm tiện ích không tự chứng minh các công việc dữ liệu đã hoàn tất.

Checker giữ allowlist pipeline và bổ sung `payload_guard` trên toàn repo: chặn đuôi dữ liệu, con trỏ LFS dù mang tên source, luật `filter=lfs` trong `.gitattributes` ở gốc hoặc thư mục con, và entry không phải tệp thường. `--staged` đọc blob index; `--tree COMMIT` đọc Git blob, không checkout hay tải LFS. Manifest không tự cho phép dữ liệu.

```bash
python scripts/check_github_payload.py --working-tree
python scripts/check_github_payload.py --staged
python scripts/check_github_payload.py --tree 8c6a52c15f749182b00240d32266cae6cc242d03
python -m pytest -q -p no:cacheprovider test_cq_tempo_features.py tests
```

Lệnh kiểm commit phát hành đã rút phải trả exit 1 và `BLOCKED`. Sau khi object lịch sử được purge, chỉ test phụ thuộc commit này mới skip; fixture tổng hợp vẫn bắt buộc. Checker chạy thủ công, chưa có CI/hook bắt buộc phía GitHub; không chứng minh ref máy chủ đã purge.

## Kết quả kiểm tra source

- 44 test source PASS, gồm 11 test mới.
- Index công bố: 69 tệp pipeline và 330 entry toàn repo PASS.
- Commit `8c6a52c` trả exit 1 với `BLOCKED`: 464 tệp pipeline, 725 entry toàn repo; không tải LFS hay in nội dung dữ liệu.
- Giữ đủ 28 tệp mà PR #5 định xóa; chỉ tài liệu `GITHUB_CODE_ONLY.md` trong nhóm này được cập nhật, 27 tệp còn lại giữ nguyên byte so với main.
- Kiểm tra nguồn nhập gốc và `git diff --cached --check` đạt. Data card công khai không còn liên kết local hỏng.
- Tag `review-2026-10-26` trỏ đúng `75fbfbf2401730761596fa213213d07ec7be5dc3` theo Y38.

Data card công khai được hiệu đính từ số đo/checksum đã có ngày 30/10: thay 14 liên kết artifact riêng bằng tên tệp và SHA256, bổ sung nguồn dữ liệu và quy tắc sử dụng test. Không chạy lại builder hoặc đổi checksum bundle riêng. Builder 28 tái lập bản đo 30/10; lần tái tạo tài liệu sau cần áp dụng tiếp hiệu đính Review 31/10 và trạng thái thực tế. Hồ sơ sự cố chi tiết và xác nhận D06/niêm phong/hash panel được quản lý riêng theo Review.

Nguồn dữ liệu: [MOOCCubeX của THU KEG](https://github.com/THU-KEG/MOOCCubeX); Yu và cộng sự, “MOOCCubeX: A Large Knowledge-centered Repository for Adaptive Learning in MOOCs”, CIKM 2021, tr. 4643–4652. Công bố dữ liệu dẫn xuất về sau cần quyết định bằng văn bản của giảng viên (QĐ40e).

## Bổ sung ngày 02/11/2026

Y44 đã có câu xác nhận D06 đủ 80/80, không FAIL của Xuân Quang, MSSV 23521294 trong card và evidence riêng; audit tự động gốc giữ human_signoff=false. Y45 đã nối được input receipt panel với manifest đầu vào bảo tồn: 18 non-test khớp, 0 mismatch, 8 test mới thiếu phía manifest cũ. Các 26 artifact model-input bảo vệ được hash lại và giữ nguyên byte khi hiệu đính tài liệu.

Người dùng đã xác nhận niêm phong test local; n_files/manifest_sha256, phạm vi 244 tệp và trạng thái mở test được ghi ở card mục 15. Gói làm việc thực nghiệm đã tách train/dev/FULL_A. Chưa coi việc niêm phong local là bằng chứng quyền Drive hoặc email đã gửi. GitHub Support/purge vẫn cần admin; 0 fork được kiểm lại ngày 02/11, Traffic mới giữ trong hồ sơ riêng.

Toàn bộ workbench, code nghiên cứu, archive, model và dự đoán trong experiments_local/ được Git ignore; checker chặn cả source của vùng này nếu force-add. Không đưa code thực nghiệm vào PR công khai theo yêu cầu người dùng. Kiểm tra cập nhật: 13 test tập trung về payload/seal PASS và working-tree payload checker PASS. Những con số 44 test ở phần trước là kết quả source ngày 31/10, không phải kết quả chạy lại toàn bộ suite ngày 02/11.

Vault test đã niêm phong được tách ra ngoài workbench thực nghiệm và được Git ignore riêng; kiểm `find_test_files` trên toàn workbench trả danh sách rỗng. Checker chặn cả source force-add từ vùng bàn giao private này.
