# Tái tạo pipeline và đóng gói dữ liệu

## Dùng dữ liệu đã phát hành

Bốn bộ dữ liệu có đủ artifact đã chọn để đọc các bảng, nhãn, feature và loader tương ứng sau `git lfs pull`. File `PUBLICATION_MANIFEST.json` là danh sách chính xác từng file, SHA256, số bytes và cờ LFS. Không có symlink hoặc đường dẫn máy cá nhân trong manifest phát hành.

Đầu vào builder gốc và manifest lịch sử bên trong các bundle giữ nguyên để truy nguyên lần chạy trước. Một đường dẫn tuyệt đối ghi trong manifest lịch sử không có nghĩa đường dẫn đó phải tồn tại trên máy mới; các loader dùng layout tương đối của bundle.

## Build từ raw

Không cần chạy lại toàn pipeline để đọc dữ liệu phát hành. Để tái tính từ đầu cần raw/cache và các output trung gian sau; chúng không được thêm vào bản phát hành ngoài phạm vi bốn thư mục.

| Đầu vào cần chuẩn bị | Dùng ở bước |
|---|---|
| Raw JSON/TXT/TSV MOOCubeX; `input/moocubex_parquet/` sau bước 01 | 01–02, chuẩn hóa event và các builder CQ. |
| `review_time_phases/tables/` và mapped problem/video event stages | 04–06 và 15–16. |
| Cache Cardiff cùng model lock: `afelo_sentiment_cardiff/` và phần bổ sung `final_quality_v2/sentiment/` | 07, 15 và 25/27. Không thay model/revision nếu muốn đối chiếu gói lịch sử. |
| `final_quality_v2/`, sau đó `final_quality_v3/` | 15 → 18 → 21. V4 kế thừa nhãn, không tính nhãn trực tiếp từ model catalog. |
| `phase_features_v2/`, `phase_features_v3/` và `fullA/` | 16 → 19/20 → 22. |
| Review 26–27/10, giải trình công thức, model locks và `_stage` từ build V4.2 | 28 nếu muốn tái tạo evidence/card; các nguồn này không cần để đọc X/y đã phát hành. |

Đặt đầu vào theo layout tương đối bên dưới `data_pipeline/`, hoặc dùng tham số đường dẫn của script khi có. Bước 01/02 giữ notebook gốc dùng Kaggle; đọc phần cấu hình input/output trong notebook trước khi chạy. Python builder dùng root theo vị trí file. Dependency sentiment nằm trong `afelo_sentiment_cardiff/requirements-sentiment.txt`; dependency V4 và V4.2 có file requirements riêng.

Chuỗi build chính, sau khi chuẩn bị đúng các đầu vào/cache:

```text
01 raw → Parquet
02 Parquet → dimensions/bridges
04/05/06 mapped events + times
07 sentiment cache; 15 quality V2; 18 quality V3; 21 quality V4
09/16 phase V2; 19 phase V3; 20 fullA; 22 phase V4; 23 data card V4
25 CQ V4.2 train/dev/FULL_A
27 CQ V4.2 test, transform bằng train policy đã đóng băng
29 teacher_subgroup
28 data card V4.2
26 kiểm tra độc lập V4.2
```

Một số bước chuẩn bị dùng output của EDA 04 như mapped event stage; vì vậy source 04 được giữ dù không xuất bản báo cáo EDA. Các script cũ tính lại dữ liệu có thể cần thời gian/môi trường riêng; không chạy toàn chuỗi chỉ để cập nhật Git.

## Tạo lại bản phát hành từ máy có local symlink

File `.local_data_links.json` là cấu hình máy cá nhân và được giữ ngoài Git. `prepare_dataset_release.py` dùng nó để xác nhận đích của `input/`, `final_quality_v4/`, `phase_features_v4/`; V4.2 có thể là thư mục thật không cần entry trong mapping.

Chạy từ một checkout khác với nơi giữ dữ liệu nguồn:

```bash
python data_pipeline/scripts/prepare_dataset_release.py \
  --source-root /path/to/source-repo/data_pipeline \
  --links /path/to/source-repo/data_pipeline/.local_data_links.json \
  --output-root /path/to/release-repo/data_pipeline
```

Script chỉ materialize bốn dataset, kiểm checksum nguồn và bản sao, copy code phụ thuộc và tạo manifest/allowlist. Giữ các file packaging, README và test publication trong checkout đích. Source không được xóa/di chuyển/sửa. Trên APFS, bản sao sử dụng copy-on-write khi khả dụng để giảm dung lượng tạm.

## Git LFS

Root repo khai báo LFS cho Parquet của đúng bốn dataset trong `.gitattributes`. Theo dõi LFS không tự mở `.gitignore`; allowlist phát hành được tạo theo từng file trong manifest. Muốn cập nhật dataset cần tạo lại manifest/allowlist rồi chạy checker trước commit. `git push` sử dụng pre-push hook để upload object LFS; các phiên bản mới của một Parquet được tính là object mới nếu nội dung đổi.
