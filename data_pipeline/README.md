# MOOCubeX — data model, final quality và phase features

Bản phát hành ngày 31/10/2026 chứa bốn nhóm dữ liệu và code liên quan theo phạm vi bàn giao đã chọn. Git quản lý source/tài liệu/metadata; **Git LFS quản lý các file Parquet**. Các snapshot được sao chép từ dữ liệu đã có, giữ nguyên bytes và SHA256; không tính lại nhãn, feature, preprocessing hoặc split khi đóng gói.

## Lấy dữ liệu

Cài Git LFS, clone nhánh phát hành hoặc `main` sau khi PR được merge, rồi chạy trong repo:

```bash
git lfs install --local
git lfs pull
git lfs fsck
```

Nếu chưa tải LFS, các file `.parquet` trong checkout có thể chỉ là con trỏ text. `git lfs pull` tải nội dung Parquet thật. Dung lượng của từng nhóm và SHA256 của từng file nằm trong [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json).

## Nội dung

| Thư mục | Mục đích |
|---|---|
| `input/moocubex_model/` | Các dimension, bridge, catalog chương và báo cáo chất lượng của bước 02. |
| `final_quality_v4/` | Chỉ số chất lượng cuối kỳ, q/y, cohort, course family và split. |
| `phase_features_v4/` | Feature snapshot V4 theo P1–P4 và FULL_A, các view raw/ready và loader. |
| `phase_features_v4_2/` | Feature CQ CD3: 85 raw, learner 76, selector 74; train/dev/test P1–P4 và FULL_A; metadata teacher_subgroup. |

Tên chuẩn là `phase_features_v4` và `phase_features_v4_2`. Dữ liệu phase là snapshot tích lũy tại cutoff, không phải một chuỗi sự kiện thô lấy mẫu đều theo thời gian.

- [DATA_CARD_V4.md](DATA_CARD_V4.md)
- [DATA_CARD_V4_2.md](DATA_CARD_V4_2.md)
- [Giải trình công thức](BANG_GIAI_TRINH_CONG_THUC_FEATURE_VA_NHAN_CQ_V4_2026_10_23.md)
- [Hướng dẫn tái tạo và phụ thuộc](REPRODUCING.md)

## Code được giữ

Các notebook 01–02 chuyển raw và tạo data model. Source 04–09 chuẩn bị event/thời gian/sentiment/phase; 15–16, 18–20 là các phụ thuộc của pipeline V4. Source 21–23 tạo final/phase V4; 25–29 tạo và kiểm tra V4.2, data card và teacher metadata. Các helper được import động được giữ cùng tên file để source hiện hành tiếp tục chạy.

Các notebook/report EDA độc lập và code audit ngoài chuỗi build này được bỏ khỏi bản phát hành. Lịch sử Git trước đây vẫn được giữ. Dữ liệu raw MOOCubeX, model weights, sentiment cache và các output V1–V3 không nằm trong bốn bộ dữ liệu phát hành; xem hướng dẫn tái tạo để chuẩn bị chúng khi cần build từ đầu.

## Kiểm tra

Từ thư mục repo cha:

```bash
python -m pip install -r data_pipeline/requirements-v4_2.txt
python -m pytest -q -p no:cacheprovider data_pipeline/test_cq_tempo_features.py data_pipeline/tests
python data_pipeline/26_verify_phase_features_v4_2.py --root data_pipeline/phase_features_v4_2 --report /tmp/phase_features_v4_2_validation.json
python data_pipeline/scripts/check_github_payload.py --working-tree
```

Trước commit dùng `--staged` thay cho `--working-tree` để kiểm tra blob thật trong Git index. Checker yêu cầu tất cả Parquet trong manifest được stage bằng LFS, đúng SHA256/kích thước, không còn symlink và không có file ngoài phạm vi đã chọn. Metadata của dataset có thể chứa ID quan sát; kiểm tra credential vẫn áp dụng cho các file text.

Staging/scratch/cache chưa được checksum bị loại khỏi gói. Các test log lịch sử và một loader bytecode đã nằm trong checksum của bundle cũ được giữ nguyên để bảo toàn kiểm tra gốc; chúng được đánh dấu `preserved_indexed_history` trong manifest. Không dùng các artifact đó thay cho source hoặc môi trường hiện hành.

## Phạm vi thực nghiệm

V4.2 có test feature theo yêu cầu bàn giao; test không dùng cho fit preprocessing hoặc QA23 và pipeline dữ liệu không huấn luyện mô hình. Việc đánh giá mô hình/test endpoint vẫn theo giao thức và quyết định freeze. Đóng gói LFS không phải quyết định phê duyệt M0/M1.
