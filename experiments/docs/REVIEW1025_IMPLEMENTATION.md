# Thực nghiệm theo review ngày 25 tháng 10 năm 2026

Phạm vi: chuẩn bị code và tính lại artifact dev đã có. Không có fit CQ/CD3 mới,
không mở test, không chạy panel và không tự duyệt learner, margin hay rho.

## Phần đã cài

| Yêu cầu | Mã và cách thực hiện |
|---|---|
| QĐ29 | Giữ `cq_patch_1025.phase_gap_report`; gate S−A chỉ bước 1 với cùng CD3/L0/seed42. Bộ sinh follow-up mới thêm S cấu hình cuối để báo cáo, không kiểm lại gate |
| QĐ30 | `--review-1025` của endpoint xuất UL thô, ULA8, ul_decision, EC sau A8; thiếu lớp giữ cờ và fallback UL thô. Wrapper verdict yêu cầu seeds11/29/47; K1 loại C03/native; headroom chọn trên A8 và báo raw song song |
| QĐ31 | Cold-start theo họ khóa, warm-start thuộc RQ5 GĐ2; full-validation là A. Không thêm bộ TEMPO-DQ gốc hoặc huấn luyện tiếp qua phase |
| QĐ32 | NLL/AUC/AP E, precision/recall, calibration, số họ có E và có E đúng; từng phase và gộp. CI lấy mẫu họ chung cả bốn phase; thuật toán weighted tương đương appendix được kiểm với score ties |
| Y29 | `feature_variant=qa23_all`, `feature_view=cd3`; bundle/runtime riêng; config hash và guard so variant. FullTrain-B seed42 natural, không là ứng viên; lưu gain của các cột QA giữ lại |
| Y30 | Ready learner columns là V1 dưới tên cũ; raw_features mới dùng cho D06/QA23. Script audit phân biệt LL thô/A8 của bảng hai nhãn |
| Y32 | Script đọc V20/V50/learner_code, kiểm dòng/nhãn/họ, xuất subgroup, calibration, E false positives, rounds đã lưu, code gain và course contributions |

Patch/test cô gửi được trích cơ học, không sửa thân hàm; hash nguồn ở
`REVIEW1025_APPENDIX_SOURCE.json`. Test khớp A8 với cài đặt của repo phải chạy,
không được bỏ qua như khi cô chạy độc lập.

`dp_by_phase` dùng prior hiệu dụng của các dòng chọn **trong từng phase**, không
nhầm với prior gộp của model dùng trong A8. Phase không được chọn có dp chưa xác
định, không thay ngầm bằng prior gộp. Runner lưu prior từng phase cho run mới.

## Lệnh sau khi có CD3 thật

Chạy trong `experiments/`, cài package và extras cần thiết theo pyproject.
Các đường dẫn dưới đây phải trỏ tới bundle dev-only đã kiểm nhận; không mount
test vào runtime thực nghiệm. Không chạy lưới/headroom trên V4 để thay CD3.

```powershell
python scripts/prepare_cq_cd3_dev.py --raw-root <CD3_RAW_DEV_ONLY> --v4-bundle <V4_RUNTIME_DEV_ONLY> --output-bundle <FRESH_CD3_BUNDLE> --processed-root <FRESH_CD3_PROCESSED>
python scripts/prepare_cq_cd3_dev.py --raw-root <CD3_RAW_DEV_ONLY> --v4-bundle <V4_RUNTIME_DEV_ONLY> --output-bundle <FRESH_QA23_ALL_BUNDLE> --processed-root <FRESH_QA23_ALL_PROCESSED> --feature-variant qa23_all
python scripts/compute_cq_v4_dev_endpoints.py --input-dir <CD3_BUNDLE> --artifacts <DEV_ARTIFACTS> --output <FRESH_ENDPOINT_JSON> --review-1025
python scripts/compute_cq_review2310_headroom.py --input <AUDITED_HEADROOM_JSON> --output <FRESH_PROPOSAL_JSON> --review-1025
```

Endpoint A8 dùng prior Pool B train theo phase, không dùng prior dev/test làm
target. Headroom JSON có `primary_endpoint=mean_log_loss_A8_P1_P4`,
`c02_by_ratio` là `ul_decision` (A8 nếu xác định), `c02_raw_by_ratio` song song,
`full_by_model_seed` đủ42/43/44, bốn budget và năm selector seed. Lượt thiếu lớp
được liệt kê trong `a8_undefined_runs`, không giả vờ có A8.

Sau lựa chọn đo thực, `materialize_cq_review2510_configs.py` tạo 13 follow-up
config: 11 kế thừa, S cuối, qa23_all. Receipt lựa chọn phải xác nhận gate bước1,
hash config và feature contract, split dev/test_data_read=false. Không sinh
config panel/test. `qa23_all` không dùng để chọn config chính.

## Phần chưa hoàn tất bằng code

CD3 thật, QA23 thật, mapping teacher_subgroup, D06 thủ công, data card sửa và
giờ tài nguyên còn lại cần người phụ trách cung cấp. Tag/PR/quyền Drive được
báo theo trạng thái thật trong hồ sơ, không tự coi chuẩn bị code là đã xong.
`references.bib` hiện có bốn tài liệu nền tảng, chưa là danh mục hoàn chỉnh Y17.

Môi trường kiểm thử local có pytest9.1.1, khác extra dev khai báo pytest<9;
báo cáo JUnit/environment đi kèm phải giữ khai báo này. Toy integration có fit
CPU nhỏ, không được tính là chất lượng CQ/CD3. Các benchmark V4 vẫn là lịch sử.
