# Cập nhật thực nghiệm theo review ngày 26 tháng 10 năm 2026

Cấu hình M0 được cô duyệt có điều kiện theo QĐ33; test vẫn khóa. Các hàm bổ sung
trong `cq_patch_1026.py` và test được trích nguyên từ phụ lục, có hash ở
`REVIEW1026_APPENDIX_SOURCE.json`. File code mới đều có ghi chú đầu trang.

| Yêu cầu | Cài đặt và sản phẩm |
|---|---|
| QĐ33 | `verdict_proposal` mặc định đúng năm seed 11/29/47/71/101; mean difference ≤ −δ/2 và ≥4/5 cùng chiều. `historical=True` giữ quy tắc cũ; K1 vẫn loại C03/native. |
| O32 | Endpoint `--review-1026`: giữ global A8 làm primary; thêm phase A8 và trạng thái không xác định khi thiếu phase/lớp. FullTrain-A không gán phase train giả. Wrapper kết luận báo cờ prior/A8-dependent khi có đủ số. |
| O33 | Đọc indices/weights từ method artifact, kiểm SHA256 rồi báo physical budget, positive support, zero share, ESS và max/mean theo 12 tầng. Không suy trọng số đều khi thiếu payload. |
| QĐ34 | FullTrain-B có AP floor cho từng model; mọi lượt có cờ E mô tả, không dùng cờ để loại coreset hoặc chọn cấu hình. |
| QĐ35 | Gctx chỉ khám phá từ raw Pool B train, trừ phase_fraction. `cd3_minus_ctx` giữ selector 74 cột, loại Gctx khỏi learner, fit lại V1 trên train và dùng runtime/contract riêng. Guard chỉ cho FullB/C02 ablation đã đăng ký. |
| Y34 | Extra dev chấp nhận pytest 9 đã dùng; psutil nhận bản7.0.0 đúng manifest; runner xuất version freeze và bắt buộc Python 3.13.2/NumPy-pandas 2.2.3/sklearn1.6.1/XGB3.0.0. |
| Y35/Y36 | Endpoint báo constant-marginal LL; runner thêm depth4 seed43/44 chỉ mô tả, không chọn lại cấu hình. |
| O34 | Calibration `--review-1026` dùng đúng 13 selector, loại CoreTab subset lịch sử khỏi danh sách 14 của lệnh cũ. Không đổi lệnh lịch sử. |

## Tái lập CPU

`scripts/run_cq_review2610_cpu.py` nhận các đường dẫn bằng CLI, kiểm hash
train/dev, khám phá Gctx, tạo bundle riêng rồi chạy đúng 10 fit/40 dev eval:
FullB minus-context ba model seed; C02 minus-context năm selector seed;
depth4 bổ sung hai model seed. Config, log, source fingerprint và model/dự đoán
được giữ. Source fingerprint phản ánh cả mã đang chỉnh trước commit; không ghi
nhầm fit này thành đã dùng một commit mới được tạo sau khi fit.

`scripts/analyze_cq_review2610_cpu.py` tính endpoint, CI2.000 lượt theo họ,
so sánh paired với bản main và replay 128 dòng P4 của từng model. Seed được
giữ cố định trong bootstrap; SD seed và CI theo họ có ý nghĩa khác nhau.

`scripts/report_cq_v50_tp_groups.py` chỉ đọc archive dev V50 để bổ sung số E
dự đoán đúng theo người học cũ/mới. Đây là V4 lịch sử, không là kết quả CD3 mới.

`scripts/materialize_cq_review2610_panel.py` sinh năm config panel, hai learner
đã duyệt, cùng ba template A2/A4 có nguồn khai báo. A3 đã đăng ký là random
trong ô của SKQ-PC standard; không tự gọi nó là source-matched LRQ. Script chỉ
chuẩn bị config, không chạy panel hoặc test. Muốn thực thi còn cần mã hợp nhất,
môi trường khóa và G10/QĐ16 của selector sắp chạy. D06/data card FAIL sẽ làm
kết quả panel phải hủy và chạy lại theo quyết định của cô.

## Dependency và môi trường

Faiss1.15.1/Cython3.3.0 được thử trong thư mục dependency riêng, không nâng
NumPy/sklearn/XGB chung. Wheel hashes và build logs thuộc hồ sơ thực nghiệm
riêng. GoodPoints phải build đúng commit
`181b12ade5dd02a61b65ed5cb9444c916fa3135f`, giữ direct_url provenance; không
thay bằng proxy. Nếu Windows thiếu C++ Build Tools, dùng Linux/Kaggle CPU.
Không có phương pháp nào trong phần sửa giao thức/ablation bắt buộc GPU.

PR #2/tag và PR cập nhật 26/10 báo theo trạng thái remote thật; code không tự
merge hoặc tự cấp GO chính thức. D06 ký thủ công, nguồn thời gian feature và
builder vẫn cần nhóm dữ liệu xác nhận. Giờ máy/người chạy phải do nhóm cung cấp.

Core lock và lock dependency Linux ở `experiments/environments/`; bản freeze toàn bộ môi trường đo nằm trong hồ sơ Drive. Suite311test được bổ sung2test riêng cho đơn vị thời gian/speed và biên cutoff của Y37, tổng313test khác nhau đã kiểm; không ghi đây là một lần gọi pytest313test.
