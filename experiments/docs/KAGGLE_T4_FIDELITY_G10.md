# Chạy fidelity gate G10 trên Kaggle Free T4

Đây là kiểm tra toy cho source/commit/interface của baseline, không phải kết quả
CQ v1 và không được dùng làm số liệu bài báo. Artifact không chứa row ID, index
đã chọn hoặc trọng số mức dòng.

## Điều kiện

- Accelerator của notebook: GPU T4. Script kiểm kê phải ghi đúng số GPU thực tế;
  không mặc định Kaggle luôn cấp hai T4.
- Repo ở đúng commit/branch cần kiểm.
- Internet bật trong giai đoạn cài package và fetch source chính thức.

## Lệnh

```bash
cd /kaggle/working/skq_experiment
python -m pip install -q -e ".[kaggle-panel]"
python scripts/fetch_official_repos.py
python scripts/capture_kaggle_environment.py \
  --output /kaggle/working/skq_artifacts/environment/kaggle_t4.json
python scripts/run_method_fidelity_toys.py \
  --device cuda \
  --external-root /kaggle/working/skq_experiment/external/repos \
  --output /kaggle/working/skq_artifacts/g10/toy_results.json
python scripts/audit_method_fidelity.py \
  --toy-evidence /kaggle/working/skq_artifacts/g10/toy_results.json \
  --require-passed-toys \
  --output /kaggle/working/skq_artifacts/g10/audit.json
```

`run_method_fidelity_toys.py` và audit strict trả mã `0` chỉ khi đủ 10/10
baseline success. Thiếu `faiss-cpu`, GoodPoints build lỗi, source sai commit hoặc
upstream trả lỗi đều phải giữ trạng thái riêng và làm gate fail; không thay proxy.

GoodPoints có Cython extension ``ktc``. Checkout trong ``external/repos`` dùng để
đối chiếu URL và commit; runtime phải import package đã được pip build từ đúng
commit khóa. Adapter đọc ``direct_url.json`` của distribution để kiểm tra URL và
full commit trước khi gọi ``goodpoints.kt.thin``. Không đưa checkout chưa build
vào ``sys.path`` vì làm như vậy sẽ che package runtime và tạo lỗi import giả.

## Evidence cần giữ private/Drive

- `environment/kaggle_t4.json`;
- `g10/toy_results.json`;
- `g10/audit.json`;
- log cài package/fetch source và repository commit.

Chỉ hash/manifest của evidence được phép đưa vào claim ledger public.

## Evidence đã xác minh ngày 16/10/2026

- Code commit: `fb4bd97a47f1c897b029199a4648b9b39d9d0317`.
- Strict audit: `passed`, 10/10 method `success`, không finding.
- Environment: Kaggle Free, hai Tesla T4, CUDA available, Python 3.13.15.
- Public manifest: `docs/evidence/manifests/EV-G10-FIDELITY-001.json`.
- Private archive SHA-256:
  `4fae924fd14f2a0032ba03ccdb007bf5a1ad6fcf8984bb180417a1bb3c5ab0e5`.

Đây là evidence fidelity/interface trên toy data, không phải kết quả hiệu năng CQ
v1. BDIS ở lượt này có contract `native_realized` và trả 68/120 dòng; không dùng
G10 để tuyên bố so sánh exact-budget hay ưu thế thực nghiệm.
