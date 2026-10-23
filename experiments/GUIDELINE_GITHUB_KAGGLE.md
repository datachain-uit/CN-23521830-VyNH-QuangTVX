# Hướng dẫn GitHub → Kaggle

Tài liệu này hướng dẫn người chưa có ngữ cảnh đưa project lên GitHub, gắn dữ liệu
trên Kaggle và chạy đúng thứ tự. Dữ liệu raw và artifact lớn luôn nằm ngoài Git;
GitHub chỉ lưu code, config, test và các manifest nhỏ.

## 1. Chuẩn bị trước khi đưa lên GitHub

Từ thư mục project:

```powershell
cd C:\source\paper\skq_experiment
python -m pip install -e ".[kaggle-full,data,dev]"
python scripts\fetch_official_repos.py
pytest
skq doctor
```

Chỉ tiếp tục khi test đều đạt. Kiểm tra `git status` và bảo đảm không có file dữ
liệu, model hoặc predictions. `.gitignore` đã chặn các thư mục lớn, nhưng người
đẩy code vẫn phải đọc danh sách thay đổi trước khi commit.

## 2. Tạo GitHub repository

```powershell
git init
git add .
git status
git commit -m "Initialize reproducible SKQ experiment"
git branch -M main
git remote add origin https://github.com/<tai-khoan>/<ten-repo>.git
git push -u origin main
```

Thay `<tai-khoan>` và `<ten-repo>` bằng repository thật. Không lưu Kaggle token,
API key hoặc đường dẫn cá nhân trong code/config.

## 3. Dữ liệu nào cần có trên Kaggle

Tạo một hoặc nhiều Kaggle Dataset riêng tư, giữ nguyên các thư mục con sau:

```text
skq-public-data/
├── adult_uci2_v1/
├── creditcard_ulb2013_v1/
├── letter_uci59_v1/
├── covertype_uci31_v1/
├── jannis_openml41168_v1/
└── helena_openml41169_v1/
```

Nguồn cần tải:

- Adult: `https://archive.ics.uci.edu/dataset/2/adult`.
- Letter: `https://archive.ics.uci.edu/dataset/59/letter+recognition`.
- Covertype: `https://archive.ics.uci.edu/dataset/31/covertype`.
- Credit Card: Kaggle dataset `mlg-ulb/creditcardfraud`.
- Jannis: OpenML data ID `41168`.
- Helena: OpenML data ID `41169`.

Không gộp MOOCCubeX vào public-static dataset. MOOCCubeX cần Kaggle Dataset riêng
vì có manifest, feature và temporal split riêng.

## 4. Tạo Kaggle Notebook

1. Chọn **New Notebook**.
2. Bật Internet chỉ khi cần clone GitHub hoặc cài dependency.
3. Gắn Kaggle Dataset chứa public data bằng **Add Input**.
4. Chọn GPU chỉ cho MLP hoặc synthetic generation; selector và ML tabular có thể
   chạy CPU.
5. Bật lưu output vì artifact cần tải về sau mỗi stage.

Cell đầu tiên:

```bash
!git clone https://github.com/<tai-khoan>/<ten-repo>.git /kaggle/working/skq_experiment
%cd /kaggle/working/skq_experiment
!python -m pip install -q -e ".[kaggle-panel,data]"
!python scripts/fetch_official_repos.py
!skq doctor
```

`kaggle-panel` cài BDIS (`faiss-cpu`), AutoCoreset
(`iterative-stratification`), PyTorch và downstream learner nhưng chưa thay JAX
có sẵn của Kaggle. KIP chạy ở pha thứ hai sau khi panel đã lưu xong;
script `run_kaggle_split_env.py` sẽ đồng bộ `jax`, `jaxlib`,
`jax-cuda12-plugin`, `jax-cuda12-pjrt` về 0.4.38 và giữ
`neural-tangents==0.6.5`. Tất cả gói JAX/CUDA phải cùng version; chỉ pin
`jax/jaxlib` trong khi để plugin Kaggle bản mới sẽ gây lỗi `_sparse`.

Nếu Internet bị tắt, upload source dưới dạng Kaggle Dataset hoặc Notebook input,
sau đó copy vào `/kaggle/working/skq_experiment`. Không sửa code trực tiếp trong
cell vì thay đổi đó khó truy vết bằng commit.

## 5. Khai báo đường dẫn Kaggle

Trong config chạy, dùng:

```json
{
  "extends": "pilot_adult_full_seed11_v2.json",
  "paths": {
    "raw_root": "/kaggle/input/skq-public-data",
    "processed_root": "/kaggle/working/skq_data/processed",
    "artifact_root": "/kaggle/working/skq_artifacts",
    "external_root": "/kaggle/working/skq_experiment/external/repos"
  }
}
```

`/kaggle/input` chỉ đọc; mọi file tạm, processed data và artifact phải ghi vào
`/kaggle/working`.

## 6. Thứ tự lệnh trên Kaggle

```bash
# 1. Kiểm tra môi trường và test nhanh
!pytest -q
!skq smoke --config configs/s0_smoke.json

# 2. Tải/chuẩn bị Adult và kiểm tra manifest
!skq fetch-data --dataset adult_uci2_v1 --config configs/pilot_adult_full_seed11_v2.json
!skq prepare --dataset adult_uci2_v1 --config configs/pilot_adult_full_seed11_v2.json

# 3. Quick check rồi candidate matrix; LRQ được screen trên dev với parent khai báo trước
!skq run --config configs/pilot_adult_quick_seed11.json --selector-seed 11
!python scripts/run_kaggle_split_env.py \
  --config configs/kaggle_pilot.json \
  --dataset adult_uci2_v1 \
  --artifact-root /kaggle/working/skq_artifacts \
  --archive /kaggle/working/skq_results.tar.gz \
  --max-ram-gb 12 --timeout-seconds 7200 --max-threads 4

# 4. Sau dev screen, tạo freeze_manifest schema v3 theo docs/freeze_manifest.example.json.
#    Chỉ sau đó kết quả P04/P05 mới được phép đi vào confirmatory/test.
!skq run --config configs/s2_confirm.json --dataset adult_uci2_v1

# 5. Script hai pha đã aggregate và đóng gói; chỉ cần tải skq_results.tar.gz.
```

Hai pha dùng cùng config, split, seed, `experiment_id`, artifact root và
protocol hash. Pha panel gọi `--exclude-method s_kip_tdbench`; pha KIP gọi
`--method s_kip_tdbench` trong process mới. Runner gộp ledger theo `run_id`,
aggregate quét tất cả manifest, rồi archive chứa cả hai pha. File
`kaggle_execution_summary.json`, `environment_panel.txt` và
`environment_kip.txt` lưu provenance môi trường.

### CourseQuality MED: full matrix trên bốn test

Sau `prepare-external`, tạo freeze chuyển giao đã khai báo trước rồi chạy cùng
orchestrator. Freeze này không tuyên bố là winner chọn từ CQ; nó khóa trước hai
parent source CoreTab-XGB và Gonzalez từ protocol Adult, trước khi đọc metric CQ
test. Cấu hình v4 chứa đủ 28 phương pháp của panel Adult hiện hành:

```bash
!skq freeze --config configs/kaggle_course_quality_temporal_v4_flat.json \
  --base-winner-method p02_skq_coretab_xgb \
  --base-winner-structure n02_coretab_xgb_subset \
  --published-reference n02_coretab_xgb_subset \
  --proposed-winner p08_skq_gonzalez_lrq_sq \
  --selection-basis transferred_full_adult_v4_portfolio_predeclared_before_cq_test \
  --allowed-parent-source n02_coretab_xgb_subset \
  --allowed-parent-source gonzalez_pool_2x

!python scripts/run_kaggle_split_env.py \
  --config configs/kaggle_course_quality_temporal_v4_flat.json \
  --dataset course_quality_med_v1 \
  --artifact-root /kaggle/working/skq_artifacts \
  --archive /kaggle/working/skq_results.tar.gz \
  --prepare-autocoreset \
  --max-ram-gb 24 --timeout-seconds 21600 --max-threads 4
```

Plan full v4 là 28 method × 5 learner và bốn temporal test, tức 560 ledger row.
Hai `--allowed-parent-source` chỉ mở các parent đã hiện diện trong config và đã
được khai báo trước; chúng không chọn winner từ CQ và không đọc metric test.
Với CQ lớn, có thể lặp `--include-method METHOD_ID` để chia 28 method qua nhiều
Kaggle job. Đây chỉ là bộ lọc lịch chạy: mỗi job vẫn dùng nguyên config và freeze,
nên protocol hash không đổi. Các partition phải rời nhau và hợp lại đủ 28 method.
Khi chạy một notebook liền mạch, nên thêm
`--autocoreset-timeout-seconds 1800`: boundary AutoCoreset vẫn được thử đúng
source nhưng không thể giữ toàn bộ panel vô thời hạn trên dữ liệu hàng triệu dòng.
`--prepare-autocoreset` chạy boundary native trước panel; nếu upstream không hoàn
tất, runner vẫn ghi trạng thái thật thay vì bỏ method. KIP bị resource gate hợp
lệ không làm mất archive hoặc khiến Save & Run bị đánh lỗi hạ tầng.

Runner hiện ghi đè đúng run directory khi cùng run ID và gộp ledger theo run ID,
nhưng chưa có scheduler tự bỏ qua mọi run thành công. Vì vậy nên chạy theo method
hoặc checkpoint sau từng nhóm trên Kaggle; không giả định lệnh bị ngắt sẽ tự resume
từ đúng vòng lặp bên trong selector.

Trong lần chạy đầu, luôn thêm `--selector-seed 11`. Sau khi artifact và metric đã
được kiểm tra, có thể chạy nhiều seed bằng cách lặp option:

```bash
!skq run --config configs/pilot_adult_full_seed11_v2.json --dataset adult_uci2_v1 \
  --selector-seed 11 --selector-seed 29 --selector-seed 47
```

Trên máy có RAM hạn chế, đặt guard ngay trong lệnh:

```bash
!skq run --config configs/pilot_adult_full_seed11_v2.json --dataset adult_uci2_v1 \
  --selector-seed 11 --max-ram-gb 12 --timeout-seconds 7200 \
  --max-estimated-operations 50000000000
```

Mặc định không lưu fitted model; selection, prediction nén, metric, cost và manifest
vẫn được giữ. Chỉ thêm `--save-model` cho run cần checkpoint.

## 7. Checkpoint để tránh mất kết quả

Sau mỗi dataset hoặc selector seed, nén artifact:

```bash
!tar -czf /kaggle/working/skq_checkpoint.tar.gz -C /kaggle/working skq_artifacts
```

Chọn **Save Version** để Kaggle giữ file trong Output. Khi chạy notebook mới, gắn
output của version trước làm input rồi giải nén sang `/kaggle/working`.

## 8. Repo đối chứng

Các repo tác giả được khóa commit trong `external/official_repos.lock.json`.
CoreTab và BDIS có adapter native trực tiếp; CRAIG feature-space là adaptation R1.
AutoCoreset dùng driver environment
riêng; KIP/MTT/GM/DATM giữ dependency nặng của TDBench. Output đều mang commit/provenance.

Quy trình:

1. Chạy `python scripts/fetch_official_repos.py` để clone/kiểm commit.
2. Lưu `pip freeze`, lệnh chạy và patch nếu có.
3. Chạy native smoke trên dataset anchor.
4. Kiểm tra row identity và output type.
5. AutoCoreset chạy `python scripts/run_autocoreset_native.py --config ... --dataset ... --seed 11`; chỉ artifact qua gate mới được runner nhập.

KIP cần JAX và neural-tangents; MTT/GM/DATM cần PyTorch. Các method này không bị chặn bởi trần
500 dòng nữa: budget 5% là biến thực nghiệm, còn preflight chỉ chặn khi ước lượng
RAM hoặc số phép tính vượt policy. OOM/timeout runtime được ghi đúng loại tài
nguyên. Với Adult, KIP-TDBench yêu cầu target batch `10*N` mỗi lớp không hoàn lại,
nên requested 1.384 dòng được giữ trong manifest nhưng source chỉ khả thi ở
`N=666` mỗi lớp, tức realized 1.332 dòng. MTT vẫn nhắm 692 mỗi lớp = 1.384 dòng.
Đây là giới hạn contract của source KIP, không phải lỗi hay thay đổi thuật toán.

## 9. Tải kết quả về máy

Từ trang Output của Kaggle tải:

- `skq_checkpoint.tar.gz`;
- bảng tổng hợp CSV/JSON;
- log lỗi/OOM/timeout;
- freeze manifest;
- hình và bảng dùng cho bài báo.

Giải nén vào một thư mục artifact riêng, không ghi đè dữ liệu local chưa kiểm
tra. Sau đó chạy `skq aggregate` để tái tạo leaderboard và kiểm tra run IDs.

## 10. Những lỗi cần tránh

- Không upload raw data hoặc token lên GitHub.
- Không dùng test để chọn base winner.
- Không chạy FullTrain lặp lại theo selector seed.
- Không ép native method về exact 5% bằng trim/pad.
- Không đổi method khi OOM rồi giữ nguyên tên trong bảng.
- Không ghi metric không xác định thành 0.
- Không chỉnh portfolio query, kernel hoặc budget sau khi đã xem test.
