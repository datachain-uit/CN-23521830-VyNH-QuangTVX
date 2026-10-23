# Mở rộng dataset mà không sửa runner

Runner chỉ nhận processed contract gồm `X/y/row_ids` cho train, dev và test.
Dataset nguồn được cô lập trong `src/skq_exp/data/adapters` theo ba lớp:

1. `DatasetAdapter`: contract tối thiểu, dùng được cho CSV, Parquet hoặc nguồn khác.
2. `PresplitCSVAdapter`: dùng chung I/O, manifest, hash, group audit và temporal test.
3. `TrainOnlyOrdinalCSVAdapter`: thêm categorical mapping và median fit trên train.

CourseQuality chỉ kế thừa lớp 3 và khai báo phần riêng trong
`adapters/course_quality.py`. Lệnh CLI không biết tên CourseQuality:

```bash
skq prepare-external \
  --config configs/my_dataset.json \
  --dataset my_dataset_v1 \
  --input-dir /path/to/snapshot
```

## Thêm một CSV dataset pre-split

Tạo `adapters/my_dataset.py`:

```python
from .presplit_csv import PresplitCSVLayout, TrainOnlyOrdinalCSVAdapter


class MyDatasetAdapter(TrainOnlyOrdinalCSVAdapter):
    dataset_id = "my_dataset_v1"
    layout = PresplitCSVLayout(
        train_file="train.csv",
        dev_file="dev.csv",
        test_files=("test.csv",),
    )
    expected_rows_by_file = {
        "train.csv": 10000,
        "dev.csv": 2000,
        "test.csv": 2000,
    }
    expected_processed_features = 20
    target_column = "target"
    label_mapping = {"no": 0, "yes": 1}
    drop_columns = frozenset({"record_id"})
    group_columns = ("entity_id",)
    preprocessing_contract = "my_dataset_train_only_fit_v1"
```

Sau đó:

1. thêm metadata `DatasetSpec` vào `data/registry.py`;
2. import class và thêm instance vào `adapters/registry.py`;
3. thêm unit test bằng snapshot nhỏ;
4. chạy `skq prepare-external` và kiểm tra ba manifest.

Nếu dataset không dùng ordinal encoding, kế thừa trực tiếp `PresplitCSVAdapter`
và hiện thực `transform_frame`, `preprocessor_payload`, `schema_payload`. Không
được fit encoder/imputer/scaler trên dev hoặc test. `group_columns` phải phản ánh
đơn vị độc lập của nghiên cứu; nếu bỏ trống, paper evidence gate luôn đóng.
