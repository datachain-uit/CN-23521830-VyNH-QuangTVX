# CD3 receipt compatibility checks

The data builder exports Arrow timestamp[us] enroll_time, while the immutable
V4 source stores Arrow date32. Preparation compares these by exact timestamps,
without rounding to a day, dropping timezones or interpreting numeric epochs.
All other spine fields retain strict equality. A changed microsecond, date,
timezone or row key is rejected before preprocessing.

Preparation also sorts each input by row_id before fitting QA23/V1/category
maps. This matches the data builder's declared order and makes fit independent
of the physical order of rows in parquet files. The teacher-supplied feature
helper and selector algorithms remain unchanged. Regression checks include a
permutation of all input tables and an explicit microsecond-change rejection.

Source raw data, source ready data, and locally prepared views stay separate.
Numeric ready matrices are compared by row_id with explicit float tolerance;
their parquet bytes need not match across library/platform versions. QA23
needs_review/hold_out decisions are checked separately from rank/NMI values.
Cyclical sin/cos floating-point ties can change ranks across environments, so
do not report identical QA statistics when only decisions are reproduced.
Private receipt reports keep both audits and the numerical differences.

These checks do not certify event provenance, sign off D06, grant panel/test
approval, or choose a new feature set. Main CD3 and qa23_all keep separate
bundles/contracts even when there are no held-out columns.
