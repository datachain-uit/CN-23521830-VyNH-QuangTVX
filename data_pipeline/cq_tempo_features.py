"""Đặc trưng CQ theo TEMPO-DQ (mã nguồn CD3), dựng trên khung V4 của nhóm (QĐ28).

Nguồn: github.com/datachain-uit/CD3, commit 5b14052.
- CQ/Feature_extraction/features/build_scenario_phase_features.py: video_features,
  problem_features, comment_features, merge_features, select_events_until_cutoff.
- CQ/Feature_extraction/processing/process_*.py: làm sạch sự kiện; ngữ cảnh khóa,
  giảng viên, trường.
- imputation_core/wide_imputation.py, model/data.py: quy tắc đầu vào CQ_RAW_EARLY;
  tiền xử lý V1-Median, fit trên train gộp P1–P4 (TRAIN_POOLED_P1_P4).
- CQ/Feature_extraction/validation/audit_cq_label_proxy.py: audit QA23.

Không lấy: nhãn CQ của CD3 (nhóm giữ nhãn theo QĐ27); suy luận lịch mở khóa, khối thời
gian và window_duration_seconds (dùng mốc cắt V4); tác vụ LO; các biến thể V0, V2–V16;
SMOTE; mô hình chuỗi.

Khung (spine): một dòng mỗi lượt ghi danh × phase, lấy nguyên từ V4 (cột META của
22_build_phase_features_v4.py, kể cả snapshot FULL_A): enrollment_id, user_id,
course_id, phase, enroll_time, cutoff_time. Bảng sự kiện mang enrollment_id (ghép theo
user_id, course_id) và dùng cùng đồng hồ với cutoff của V4 (không đổi múi giờ như CD3).

Quy trình: build_cd3_phase_features (hoặc build_in_course_chunks) → qa23_audit trên
Pool B train (từng phase P1–P4) → CQRawEarlyViews.fit(Pool B train,
drop=held_out_columns(audit)) → transform cho Pool B, dev P1–P4 và FULL_A.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PHASE_FRACTION = {"P1": 0.25, "P2": 0.50, "P3": 0.75, "P4": 0.90,
                  "FULL_A": 1.0}                      # FULL_A: snapshot λ = 1 của V4
SESSION_GAP_SECONDS = 30 * 60
KG = ["enrollment_id", "phase", "course_id"]


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    """try_divide của Spark: NULL khi mẫu số bằng 0 hoặc NULL."""
    return a / b.where(b != 0)


def _days(t: pd.Series, e: pd.Series) -> pd.Series:
    """datediff của Spark: hiệu số ngày lịch, bỏ giờ."""
    return (t.dt.normalize() - e.dt.normalize()).dt.days.astype(float)


def _epoch_seconds(t: pd.Series) -> pd.Series:
    """unix_timestamp của Spark (giây, làm tròn xuống); không phụ thuộc độ phân giải."""
    epoch = pd.Timestamp("1970-01-01", tz=t.dt.tz)
    return ((t - epoch) // pd.Timedelta(seconds=1)).astype(float)


def _select(events: pd.DataFrame, spine: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """select_events_until_cutoff: enroll_time <= availability_time <= cutoff_time."""
    cols = ["enrollment_id", "phase", "course_id", "enroll_time", "cutoff_time"]
    x = events.drop(columns=[c for c in cols[1:] + ["user_id"] if c in events])
    x = x.merge(spine[cols], on="enrollment_id")
    return x[(x[time_col] >= x["enroll_time"]) & (x[time_col] <= x["cutoff_time"])].copy()


# 1. Làm sạch sự kiện (process_video.py, process_problem.py, process_comment.py) -----
def clean_video_events(v: pd.DataFrame) -> pd.DataFrame:
    ok = (v["start_point"].notna() & v["end_point"].notna()
          & (v["end_point"] >= v["start_point"])
          & (v["speed"] > 0) & v["event_time"].notna())
    return v[ok].copy()


def clean_problem_events(p: pd.DataFrame) -> pd.DataFrame:
    ok = (p["submit_time"].notna() & (p["full_score"] > 0)
          & p["attempts"].notna() & (p["attempts"] >= 0))
    return p[ok].copy()


def clean_comment_events(c: pd.DataFrame) -> pd.DataFrame:
    """Tập comment trước lọc text (C_all của V4). CD3 khử trùng (comment_id, course_id)
    trước khi ghép lượt ghi danh; ở đây sự kiện đã mang enrollment_id nên khóa khử trùng
    gồm cả enrollment_id, để mỗi lượt ghi danh của cùng (user, course) giữ comment."""
    c = c[c["comment_id"].notna() & c["event_time"].notna()]
    keys = [k for k in ("comment_id", "course_id", "enrollment_id") if k in c]
    return c.drop_duplicates(keys).copy()


# 2. Ba họ đặc trưng tích lũy đến mốc cắt --------------------------------------------
def video_features(events: pd.DataFrame, spine: pd.DataFrame,
                   course_ctx: pd.DataFrame) -> pd.DataFrame:
    x = _select(events, spine, "event_time")
    x["segment_duration"] = (x["end_point"] - x["start_point"]).astype(float)
    x["real_watch_time"] = x["segment_duration"] / x["speed"]
    x["day_of_week"] = (x["event_time"].dt.dayofweek + 1) % 7 + 1   # Spark: 1 = CN
    x["hour"] = x["event_time"].dt.hour
    x["watch_day"] = x["event_time"].dt.normalize()
    x["days_from_enroll"] = _days(x["event_time"], x["enroll_time"])
    x["fast_forward"] = (x["speed"] > 1).astype(float)
    g = x.groupby(KG)
    out = pd.DataFrame({
        "segment_duration_seconds": g["segment_duration"].sum(),
        "avg_segment_duration": g["segment_duration"].mean(),
        "avg_real_watch_time": g["real_watch_time"].mean(),
        "segment_count": g.size().astype(float),
        "avg_playback_speed": g["speed"].mean(),
        "avg_day_of_week": g["day_of_week"].mean(),
        "unique_video_count": g["resource_id"].nunique().astype(float),
        "video_active_day_count": g["watch_day"].nunique().astype(float),
        "first_video_day_from_enroll": g["days_from_enroll"].min(),
        "last_video_day_from_enroll": g["days_from_enroll"].max(),
        "video_day_from_enroll_std": g["days_from_enroll"].std(),
        "min_segment_duration": g["segment_duration"].min(),
        "max_segment_duration": g["segment_duration"].max(),
        "segment_duration_std": g["segment_duration"].std(),
        "var_playback_speed": g["speed"].var().fillna(0.0),
        "fast_forward_ratio": g["fast_forward"].mean(),
    })
    n, uv = out["segment_count"], out["unique_video_count"]
    out["avg_segment_count_per_video"] = _div(n, uv)
    out["video_repeat_segment_ratio"] = _div(n - uv, n)
    out["video_events_per_active_day"] = _div(n, out["video_active_day_count"])
    out = out.join(_mode(x, "hour", "most_common_hour"))
    out = out.join(_mode(x, "watch_day", "most_common_day"))
    out = out.join(_session_std(x)).reset_index()
    ctx = course_ctx[["course_id", "video_counts"]].rename(
        columns={"video_counts": "course_video_count"})
    out = out.merge(ctx, on="course_id", how="left")
    out["video_catalog_coverage"] = _div(out["unique_video_count"],
                                         out["course_video_count"])
    return out


def _mode(x: pd.DataFrame, col: str, name: str) -> pd.Series:
    """Giá trị xuất hiện nhiều nhất; hòa thì lấy giá trị nhỏ nhất."""
    c = x.groupby(KG + [col]).size().rename("n").reset_index()
    c = c.sort_values(KG + ["n", col], ascending=[True] * len(KG) + [False, True])
    return c.drop_duplicates(KG).set_index(KG)[col].rename(name)


def _session_std(x: pd.DataFrame) -> pd.Series:
    """Phiên mới khi hai segment liên tiếp (theo event_time, resource_id) cách nhau hơn
    30 phút; độ dài phiên = max(t + real_watch_time) − min(t); SD qua các phiên
    (một phiên thì 0)."""
    x = x.sort_values(KG + ["event_time", "resource_id"])
    ts = _epoch_seconds(x["event_time"])
    keys = [x[k] for k in KG]
    gap = ts - ts.groupby(keys).shift()
    sid = (gap.isna() | (gap > SESSION_GAP_SECONDS)).astype(int).groupby(keys).cumsum()
    s = pd.DataFrame({**{k: x[k] for k in KG}, "sid": sid, "start": ts,
                      "end": ts + x["real_watch_time"]})
    s = s.groupby(KG + ["sid"]).agg(start=("start", "min"), end=("end", "max"))
    dur = (s["end"] - s["start"]).groupby(level=KG).std().fillna(0.0)
    return dur.rename("session_duration_std")


def problem_features(events: pd.DataFrame, spine: pd.DataFrame,
                     catalog: pd.DataFrame) -> pd.DataFrame:
    x = _select(events, spine, "submit_time")
    x["score_clean"] = pd.to_numeric(x["score"], errors="coerce").fillna(0.0)
    x["score_fraction"] = _div(x["score_clean"], x["full_score"].astype(float))
    x["submit_day"] = _days(x["submit_time"], x["enroll_time"])
    correct = pd.to_numeric(x["is_correct"], errors="coerce").eq(1)     # NULL → 0
    x["correct"] = correct.astype(float)
    x["incorrect"] = 1.0 - x["correct"]
    x["attempts_clean"] = pd.to_numeric(x["attempts"], errors="coerce").fillna(0.0)
    g = x.groupby(KG)
    out = pd.DataFrame({
        "problem_count": g["problem_id"].nunique().astype(float),
        "exercise_count": g["exercise_id"].nunique().astype(float),
        "problem_context_count": g["context_id"].nunique().astype(float),
        "score_sum": g["score_clean"].sum(), "score_mean": g["score_clean"].mean(),
        "score_std": g["score_clean"].std(),
        "score_fraction_mean": g["score_fraction"].mean(),
        "score_fraction_std": g["score_fraction"].std(),
        "correct_problem_count": g["correct"].sum(),
        "incorrect_problem_count": g["incorrect"].sum(),
        "correct_ratio": g["correct"].mean(), "correct_ratio_std": g["correct"].std(),
        "attempts_sum": g["attempts_clean"].sum(),
        "attempts_mean": g["attempts_clean"].mean(),
        "attempts_std": g["attempts_clean"].std(),
        "first_submit_day_from_enroll": g["submit_day"].min(),
        "last_submit_day_from_enroll": g["submit_day"].max(),
        "avg_submit_day_from_enroll": g["submit_day"].mean(),
        "submit_day_from_enroll_std": g["submit_day"].std(),
    }).reset_index()
    cat = catalog.drop_duplicates(["course_id", "problem_id"]).groupby("course_id")
    totals = pd.DataFrame({                       # F.sum: NULL khi mọi full_score NULL
        "course_problem_count": cat["problem_id"].nunique().astype(float),
        "course_problem_full_score_sum": cat["full_score"].sum(min_count=1),
    }).reset_index()
    out = out.merge(totals, on="course_id", how="left")
    out["problem_catalog_coverage"] = _div(out["problem_count"],
                                           out["course_problem_count"])
    out["problem_score_catalog_coverage"] = _div(out["score_sum"],
                                                 out["course_problem_full_score_sum"])
    return out


SENTIMENT = ("positive", "negative", "neutral")


def comment_features(events: pd.DataFrame, spine: pd.DataFrame,
                     course_ctx: pd.DataFrame,
                     use_sentiment: bool = True) -> pd.DataFrame:
    """text là nội dung gốc (CD3 v2 đặt text_translated = text_original). Sentiment lấy
    từ dự đoán Cardiff đã có của nhóm (nhãn được đưa về chữ thường; CD3 so khớp đúng
    chữ “positive”, “neutral”, “negative”); use_sentiment=False tái lập bản CD3 hiện
    tại (các cột sentiment NULL)."""
    x = _select(events, spine, "event_time")
    x["text_length"] = x["text"].astype("string").str.strip(" ").str.len().astype(float)
    x["comment_day"] = _days(x["event_time"], x["enroll_time"])
    if use_sentiment and "sentiment_label" in x:
        label = x["sentiment_label"].astype("string").str.strip().str.lower()
    else:
        label = pd.Series(pd.NA, index=x.index)
    for s in SENTIMENT:
        col = f"{s}_score"
        x[col] = x[col].astype(float) if use_sentiment and col in x else np.nan
        x[f"is_{s}"] = (label == s).fillna(False).astype(float)
    g = x.groupby(KG)
    data = {
        "comment_count": g.size().astype(float),
        "unique_commented_resource_count": g["resource_id"].nunique().astype(float),
        "avg_comment_length_chars": g["text_length"].mean(),
        "min_comment_length_chars": g["text_length"].min(),
        "max_comment_length_chars": g["text_length"].max(),
        "comment_length_std_chars": g["text_length"].std(),
    }
    for s in SENTIMENT:
        data[f"avg_{s}_score"] = g[f"{s}_score"].mean()
    for s in SENTIMENT:
        data[f"{s}_score_std"] = g[f"{s}_score"].std()
    for s in ("positive", "neutral", "negative"):
        data[f"{s}_comment_count"] = g[f"is_{s}"].sum()
    data["positive_comment_ratio"] = g["is_positive"].mean()
    data["negative_comment_ratio"] = g["is_negative"].mean()
    data.update({"first_comment_day_from_enroll": g["comment_day"].min(),
                 "last_comment_day_from_enroll": g["comment_day"].max(),
                 "avg_comment_day_from_enroll": g["comment_day"].mean(),
                 "comment_day_from_enroll_std": g["comment_day"].std()})
    ctx = course_ctx[["course_id", "resource_count"]].rename(
        columns={"resource_count": "course_resource_count"})
    out = pd.DataFrame(data).reset_index().merge(ctx, on="course_id", how="left")
    out["commented_resource_catalog_coverage"] = _div(
        out["unique_commented_resource_count"], out["course_resource_count"])
    return out


# 3. Ngữ cảnh tĩnh (process_course.py, process_teacher.py, process_school.py) ---------
def course_context(resources: pd.DataFrame) -> pd.DataFrame:
    r = resources.dropna(subset=["course_id", "resource_id"])
    r = r.drop_duplicates(["course_id", "resource_id"])
    rid = r["resource_id"].astype(str)

    def n_res(mask):
        return r[mask].groupby("course_id")["resource_id"].nunique()

    out = pd.DataFrame({
        "resource_count": n_res(rid.notna()),
        "video_counts": n_res(rid.str.startswith("V_")),
        "ex_counts": n_res(rid.str.startswith("Ex_")),
    }).fillna({"video_counts": 0, "ex_counts": 0}).astype(float)
    return out.rename_axis("course_id").reset_index()


def _profiles(people: pd.DataFrame, key: str,
              text_cols: tuple[str, ...]) -> pd.DataFrame:
    p = people.dropna(subset=[key]).drop_duplicates(key).copy()
    for col in text_cols:
        s = p[col].astype("string")
        has = (s.notna() & (s.str.strip(" ").str.len() > 0)).fillna(False)
        p[f"has_{col}"] = has.astype(float)
        p[f"{col}_length"] = np.where(has, s.str.len().fillna(0), 0).astype(float)
    return p


def teacher_context(course_teacher: pd.DataFrame, teachers: pd.DataFrame) -> pd.DataFrame:
    t = _profiles(teachers, "teacher_id", ("about",))
    t["org_name"] = t["org_name"].astype("string").str.strip(" ")
    ct = course_teacher.dropna().drop_duplicates(["course_id", "teacher_id"])
    ct = ct.merge(t, on="teacher_id")
    return ct.groupby("course_id").agg(
        teacher_count=("teacher_id", "nunique"),
        teacher_bio_coverage=("has_about", "mean"),
        teacher_avg_bio_length_chars=("about_length", "mean"),
        teacher_org_count=("org_name", "nunique"),
    ).astype(float).reset_index()


def school_context(course_school: pd.DataFrame, schools: pd.DataFrame) -> pd.DataFrame:
    s = _profiles(schools, "school_id", ("about", "motto"))
    cs = course_school.dropna().drop_duplicates(["course_id", "school_id"])
    cs = cs.merge(s, on="school_id")
    return cs.groupby("course_id").agg(
        school_count=("school_id", "nunique"),
        school_bio_coverage=("has_about", "mean"),
        school_avg_bio_length_chars=("about_length", "mean"),
        school_motto_coverage=("has_motto", "mean"),
        school_avg_motto_length_chars=("motto_length", "mean"),
    ).astype(float).reset_index()


# 4. Ghép bảng (merge_features) -------------------------------------------------------
def build_cd3_phase_features(spine, video, problem, comment, enrollments, resources,
                             catalog, course_teacher, teachers, course_school, schools,
                             use_sentiment=True):
    """Một dòng mỗi lượt ghi danh × phase. Họ đặc trưng không có sự kiện trước mốc cắt
    để NULL, kèm mask quan sát (cách CD3 đánh dấu vắng mặt cấu trúc; mask không phải
    đầu vào mô hình, QĐ3)."""
    cctx = course_context(resources)
    fams = (video_features(clean_video_events(video), spine, cctx),
            problem_features(clean_problem_events(problem), spine, catalog),
            comment_features(clean_comment_events(comment), spine, cctx, use_sentiment))
    m = spine
    for fam in fams:
        m = m.merge(fam, on=KG, how="left")
    e = enrollments[["enrollment_id", "gender", "year_of_birth", "enroll_time"]].rename(
        columns={"enroll_time": "enroll_time_context"})
    m = m.merge(e, on="enrollment_id", how="left")
    yob = pd.to_numeric(m["year_of_birth"], errors="coerce")
    m["age_at_enroll"] = m["enroll_time_context"].dt.year - yob
    m = m.drop(columns="enroll_time_context")
    ctxs = (cctx, teacher_context(course_teacher, teachers),
            school_context(course_school, schools))
    for ctx in ctxs:
        m = m.merge(ctx, on="course_id", how="left")
    probes = {"video": "segment_count", "problem": "problem_count",
              "comment": "comment_count"}
    for fam, probe in probes.items():
        m[f"{fam}_observed_mask"] = m[probe].notna().astype(int)
    m["phase_fraction"] = m["phase"].map(PHASE_FRACTION)
    return m


def build_in_course_chunks(spine, video, problem, comment, *context, n_chunks=8,
                           use_sentiment=True):
    """Như build_cd3_phase_features nhưng chạy theo lô khóa học khi bảng sự kiện lớn.
    Kết quả trùng với bản chạy một lần vì mọi đặc trưng tính trong một lượt ghi danh
    hoặc một khóa; context là (enrollments, resources, catalog, course_teacher, teachers,
    course_school, schools) đầy đủ."""
    courses = np.array(sorted(spine["course_id"].unique()))
    parts = []
    for chunk in np.array_split(courses, min(n_chunks, len(courses))):
        sp = spine[spine["course_id"].isin(chunk)]
        ids = sp["enrollment_id"].unique()
        ev = [t[t["enrollment_id"].isin(ids)] for t in (video, problem, comment)]
        parts.append(build_cd3_phase_features(sp, *ev, *context,
                                              use_sentiment=use_sentiment))
    return pd.concat(parts, ignore_index=True)


# 5. Quy tắc đầu vào CQ_RAW_EARLY (wide_imputation.py, model/data.py) -----------------
VIDEO_FEATURES = (
    "segment_duration_seconds", "avg_segment_duration", "avg_real_watch_time",
    "segment_count", "avg_playback_speed", "avg_day_of_week", "unique_video_count",
    "video_active_day_count", "first_video_day_from_enroll", "last_video_day_from_enroll",
    "video_day_from_enroll_std", "min_segment_duration", "max_segment_duration",
    "segment_duration_std", "var_playback_speed", "fast_forward_ratio",
    "avg_segment_count_per_video", "video_repeat_segment_ratio",
    "video_events_per_active_day", "most_common_hour", "most_common_day",
    "session_duration_std", "course_video_count", "video_catalog_coverage")
PROBLEM_FEATURES = (
    "problem_count", "exercise_count", "problem_context_count", "score_sum", "score_mean",
    "score_std", "score_fraction_mean", "score_fraction_std", "correct_problem_count",
    "incorrect_problem_count", "correct_ratio", "correct_ratio_std", "attempts_sum",
    "attempts_mean", "attempts_std", "first_submit_day_from_enroll",
    "last_submit_day_from_enroll", "avg_submit_day_from_enroll",
    "submit_day_from_enroll_std", "course_problem_count", "course_problem_full_score_sum",
    "problem_catalog_coverage", "problem_score_catalog_coverage")
COMMENT_FEATURES = (
    "comment_count", "unique_commented_resource_count", "avg_comment_length_chars",
    "min_comment_length_chars", "max_comment_length_chars", "comment_length_std_chars",
    "avg_positive_score", "avg_negative_score", "avg_neutral_score", "positive_score_std",
    "negative_score_std", "neutral_score_std", "positive_comment_count",
    "neutral_comment_count", "negative_comment_count", "positive_comment_ratio",
    "negative_comment_ratio", "first_comment_day_from_enroll",
    "last_comment_day_from_enroll", "avg_comment_day_from_enroll",
    "comment_day_from_enroll_std", "course_resource_count",
    "commented_resource_catalog_coverage")
STATIC_FEATURES = (
    "gender", "year_of_birth", "age_at_enroll", "resource_count", "video_counts",
    "ex_counts", "teacher_count", "teacher_bio_coverage", "teacher_avg_bio_length_chars",
    "teacher_org_count", "school_count", "school_bio_coverage",
    "school_avg_bio_length_chars", "school_motto_coverage",
    "school_avg_motto_length_chars")
# Allowlist: chỉ các cột do build_cd3_phase_features sinh ra; nhãn, mã định danh, split
# và mask không bao giờ vào mô hình, dù khung dữ liệu có chứa chúng.
CANDIDATES = VIDEO_FEATURES + PROBLEM_FEATURES + COMMENT_FEATURES + STATIC_FEATURES
CQ_LABEL_PROXY_RE = re.compile(
    r"(?:^cq_label_|^(?:coelo|afelo|acelo|triad_distance)_|"
    r"^cq_(?:distance|proximity|score_g)|"
    r"(?:catalog_)?coverage$|_coverage$|score_fraction|correct_ratio|watch_ratio|"
    r"video_ratio|problem_ratio|scaled_watch|watch_count_scaled|scaled_attempts)", re.I)
CD3_EXCLUDED = {"year_of_birth", "age_at_enroll", "most_common_day"}   # BASE_EXCLUDED
# Phần của LO_LABEL_PROXY_COLUMNS (CD3 loại cho cả CQ và LO) khớp được tên cột trong
# bảng rộng của CQ; thông tin của video_counts vẫn còn trong course_video_count.
CD3_SHARED_PROXY = {"video_counts"}
CATEGORICAL = ("gender",)
CHANNELS = ("phase_fraction",)


def add_hour_cyclical(frame: pd.DataFrame) -> pd.DataFrame:
    """most_common_hour → sin/cos chu kỳ 24 giờ, như wide_imputation.py."""
    out = frame.copy()
    hour = pd.to_numeric(out.pop("most_common_hour"), errors="coerce")
    out["most_common_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["most_common_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def _wide_name(c: str) -> str:
    """Tên cột trong bảng rộng của CD3 (materialize_cumulative_phase_features.py): cột
    theo phase mang hậu tố _P1.._P4, cột tĩnh giữ nguyên tên. Quy tắc loại của CD3 áp
    lên tên này, nên mẫu ...coverage$ chỉ loại cột độ phủ tĩnh; độ phủ danh mục tính
    đến mốc cắt (ví dụ problem_catalog_coverage_P1) vẫn là đặc trưng, như QA23 và
    CQ/registries.py của CD3 ghi rõ."""
    return c if c in STATIC_FEATURES else f"{c}_P1"


def raw_early_columns(frame: pd.DataFrame) -> dict:
    """Cột của learner view (XGBoost, LR) và selector view theo CQ_RAW_EARLY.
    selector = cột số đã qua V1; learner = selector + gender (mã danh mục) +
    phase_fraction (kênh thời gian, giữ nguyên giá trị như model/data.py)."""
    names = []
    for c in CANDIDATES:
        names += ["most_common_hour_sin", "most_common_hour_cos"] \
            if c == "most_common_hour" else [c]
    keep = [c for c in names if c in frame and c not in CD3_EXCLUDED
            and c not in CD3_SHARED_PROXY
            and not CQ_LABEL_PROXY_RE.search(_wide_name(c))]
    selector = [c for c in keep if c not in CATEGORICAL]
    learner = selector + [c for c in CATEGORICAL + CHANNELS if c in frame]
    return {"learner": learner, "selector": selector}


def _category_strings(s: pd.Series) -> pd.Series:
    """Chuỗi hóa giá trị danh mục; số nguyên lưu dạng float (do có NULL) được đưa về
    dạng nguyên để train (float) và dev (int) cho cùng chuỗi. Khác CD3 có chủ ý: CD3
    chuỗi hóa trực tiếp nên “1.0” và “1” là hai giá trị khác nhau."""
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s)
        if (v.dropna() % 1 == 0).all():
            return v.astype("Int64").astype("string").fillna("<MISSING>")
    return s.astype("string").fillna("<MISSING>")


def fit_category_maps(train: pd.DataFrame, columns=CATEGORICAL) -> dict:
    """_fit_categories của CD3: mã 1..k theo thứ tự xuất hiện trong train, NULL là một
    hạng riêng."""
    maps = {}
    for c in columns:
        values = _category_strings(train[c])
        maps[c] = {v: i for i, v in enumerate(pd.unique(values), start=1)}
    return maps


def encode_categories(frame: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """_encode_categories của CD3: giá trị chưa gặp ở train nhận mã 0."""
    out = frame.copy()
    for c, mapping in maps.items():
        out[c] = _category_strings(out[c]).map(mapping).fillna(0).astype("int32")
    return out


# 6. Tiền xử lý V1 (V1-Median của CD3) và hai view đầu vào ---------------------------
COUNT_RE = re.compile(r"(count|counts|sum|seconds|duration|length|attempts)", re.I)


class TempoV1Preprocessor:
    """V1-Median của CD3, viết gọn: log1p cho cột đếm (giá trị không âm), rồi chuẩn hóa
    robust (trừ median, chia IQR; IQR = 0 thì chia 1, như RobustScaler) với thống kê
    lấy từ giá trị quan sát của Pool B train. Cột theo phase dùng mọi dòng P1–P4
    (TRAIN_POOLED_P1_P4); cột tĩnh dùng một dòng mỗi lượt ghi danh, như bảng rộng của
    CD3. Ô NULL (vắng mặt cấu trúc hoặc thiếu quan sát) nhận 0 sau chuẩn hóa, đúng bằng
    median train. Không tạo chỉ báo thiếu dữ liệu (QĐ3)."""

    def fit(self, train: pd.DataFrame, columns: list[str],
            key: str = "enrollment_id") -> "TempoV1Preprocessor":
        self.columns = list(columns)
        self.log_columns = [c for c in self.columns if COUNT_RE.search(c)]
        x = self._log(train[self.columns])
        once = ~train[key].duplicated() if key in train else pd.Series(True, train.index)
        self.center, self.scale = {}, {}
        for c in self.columns:
            v = x.loc[once, c] if c in STATIC_FEATURES else x[c]
            v = v.dropna().to_numpy()
            if len(v):
                q25, q50, q75 = np.percentile(v, [25, 50, 75])
                iqr = q75 - q25                    # _handle_zeros_in_scale của sklearn
                self.center[c] = q50
                self.scale[c] = 1.0 if iqr < 10 * np.finfo(float).eps else iqr
        return self

    def _log(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.astype(float).copy()
        for c in self.log_columns:
            ok = x[c] >= 0
            x.loc[ok, c] = np.log1p(x.loc[ok, c])
        return x

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = self._log(frame[self.columns])
        out = pd.DataFrame(0.0, index=frame.index, columns=self.columns)
        for c in self.center:                    # cột không có giá trị train: giữ 0
            out[c] = (x[c] - self.center[c]) / self.scale[c]
        return out.fillna(0.0)


class CQRawEarlyViews:
    """Fit một lần trên Pool B train (P1–P4 gộp), rồi biến đổi mọi bảng: Pool B, dev
    P1–P4, snapshot FULL_A của FullTrain-A. XGBoost, LR và selector dùng cùng ma trận
    V1; LR one-hot thêm cột gender theo đặc tả của nhóm. drop: các cột QA23 gắn cờ,
    tạm loại khỏi cả hai view cho đến khi giảng viên duyệt."""

    def fit(self, train: pd.DataFrame, drop=()) -> "CQRawEarlyViews":
        train = add_hour_cyclical(train)
        cols = raw_early_columns(train)
        self.cols = {k: [c for c in v if c not in set(drop)] for k, v in cols.items()}
        self.v1 = TempoV1Preprocessor().fit(train, self.cols["selector"])
        self.maps = fit_category_maps(train, [c for c in CATEGORICAL if c in train])
        return self

    def transform(self, frame: pd.DataFrame) -> dict:
        frame = add_hour_cyclical(frame)
        selector = self.v1.transform(frame)
        extra = encode_categories(frame[list(self.maps)], self.maps)
        for c in CHANNELS:
            if c in self.cols["learner"]:
                extra[c] = frame[c].astype(float)
        learner = pd.concat([selector, extra], axis=1)[self.cols["learner"]]
        return {"learner": learner, "selector": selector}


# 7. Audit QA23 (audit_cq_label_proxy.py) ---------------------------------------------
REVIEWED_BEHAVIOURAL_BASES = {
    "problem_count", "exercise_count", "problem_context_count", "score_sum",
    "correct_problem_count", "attempts_sum", "problem_catalog_coverage",
    "problem_score_catalog_coverage"}
REVIEW_NOTE = "reviewed_prefix_observed_behaviour__not_label_derived"


def _ntile(x: pd.Series, k: int) -> np.ndarray:
    """F.ntile(k) trên thứ tự tăng của x: k nhóm gần bằng nhau, nhóm đầu nhiều hơn."""
    n = len(x)
    pos = np.empty(n, dtype=int)
    pos[np.argsort(x.to_numpy(), kind="stable")] = np.arange(n)
    q, r = divmod(n, k)
    big = r * (q + 1)
    return np.where(pos < big, pos // (q + 1), r + (pos - big) // max(q, 1))


def normalised_mutual_information(x: pd.Series, y: pd.Series,
                                  bins: int = 20) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    pxy = pd.crosstab(_ntile(x, bins), _ntile(y, bins)).to_numpy() / n
    px, py = pxy.sum(1), pxy.sum(0)
    nz = pxy > 0
    mi = (pxy[nz] * np.log(pxy[nz] / np.outer(px, py)[nz])).sum()
    hx = -(px[px > 0] * np.log(px[px > 0])).sum()
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    return float(mi / np.sqrt(hx * hy)) if hx > 0 and hy > 0 else 0.0


def qa23_audit(train: pd.DataFrame, label_score: str, columns: list[str],
               rho_threshold: float = 0.5) -> pd.DataFrame:
    """Spearman (rank min, như percent_rank) và NMI giữa từng đặc trưng và nhãn liên tục,
    trên các dòng train của từng phase (CD3 chạy QA23 cho P1–P4, mỗi lần duyệt mọi cột
    tiền tố đã quan sát). Như CD3: needs_review = |rho| >= ngưỡng; tám cột hành vi CD3
    đã duyệt có proxy_review_note. hold_out = cần duyệt nhưng chưa có ghi chú; theo QĐ28
    các cột này tạm loại khỏi hai view cho đến khi giảng viên duyệt. Dùng tên cột gốc
    (trước V1)."""
    rows = []
    for phase, part in train.groupby("phase", sort=True):
        for c in columns:
            pair = part[[c, label_score]].apply(pd.to_numeric, errors="coerce").dropna()
            rho = np.nan
            if len(pair) > 1 and pair[c].nunique() > 1:      # cột hằng: không tính rho
                r = pair.rank(method="min")
                rho = r[c].corr(r[label_score])
            rows.append({"phase": phase, "column": c, "n_obs": len(pair),
                         "spearman_rho": rho,
                         "nmi": normalised_mutual_information(pair[c], pair[label_score]),
                         "proxy_review_note": REVIEW_NOTE
                         if c in REVIEWED_BEHAVIOURAL_BASES else None})
    out = pd.DataFrame(rows)
    out["needs_review"] = out["spearman_rho"].abs() >= rho_threshold
    out["hold_out"] = out["needs_review"] & out["proxy_review_note"].isna()
    out = out.sort_values(["phase", "spearman_rho"], key=lambda s: -s.abs()
                          if s.name == "spearman_rho" else s)
    return out.reset_index(drop=True)


def held_out_columns(audit: pd.DataFrame) -> list[str]:
    """Cột hold_out ở bất kỳ phase nào; truyền vào CQRawEarlyViews(drop=...)."""
    return sorted(audit.loc[audit["hold_out"], "column"].unique())
