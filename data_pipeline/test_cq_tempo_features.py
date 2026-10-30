import numpy as np
import pandas as pd
import pytest

from cq_tempo_features import (CANDIDATES, COMMENT_FEATURES, PROBLEM_FEATURES,
                               STATIC_FEATURES, VIDEO_FEATURES, CQRawEarlyViews,
                               TempoV1Preprocessor, add_hour_cyclical,
                               build_cd3_phase_features, build_in_course_chunks,
                               encode_categories,
                               fit_category_maps, held_out_columns, qa23_audit,
                               raw_early_columns, REVIEW_NOTE)

T = pd.Timestamp
VIDEO_COLS = ["enrollment_id", "resource_id", "event_time", "start_point", "end_point",
              "speed"]
PROBLEM_COLS = ["enrollment_id", "problem_id", "exercise_id", "context_id", "score",
                "full_score", "is_correct", "attempts", "submit_time"]
COMMENT_COLS = ["enrollment_id", "course_id", "comment_id", "resource_id", "event_time",
                "text", "positive_score", "negative_score", "neutral_score",
                "sentiment_label"]


def _data():
    spine = pd.DataFrame({
        "enrollment_id": ["e1", "e1", "e2", "e2"],
        "user_id": ["U_1", "U_1", "U_2", "U_2"],
        "course_id": ["C_1"] * 4, "phase": ["P1", "P4", "P1", "P4"],
        "enroll_time": [T("2020-01-01"), T("2020-01-01"), T("2020-01-05"),
                        T("2020-01-05")],
        "cutoff_time": [T("2020-01-11"), T("2020-02-06"), T("2020-01-06"),
                        T("2020-01-20")]})
    video = pd.DataFrame([
        ("e1", "V_a", "2020-01-01 10:00", 0, 100, 1.0),
        ("e1", "V_a", "2020-01-01 10:20", 100, 160, 2.0),
        ("e1", "V_b", "2020-01-03 09:00", 0, 50, 1.5),
        ("e1", "V_c", "2020-01-20 12:00", 0, 30, 1.0),
        ("e1", "V_d", "2020-01-02 08:00", 50, 10, 1.0),   # end < start: loại khi làm sạch
        ("e1", "V_d", "2020-01-02 08:00", 0, 10, 0.0),    # speed = 0: loại
        ("e1", "V_a", "2019-12-31 08:00", 0, 10, 1.0),    # trước ghi danh: loại khi chọn
        ("e1", "V_a", "2020-02-10 08:00", 0, 10, 1.0)],   # sau mốc cắt P4: loại
        columns=VIDEO_COLS)
    video["event_time"] = pd.to_datetime(video["event_time"])
    problem = pd.DataFrame([
        ("e1", "Pm_1", "Ex_1", "K1", 2.0, 4.0, 1.0, 1.0, "2020-01-02 08:00"),
        ("e1", "Pm_2", "Ex_1", "K1", None, 2.0, 0.0, 3.0, "2020-01-04 08:00"),
        ("e1", "Pm_3", "Ex_2", None, 1.0, 1.0, None, 0.0, "2020-01-05 08:00"),
        ("e1", "Pm_4", "Ex_2", "K2", 1.0, 0.0, 1.0, 1.0, "2020-01-05 09:00"),  # full = 0
        ("e1", "Pm_5", "Ex_2", "K2", 1.0, 1.0, 1.0, None, "2020-01-05 10:00"),  # NULL
        ("e1", "Pm_6", "Ex_2", "K2", 3.0, 3.0, 1.0, 2.0, "2020-01-25 08:00")],
        columns=PROBLEM_COLS)
    problem["submit_time"] = pd.to_datetime(problem["submit_time"])
    comment = pd.DataFrame([
        ("e1", "C_1", "Cm_1", "V_a", "2020-01-02 08:00", "  hello ", .7, .1, .2,
         "positive"),
        ("e1", "C_1", "Cm_2", None, "2020-01-03 08:00", "ok", .2, .5, .3, "negative"),
        ("e1", "C_1", "Cm_2", None, "2020-01-03 08:00", "ok", .2, .5, .3, "negative"),
        ("e1", "C_1", "Cm_3", "V_b", "2020-01-30 08:00", "abc", .1, .1, .8, "neutral")],
        columns=COMMENT_COLS)                             # dòng 2 và 3 trùng comment_id
    comment["event_time"] = pd.to_datetime(comment["event_time"])
    enrollments = pd.DataFrame({
        "enrollment_id": ["e1", "e2"], "gender": [1, 2], "year_of_birth": ["1990", "abc"],
        "enroll_time": [T("2020-01-01 07:30"), T("2020-01-05 09:00")]})
    resources = pd.DataFrame({
        "course_id": ["C_1"] * 7,
        "resource_id": ["V_a", "V_b", "V_c", "V_x", "Ex_1", "Ex_2", "D_1"]})
    catalog = pd.DataFrame({
        "course_id": ["C_1"] * 5,
        "problem_id": ["Pm_1", "Pm_2", "Pm_3", "Pm_6", "Pm_7"],
        "full_score": [4.0, 2.0, 1.0, 3.0, 5.0]})
    course_teacher = pd.DataFrame({"course_id": ["C_1"] * 3,
                                   "teacher_id": ["T_1", "T_2", "T_3"]})
    teachers = pd.DataFrame({"teacher_id": ["T_1", "T_2"], "org_name": ["UIT ", "UIT"],
                             "about": ["Prof.", "   "]})
    course_school = pd.DataFrame({"course_id": ["C_1"], "school_id": ["S_1"]})
    schools = pd.DataFrame({"school_id": ["S_1"], "about": [None], "motto": ["Learn"]})
    return (spine, video, problem, comment, enrollments, resources, catalog,
            course_teacher, teachers, course_school, schools)


def _build(**kw):
    return build_cd3_phase_features(*_data(), **kw).set_index(["enrollment_id", "phase"])


@pytest.fixture(scope="module")
def table():
    return _build()


def test_video_family_matches_hand_computation(table):
    r = table.loc[("e1", "P1")]
    assert r["segment_count"] == 3 and r["segment_duration_seconds"] == 210
    assert r["avg_real_watch_time"] == pytest.approx((100 + 30 + 50 / 1.5) / 3)
    assert r["avg_day_of_week"] == pytest.approx((4 + 4 + 6) / 3)   # thứ Tư 4, thứ Sáu 6
    assert (r["unique_video_count"], r["video_active_day_count"]) == (2, 2)
    assert r["first_video_day_from_enroll"] == 0
    assert r["last_video_day_from_enroll"] == 2
    assert r["segment_duration_std"] == pytest.approx(np.std([100, 60, 50], ddof=1))
    assert r["var_playback_speed"] == pytest.approx(0.25)
    assert r["fast_forward_ratio"] == pytest.approx(2 / 3)
    assert r["video_repeat_segment_ratio"] == pytest.approx(1 / 3)
    assert r["most_common_hour"] == 10 and r["course_video_count"] == 4
    assert r["video_catalog_coverage"] == pytest.approx(0.5)
    sessions = [1230, 50 / 1.5]              # phiên 10:00–10:20:30; phiên 09:00 (1 segment)
    assert r["session_duration_std"] == pytest.approx(np.std(sessions, ddof=1))
    p4 = table.loc[("e1", "P4")]
    assert p4["segment_count"] == 4 and p4["last_video_day_from_enroll"] == 19


def test_problem_family_and_cleaning(table):
    r = table.loc[("e1", "P1")]
    assert r["problem_count"] == 3 and r["exercise_count"] == 2
    assert r["problem_context_count"] == 1
    assert (r["score_sum"], r["score_mean"]) == (3, 1)
    assert r["score_fraction_mean"] == pytest.approx(0.5)
    assert r["correct_problem_count"] == 1
    assert r["incorrect_problem_count"] == 2                  # is_correct NULL tính là 0
    assert r["attempts_sum"] == 4
    assert r["attempts_std"] == pytest.approx(np.std([1, 3, 0], ddof=1))
    assert r["first_submit_day_from_enroll"] == 1
    assert r["last_submit_day_from_enroll"] == 4
    assert r["problem_catalog_coverage"] == pytest.approx(3 / 5)
    assert r["problem_score_catalog_coverage"] == pytest.approx(3 / 15)
    assert table.loc[("e1", "P4"), "problem_count"] == 4


def test_comment_family_with_and_without_sentiment(table):
    r = table.loc[("e1", "P1")]
    assert (r["comment_count"], r["unique_commented_resource_count"]) == (2, 1)
    assert (r["avg_comment_length_chars"], r["max_comment_length_chars"]) == (3.5, 5)
    assert r["avg_positive_score"] == pytest.approx(0.45)
    assert (r["positive_comment_count"], r["negative_comment_count"]) == (1, 1)
    assert r["commented_resource_catalog_coverage"] == pytest.approx(1 / 7)
    plain = _build(use_sentiment=False)
    assert np.isnan(plain.loc[("e1", "P1"), "avg_positive_score"])
    assert plain.loc[("e1", "P1"), "positive_comment_count"] == 0


def test_absence_is_null_with_mask_and_context_is_joined(table):
    r = table.loc[("e2", "P1")]
    for probe in ("segment_count", "problem_count", "comment_count"):
        assert np.isnan(r[probe])
    for fam in ("video", "problem", "comment"):
        assert r[f"{fam}_observed_mask"] == 0
    assert np.isnan(r["age_at_enroll"])
    assert table.loc[("e1", "P1"), "age_at_enroll"] == 30
    assert r["teacher_count"] == 2 and r["teacher_org_count"] == 1
    assert r["teacher_bio_coverage"] == 0.5
    assert (r["school_motto_coverage"], r["school_avg_motto_length_chars"]) == (1, 5)
    assert (r["resource_count"], r["video_counts"], r["ex_counts"]) == (7, 4, 2)
    assert r["phase_fraction"] == 0.25


def test_course_chunks_give_identical_table():
    spine, video, problem, comment, *context = _data()
    extra = spine.assign(course_id="C_2", enrollment_id=spine["enrollment_id"] + "b")
    spine2 = pd.concat([spine, extra], ignore_index=True)
    video2 = pd.concat([video, video.assign(enrollment_id=video["enrollment_id"] + "b")])
    enr = context[0]
    context[0] = pd.concat([enr, enr.assign(enrollment_id=enr["enrollment_id"] + "b")])
    keys = ["enrollment_id", "phase"]
    one = build_cd3_phase_features(spine2, video2, problem, comment, *context)
    many = build_in_course_chunks(spine2, video2, problem, comment, *context, n_chunks=2)
    one, many = (t.sort_values(keys).reset_index(drop=True) for t in (one, many))
    pd.testing.assert_frame_equal(one, many[one.columns])
    assert many.loc[many.enrollment_id.eq("e1b") & many.phase.eq("P1"),
                    "segment_count"].item() == 3


def test_raw_early_policy(table):
    frame = add_hour_cyclical(table.reset_index())
    frame["q_alpha_0p1"], frame["y_alpha_1"], frame["user_id_code"] = 0.3, "G", 7
    cols = raw_early_columns(frame)
    excluded = {"score_fraction_mean", "score_fraction_std", "correct_ratio",
                "correct_ratio_std", "teacher_bio_coverage", "school_bio_coverage",
                "school_motto_coverage", "age_at_enroll", "year_of_birth",
                "most_common_day", "video_observed_mask", "user_id", "course_id", "phase",
                "video_counts", "q_alpha_0p1", "y_alpha_1", "user_id_code"}
    assert excluded.isdisjoint(cols["learner"])
    kept = {"segment_count", "attempts_sum", "most_common_hour_sin", "gender",
            "phase_fraction", "course_video_count", "ex_counts",
            "video_catalog_coverage", "problem_catalog_coverage",           # độ phủ tính
            "problem_score_catalog_coverage", "commented_resource_catalog_coverage"}
    assert kept <= set(cols["learner"])
    assert len(cols["learner"]) == 76 and len(cols["selector"]) == 74
    assert cols["learner"][-2:] == ["gender", "phase_fraction"]
    assert cols["learner"][:-2] == cols["selector"]


def test_builder_output_matches_allowlist(table):
    produced = set(table.reset_index().columns) - {
        "enrollment_id", "user_id", "course_id", "phase", "enroll_time", "cutoff_time",
        "video_observed_mask", "problem_observed_mask", "comment_observed_mask",
        "phase_fraction"}
    assert produced == set(CANDIDATES)
    assert len(VIDEO_FEATURES) == 24 and len(PROBLEM_FEATURES) == 23
    assert len(COMMENT_FEATURES) == 23 and len(STATIC_FEATURES) == 15


def test_views_share_one_v1_matrix(table):
    frame = table.reset_index()
    views = CQRawEarlyViews().fit(frame)
    out = views.transform(frame)
    learner, selector = out["learner"], out["selector"]
    assert list(learner.columns) == views.cols["learner"]
    assert learner[views.cols["selector"]].equals(selector)
    assert not learner.isna().any().any()
    assert learner["phase_fraction"].tolist() == [0.25, 0.9, 0.25, 0.9]
    assert learner["gender"].tolist() == [1, 1, 2, 2]
    absent = frame.index[frame["segment_count"].isna()]
    assert (selector.loc[absent, "segment_count"] == 0).all()       # NULL → median → 0
    held = CQRawEarlyViews().fit(frame, drop=["avg_positive_score"]).transform(frame)
    assert "avg_positive_score" not in held["learner"] and held["selector"].shape[1] == 73


def test_category_codes_follow_cd3():
    train = pd.DataFrame({"gender": [2, 1, None, 2]})
    maps = fit_category_maps(train)
    assert maps["gender"] == {"2": 1, "1": 2, "<MISSING>": 3}
    dev = encode_categories(pd.DataFrame({"gender": [1, 3, None]}), maps)
    assert dev["gender"].tolist() == [2, 0, 3]                  # 3 chưa gặp ở train → 0
    test = encode_categories(pd.DataFrame({"gender": [1, 2]}), maps)     # kiểu int
    assert test["gender"].tolist() == [2, 1]


def test_v1_preprocessor_uses_train_statistics_only():
    train = pd.DataFrame({"segment_count": [1, 3, np.nan, 7],
                          "avg_playback_speed": [1.0, 1.5, np.nan, 2.0]})
    prep = TempoV1Preprocessor().fit(train, list(train.columns))
    out = prep.transform(train)
    expected = [-1.0, 0.0, 0.0, 1.0]              # log1p rồi robust scale; NULL → 0
    assert out["segment_count"].tolist() == pytest.approx(expected)
    assert out["avg_playback_speed"].tolist() == pytest.approx(expected)
    dev = pd.DataFrame({"segment_count": [15.0], "avg_playback_speed": [3.0]})
    assert prep.transform(dev).iloc[0].tolist() == pytest.approx([2.0, 3.0])


def test_v1_static_columns_use_one_row_per_enrollment():
    from sklearn.preprocessing import RobustScaler
    train = pd.DataFrame({"enrollment_id": ["a"] * 4 + ["b", "c", "d"],
                          "teacher_count": [9.0] * 4 + [1.0, 2.0, 3.0],
                          "segment_duration_std": [9.0] * 4 + [1.0, 2.0, 3.0]})
    cols = ["teacher_count", "segment_duration_std"]
    prep = TempoV1Preprocessor().fit(train, cols)
    logged = np.log1p(train[cols])                              # cả hai là cột đếm
    once = RobustScaler().fit(logged[~train["enrollment_id"].duplicated()][cols[:1]])
    pooled = RobustScaler().fit(logged[cols[1:]])
    assert prep.center["teacher_count"] == pytest.approx(once.center_[0])
    assert prep.scale["teacher_count"] == pytest.approx(once.scale_[0])
    assert prep.center["segment_duration_std"] == pytest.approx(pooled.center_[0])
    assert prep.scale["segment_duration_std"] == pytest.approx(pooled.scale_[0])


def test_edge_cases_from_cd3_semantics():
    spine, video, problem, comment, *context = _data()
    spine2 = pd.concat([spine, spine[spine.enrollment_id.eq("e1")].assign(
        enrollment_id="e1x")], ignore_index=True)                 # lượt ghi danh lặp lại
    comment2 = pd.concat([comment, comment.assign(enrollment_id="e1x")])
    comment2["sentiment_label"] = comment2["sentiment_label"].str.title()  # "Positive"
    problem2 = problem.assign(is_correct=problem["is_correct"].map(
        lambda v: None if pd.isna(v) else str(int(v))))            # is_correct dạng chuỗi
    video2 = video.assign(enroll_time=pd.Timestamp("1999-01-01"))  # cột trùng tên spine
    context[2] = context[2].assign(full_score=np.nan)             # danh mục thiếu điểm
    t = build_cd3_phase_features(spine2, video2, problem2, comment2,
                                 *context).set_index(["enrollment_id", "phase"])
    for e in ("e1", "e1x"):
        assert t.loc[(e, "P1"), "comment_count"] == 2
        assert t.loc[(e, "P1"), "positive_comment_count"] == 1
    assert t.loc[("e1", "P1"), "correct_problem_count"] == 1
    assert t.loc[("e1", "P1"), "segment_count"] == 3
    assert np.isnan(t.loc[("e1", "P1"), "course_problem_full_score_sum"])
    assert np.isnan(t.loc[("e1", "P1"), "problem_score_catalog_coverage"])


def test_qa23_follows_cd3_review_semantics():
    rng = np.random.default_rng(0)
    q = rng.random(500)
    frame = pd.DataFrame({"phase": "P4", "q": q, "proxy": -q + 0.01 * rng.random(500),
                          "noise": rng.random(500), "attempts_sum": q})
    frame = pd.concat([frame, frame.assign(phase="P1", proxy=rng.random(500))])
    audit = qa23_audit(frame, "q", ["proxy", "noise", "attempts_sum"])
    p4 = audit[audit.phase.eq("P4")].set_index("column")
    assert bool(p4.loc["proxy", "needs_review"]) and bool(p4.loc["proxy", "hold_out"])
    assert not bool(p4.loc["noise", "needs_review"])
    assert bool(p4.loc["attempts_sum", "needs_review"])          # như CD3: vẫn cần duyệt
    assert p4.loc["attempts_sum", "proxy_review_note"] == REVIEW_NOTE
    assert not bool(p4.loc["attempts_sum", "hold_out"])
    assert p4.loc["proxy", "nmi"] > p4.loc["noise", "nmi"]
    assert held_out_columns(audit) == ["proxy"]                   # cờ ở P4 là đủ


def test_ntile_bins_match_spark():
    from cq_tempo_features import _ntile
    assert _ntile(pd.Series([5, 1, 4, 2, 3, 6, 7]), 3).tolist() == [1, 0, 1, 0, 0, 2, 2]
    sizes = np.bincount(_ntile(pd.Series(np.arange(45)), 20)).tolist()
    assert sizes == [3] * 5 + [2] * 15
