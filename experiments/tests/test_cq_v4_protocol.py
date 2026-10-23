"""Contract tests cho family/split CQ V4 của Review 22/10/2026."""

from __future__ import annotations

from skq_exp.data.cq_v4_protocol import (
    build_course_families,
    concentration,
    split_course_families,
    temporal_overlap,
)


def test_family_connected_components_are_transitive_and_deterministic() -> None:
    source = {
        1: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        2: {1, 2, 3, 4, 5, 6, 7, 8, 9},
        3: {1, 2, 3},
        4: set(),
    }
    mapping = build_course_families(source)
    assert mapping[1] == mapping[2]
    assert mapping[3] != mapping[1]
    assert mapping[4] != mapping[1]
    assert mapping == build_course_families(dict(reversed(list(source.items()))))


def _families():
    return [
        {
            "course_family_id": f"f{index:02d}",
            "family_start": f"2026-{1 + index // 9:02d}-{1 + index % 9:02d}",
            "family_end": f"2026-{2 + index // 9:02d}-{1 + index % 9:02d}",
            "n_enroll_labeled": 10 + index * 3,
            "has_E": index % 3 == 0,
        }
        for index in range(20)
    ]


def test_family_split_is_deterministic_exact_and_latest_test() -> None:
    rows = _families()
    first = split_course_families(rows)
    assert first == split_course_families(rows)
    assert list(first.values()).count("test") == 4
    assert list(first.values()).count("dev") == 2
    assert {key for key, value in first.items() if value == "test"} == {
        "f16", "f17", "f18", "f19",
    }


def test_temporal_overlap_and_concentration_are_auditable() -> None:
    rows = _families()
    split = split_course_families(rows)
    overlap = temporal_overlap(rows, split)
    assert overlap["train_family_count"] == 14
    assert overlap["overlap_family_count"] <= 14
    assert overlap["requires_nolfo"] == (overlap["overlap_ratio"] > 0.05)
    summary = concentration(["f1", "f1", "f1", "f2"])
    assert summary["largest_share"] == 0.75
    assert summary["effective_families"] < 2
