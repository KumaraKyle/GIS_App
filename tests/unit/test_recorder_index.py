"""Session and label index.

Mostly straightforward storage, with two behaviours worth pinning down: labels
are rejected rather than deleted, and coverage is reported per (class, position,
day) because that triple is what determines whether a trained model will
generalise past the afternoon it was recorded.
"""

from __future__ import annotations

import pytest

from csisense.recorder.index import SessionIndex, SessionRow


def make_session(idx: SessionIndex, session_id: str, *, day: str, position: str = "a") -> None:
    idx.add_session(
        SessionRow(
            session_id=session_id,
            path=f"/data/{session_id}.parquet",
            source_kind="esp",
            n_sub=64,
            rate_hz=100.0,
            started_ns=0,
            ended_ns=10_000_000_000,
            n_frames=1000,
            duration_s=10.0,
            day=day,
            position=position,
        )
    )


@pytest.fixture
def index(tmp_path) -> SessionIndex:
    return SessionIndex(tmp_path / "index.db")


def test_session_roundtrip(index):
    make_session(index, "s1", day="2026-08-08")
    row = index.session("s1")
    assert row is not None
    assert row.n_sub == 64
    assert row.day == "2026-08-08"


def test_sessions_filtered_by_day(index):
    make_session(index, "s1", day="2026-08-08")
    make_session(index, "s2", day="2026-08-09")
    assert [s.session_id for s in index.sessions(day="2026-08-09")] == ["s2"]


def test_labels_roundtrip(index):
    make_session(index, "s1", day="d1")
    index.add_label("s1", t_start_ns=0, t_end_ns=2_000_000_000, cls="sit")

    labels = index.labels(session_id="s1")
    assert len(labels) == 1
    assert labels[0].cls == "sit"
    assert labels[0].duration_s == pytest.approx(2.0)


def test_rejected_labels_are_hidden_but_kept(index):
    """A fumbled repetition is evidence the protocol is too fast. Deleting it
    throws that evidence away along with the sample."""
    make_session(index, "s1", day="d1")
    label_id = index.add_label("s1", t_start_ns=0, t_end_ns=1_000_000_000, cls="sit")
    index.reject_label(label_id)

    assert index.labels(session_id="s1") == []
    assert len(index.labels(session_id="s1", include_rejected=True)) == 1
    assert index.rejection_rate() == 1.0


def test_backwards_label_is_rejected(index):
    make_session(index, "s1", day="d1")
    with pytest.raises(ValueError, match="before it starts"):
        index.add_label("s1", t_start_ns=5, t_end_ns=5, cls="sit")


def test_class_counts_exclude_rejected(index):
    make_session(index, "s1", day="d1")
    for _ in range(3):
        index.add_label("s1", t_start_ns=0, t_end_ns=1_000, cls="sit")
    bad = index.add_label("s1", t_start_ns=0, t_end_ns=1_000, cls="sit")
    index.reject_label(bad)

    assert index.class_counts() == {"sit": 3}


def test_coverage_is_keyed_by_class_position_and_day(index):
    """A model trained on one position on one day scores beautifully under
    leave-one-session-out and then fails on Tuesday, so coverage has to be
    tracked across all three axes rather than by count alone."""
    make_session(index, "s1", day="mon", position="door")
    make_session(index, "s2", day="tue", position="couch")
    index.add_label("s1", t_start_ns=0, t_end_ns=1_000, cls="sit")
    index.add_label("s1", t_start_ns=2_000, t_end_ns=3_000, cls="sit")
    index.add_label("s2", t_start_ns=0, t_end_ns=1_000, cls="sit")

    coverage = index.coverage()
    assert coverage[("sit", "door", "mon")] == 2
    assert coverage[("sit", "couch", "tue")] == 1


def test_tiers_are_counted_separately(index):
    make_session(index, "s1", day="d1")
    index.add_label("s1", t_start_ns=0, t_end_ns=1_000, cls="sit", tier="coarse")
    index.add_label("s1", t_start_ns=2_000, t_end_ns=3_000, cls="swipe", tier="fine")

    assert index.class_counts(tier="coarse") == {"sit": 1}
    assert index.class_counts(tier="fine") == {"swipe": 1}


def test_deleting_a_session_removes_its_labels(index):
    make_session(index, "s1", day="d1")
    index.add_label("s1", t_start_ns=0, t_end_ns=1_000, cls="sit")
    index.delete_session("s1")

    assert index.session("s1") is None
    assert index.labels(include_rejected=True) == []


def test_index_survives_reopening(tmp_path):
    path = tmp_path / "index.db"
    first = SessionIndex(path)
    make_session(first, "s1", day="d1")

    assert SessionIndex(path).session("s1") is not None
