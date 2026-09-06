from pathlib import Path

from tamfis_code.runtime.memory import MemoryError, MemoryRecord, MemoryStore, MemoryType
import pytest


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(root=tmp_path / "memory")


def test_save_and_load_round_trip(tmp_path: Path):
    store = _store(tmp_path)
    saved = store.save(MemoryRecord(
        name="Prefers Terse Replies",
        type=MemoryType.FEEDBACK,
        description="User wants short responses",
        content="Do not summarize at the end of every response.",
    ))
    assert saved.name == "prefers-terse-replies"
    loaded = store.load("Prefers Terse Replies")
    assert loaded is not None
    assert loaded.content == "Do not summarize at the end of every response."
    assert loaded.type == MemoryType.FEEDBACK


def test_delete_removes_record_and_index_entry(tmp_path: Path):
    store = _store(tmp_path)
    store.save(MemoryRecord(name="x", type=MemoryType.PROJECT, description="d", content="c"))
    assert store.delete("x") is True
    assert store.load("x") is None
    assert store.delete("x") is False


def test_list_reconciles_index_after_external_deletion(tmp_path: Path):
    store = _store(tmp_path)
    store.save(MemoryRecord(name="a", type=MemoryType.USER, description="d", content="c"))
    store.save(MemoryRecord(name="b", type=MemoryType.USER, description="d", content="c"))
    (store.root / "b.json").unlink()
    records = store.list()
    assert [r.name for r in records] == ["a"]


def test_search_ranks_by_keyword_overlap(tmp_path: Path):
    store = _store(tmp_path)
    store.save(MemoryRecord(
        name="deploy-notes", type=MemoryType.PROJECT,
        description="deploy pipeline notes", content="staging deploy runs nightly",
    ))
    store.save(MemoryRecord(
        name="unrelated", type=MemoryType.REFERENCE,
        description="dashboard link", content="grafana board",
    ))
    results = store.search("how does the deploy pipeline work")
    assert results
    assert results[0].name == "deploy-notes"


def test_slugify_rejects_empty_name(tmp_path: Path):
    store = _store(tmp_path)
    with pytest.raises(MemoryError):
        store.save(MemoryRecord(name="   ", type=MemoryType.USER, description="d", content="c"))


def test_memory_type_matches_assistant_taxonomy():
    assert {t.value for t in MemoryType} == {"user", "feedback", "project", "reference"}


def test_oversized_content_is_truncated_at_save_time(tmp_path: Path):
    store = _store(tmp_path)
    huge = "x" * (store.MAX_CONTENT_CHARS * 2)
    saved = store.save(MemoryRecord(name="huge", type=MemoryType.PROJECT, description="d", content=huge))
    assert len(saved.content) <= store.MAX_CONTENT_CHARS
    assert saved.content.endswith("[truncated]")
    loaded = store.load("huge")
    assert loaded is not None
    assert len(loaded.content) <= store.MAX_CONTENT_CHARS


def test_record_count_is_capped_evicting_oldest_first(tmp_path: Path):
    store = _store(tmp_path)
    store.MAX_RECORDS = 3
    for i in range(5):
        store.save(MemoryRecord(name=f"record-{i}", type=MemoryType.PROJECT, description="d", content="c"))
    remaining = {r.name for r in store.list()}
    assert len(remaining) == 3
    # The two oldest (record-0, record-1) should have been evicted; the
    # most recently saved three survive.
    assert remaining == {"record-2", "record-3", "record-4"}


def test_re_saving_an_existing_record_refreshes_its_recency_for_the_cap(tmp_path: Path):
    store = _store(tmp_path)
    store.MAX_RECORDS = 3
    for i in range(3):
        store.save(MemoryRecord(name=f"record-{i}", type=MemoryType.PROJECT, description="d", content="c"))
    # Touch record-0 again so it's no longer the oldest by updated_at.
    store.save(MemoryRecord(name="record-0", type=MemoryType.PROJECT, description="d", content="updated"))
    store.save(MemoryRecord(name="record-3", type=MemoryType.PROJECT, description="d", content="c"))
    remaining = {r.name for r in store.list()}
    assert len(remaining) == 3
    assert "record-0" in remaining  # survived because it was refreshed
    assert "record-1" not in remaining  # evicted as the true oldest
