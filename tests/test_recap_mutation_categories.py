from types import SimpleNamespace

from tamfis_code.render import StreamRenderer


def test_recap_shows_added_updated_and_removed_files(monkeypatch, tmp_path):
    from tamfis_code.runtime import ledger as ledger_module

    record = SimpleNamespace(
        plan_steps=[],
        edits=[
            SimpleNamespace(file="new.py", operation="create", applied=True, description="create (+4/-0)"),
            SimpleNamespace(file="config.py", operation="update", applied=True, description="update (+2/-1)"),
            SimpleNamespace(file="old.py", operation="delete", applied=True, description="delete (+0/-8)"),
        ],
        tests=[], status="completed", next_action="",
    )
    monkeypatch.setattr(ledger_module, "load_ledger", lambda _session_id: record)
    from io import StringIO
    from rich.console import Console

    console = Console(file=StringIO(), no_color=True)
    renderer = StreamRenderer(console)
    renderer.print_recap(1)
    output = console.file.getvalue()
    assert "Added: new.py (+4/-0)" in output
    assert "Updated: config.py (+2/-1)" in output
    assert "Removed: old.py (+0/-8)" in output
    assert "Changed:" not in output
