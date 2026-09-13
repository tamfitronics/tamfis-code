import tempfile
from pathlib import Path
from unittest.mock import patch

from tamfis_code import background, state


def test_background_task_preserves_strict_max_turns_in_child_argv():
    original_jobs = background.JOBS_DIR
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        background.JOBS_DIR = root / "jobs"
        try:
            fake_process = type("Process", (), {"pid": 12345})()
            with patch("tamfis_code.background.subprocess.Popen", return_value=fake_process) as popen:
                background.spawn_background_task(
                    session_id=7,
                    workspace_root=root,
                    mode="audit",
                    objective="bounded audit",
                    model="auto",
                    provider=None,
                    approval_policy="auto",
                    max_turns=6,
                )

            argv = popen.call_args.args[0]
            assert argv[argv.index("--max-turns") + 1] == "6"
            assert argv.index("--max-turns") > argv.index("ask")
        finally:
            background.JOBS_DIR = original_jobs


def test_background_completion_is_reinjected_once_into_originating_session():
    original_jobs = background.JOBS_DIR
    original_state = (state.CONFIG_DIR, state.STATE_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        background.JOBS_DIR = root / "jobs"
        state.CONFIG_DIR = root / "state"
        state.STATE_PATH = root / "state" / "state.json"
        try:
            log = root / "job.log"
            log.write_text("analysis complete\nFINAL RESULT\n")
            job = background.BackgroundJob(
                id="bg-test", pid=999999, session_id=77, workspace_root=str(root),
                mode="coding", objective_preview="audit pipeline", log_path=str(log),
                prompt_path=str(root / "prompt"), started_at=1.0, goal=True,
            )
            background._write_job(job)

            background.update_job_status("bg-test", "completed", exit_code=0)
            queued = state.get_session_state(77).queued_user_instructions
            assert len(queued) == 1
            assert queued[0]["classification"] == "follow_up"
            assert "FINAL RESULT" in queued[0]["text"]
            assert "Background goal bg-test" in queued[0]["text"]

            background.update_job_status("bg-test", "completed", exit_code=0)
            assert len(state.get_session_state(77).queued_user_instructions) == 1
            assert background.read_job("bg-test")["notification_delivered"] is True
        finally:
            background.JOBS_DIR = original_jobs
            state.CONFIG_DIR, state.STATE_PATH = original_state


def test_a_configured_notification_hook_fires_on_background_job_completion():
    """Claude-Code-parity addition: update_job_status runs in a detached
    background child process with no already-running asyncio event loop,
    so it can safely asyncio.run the hook itself -- proven here with a
    real on-disk .tamfis/hooks.toml and a real subprocess, not a mock of
    run_notification_hooks."""
    original_jobs = background.JOBS_DIR
    original_state = (state.CONFIG_DIR, state.STATE_PATH)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        background.JOBS_DIR = root / "jobs"
        state.CONFIG_DIR = root / "state"
        state.STATE_PATH = root / "state" / "state.json"
        try:
            marker = root / "notified.txt"
            hooks_dir = root / ".tamfis"
            hooks_dir.mkdir()
            (hooks_dir / "hooks.toml").write_text(
                f'[[notification]]\ncommand = "cat > {marker}"\n'
            )
            log = root / "job.log"
            log.write_text("done\n")
            job = background.BackgroundJob(
                id="bg-notify", pid=999998, session_id=88, workspace_root=str(root),
                mode="coding", objective_preview="run a task", log_path=str(log),
                prompt_path=str(root / "prompt"), started_at=1.0,
            )
            background._write_job(job)

            background.update_job_status("bg-notify", "completed", exit_code=0)

            assert marker.is_file(), "notification hook never ran"
            import json
            payload = json.loads(marker.read_text())
            assert payload["event"] == "notification"
            assert payload["session_id"] == 88
            assert "bg-notify" in payload["message"]
        finally:
            background.JOBS_DIR = original_jobs
            state.CONFIG_DIR, state.STATE_PATH = original_state
