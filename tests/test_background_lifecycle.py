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
