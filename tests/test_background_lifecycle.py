import tempfile
from pathlib import Path

from tamfis_code import background, state


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
