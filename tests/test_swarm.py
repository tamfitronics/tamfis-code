import asyncio
import unittest
from unittest.mock import MagicMock, patch

from tamfis_code.agents import AgentManager, DelegatedCodingAgent
from tamfis_code.config import APPROVAL_MODES
from tamfis_code.runner import _decision_for_policy
from tamfis_code.swarm import BufferedSubagentRenderer, mutation_policy_allows_swarm, run_swarm


class MutationPolicyAllowsSwarmTests(unittest.TestCase):
    def test_auto_approving_policies_allow_swarm_mutation(self):
        for policy in ("auto", "full-auto", "safe", "workspace", "accept-edits"):
            self.assertTrue(mutation_policy_allows_swarm(policy), policy)

    def test_ask_and_deny_all_policies_do_not_allow_swarm_mutation(self):
        for policy in ("ask", "never", "read-only", "plan-only", "suggest"):
            self.assertFalse(mutation_policy_allows_swarm(policy), policy)

    def test_unknown_policy_defaults_to_disallowed(self):
        self.assertFalse(mutation_policy_allows_swarm("not-a-real-policy"))

    def test_case_and_whitespace_insensitive(self):
        self.assertTrue(mutation_policy_allows_swarm("  Auto  "))

    def test_never_drifts_from_decision_for_policy(self):
        """mutation_policy_allows_swarm deliberately duplicates
        _decision_for_policy's own auto-approving groupings (rather than
        calling it directly, to keep swarm.py decoupled from runner.py's
        approval-prompt machinery) -- this test makes any future edit to
        either list without the other fail immediately, instead of the two
        silently diverging."""
        for policy in APPROVAL_MODES:
            expected = _decision_for_policy(policy, "medium", interactive=False) == "approve_once"
            self.assertEqual(
                mutation_policy_allows_swarm(policy), expected,
                f"mutation_policy_allows_swarm({policy!r}) disagrees with _decision_for_policy",
            )


class BufferedSubagentRendererTests(unittest.TestCase):
    def test_never_constructs_a_live_region(self):
        with patch("rich.live.Live") as fake_live:
            renderers = [BufferedSubagentRenderer(f"t{i}", f"task {i}") for i in range(5)]
            for r in renderers:
                r.handle_event({"type": "model_selected", "model": "some-model", "provider": "openrouter"})
                r.handle_event({"type": "tool_call_requested", "name": "read_file"})
                r.handle_event({"type": "file_mutation", "path": "app.py"})
                r.handle_event({"type": "ai_task_failed", "message": "boom"})
                r.finish()
        fake_live.assert_not_called()

    def test_translates_events_into_on_update_calls(self):
        updates = []
        renderer = BufferedSubagentRenderer("t1", "fix the bug", on_update=lambda tid, fields: updates.append((tid, fields)))

        renderer.handle_event({"type": "model_selected", "model": "gpt-5", "provider": "openrouter"})
        renderer.handle_event({"type": "tool_call_requested", "name": "read_file"})
        renderer.handle_event({"type": "file_mutation", "path": "app.py"})
        renderer.handle_event({"type": "ai_task_failed", "message": "provider timeout"})

        self.assertEqual(len(updates), 4)
        self.assertEqual(updates[0], ("t1", {"phase": "running", "detail": "model: gpt-5"}))
        self.assertEqual(updates[1], ("t1", {"phase": "running", "detail": "calling read_file"}))
        self.assertEqual(updates[2], ("t1", {"phase": "running", "detail": "edited app.py"}))
        self.assertEqual(updates[3], ("t1", {"phase": "failed", "detail": "provider timeout"}))

    def test_unrecognized_event_types_are_a_no_op(self):
        updates = []
        renderer = BufferedSubagentRenderer("t1", "task", on_update=lambda tid, fields: updates.append((tid, fields)))
        renderer.handle_event({"type": "assistant_delta", "content": "hello"})
        renderer.handle_event({"type": "reasoning_delta", "content": "thinking..."})
        self.assertEqual(updates, [])

    def test_no_on_update_callback_is_safe(self):
        renderer = BufferedSubagentRenderer("t1", "task")
        renderer.handle_event({"type": "tool_call_requested", "name": "read_file"})
        renderer.finish()  # must not raise


class DelegatedCodingAgentRendererFactoryTests(unittest.TestCase):
    def test_defaults_to_real_stream_renderer_when_no_factory_given(self):
        fake_console = MagicMock()
        fake_renderer_instance = MagicMock()

        async def fake_run_local_agent_turn(*args, **kwargs):
            from tamfis_code.runner import TaskOutcome
            return TaskOutcome(status="completed", summary="done")

        with patch("tamfis_code.render.StreamRenderer", return_value=fake_renderer_instance) as fake_cls, \
                patch("tamfis_code.runner_local.run_local_agent_turn", new=fake_run_local_agent_turn):
            agent = DelegatedCodingAgent(
                manager=object(), provider=object(), model=None, console=fake_console,
                workspace_root="/tmp", session_id=1,
            )
            from tamfis_code.agents import AgentTask
            result = asyncio.run(agent.execute(AgentTask(id="t1", description="do something")))

        fake_cls.assert_called_once_with(fake_console)
        fake_renderer_instance.finish.assert_called_once()
        self.assertEqual(result["status"], "completed")

    def test_uses_provided_renderer_factory_instead_of_stream_renderer(self):
        fake_renderer_instance = MagicMock()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return fake_renderer_instance

        async def fake_run_local_agent_turn(*args, **kwargs):
            from tamfis_code.runner import TaskOutcome
            return TaskOutcome(status="completed", summary="done")

        with patch("tamfis_code.render.StreamRenderer") as fake_cls, \
                patch("tamfis_code.runner_local.run_local_agent_turn", new=fake_run_local_agent_turn):
            agent = DelegatedCodingAgent(
                manager=object(), provider=object(), model=None, console=MagicMock(),
                workspace_root="/tmp", session_id=1, renderer_factory=factory,
            )
            from tamfis_code.agents import AgentTask
            asyncio.run(agent.execute(AgentTask(id="t1", description="do something")))

        fake_cls.assert_not_called()
        self.assertEqual(len(factory_calls), 1)
        fake_renderer_instance.finish.assert_called_once()


class ExecuteTasksRendererFactoryTests(unittest.TestCase):
    """execute_tasks itself must build one BufferedSubagentRenderer per
    sub-task via the given factory, keyed by that sub-task's own task_id/
    description -- not the same instance reused across sub-tasks."""

    def test_execute_tasks_calls_renderer_factory_once_per_subtask(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        seen = []

        def renderer_factory(task_id, description):
            seen.append((task_id, description))
            return BufferedSubagentRenderer(task_id, description)

        captured_agent_kwargs = []
        real_init = DelegatedCodingAgent.__init__

        def capturing_init(self, **kwargs):
            captured_agent_kwargs.append(kwargs)
            real_init(self, **kwargs)

        async def fake_execute(self, task):
            return {"status": "completed", "summary": f"done:{task.description}"}

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            results = asyncio.run(manager.execute_tasks(
                ["task a", "task b"],
                manager=object(), provider=object(), model=None, console=object(), workspace_root="/tmp",
                renderer_factory=renderer_factory,
            ))

        self.assertEqual(len(results), 2)
        self.assertEqual(len(captured_agent_kwargs), 2)
        # Each DelegatedCodingAgent got its OWN per-task renderer_factory
        # closure (not the same shared callback, and not the raw
        # execute_tasks-level renderer_factory passed straight through) --
        # invoking it must reach the outer renderer_factory with this
        # sub-task's own task_id/description.
        descriptions_seen = set()
        for kwargs in captured_agent_kwargs:
            factory = kwargs["renderer_factory"]
            self.assertIsNotNone(factory)
            renderer = factory()
            self.assertIsInstance(renderer, BufferedSubagentRenderer)
            descriptions_seen.add(renderer.description)
        self.assertEqual(descriptions_seen, {"task a", "task b"})
        self.assertEqual(len(seen), 2)
        self.assertEqual(len({task_id for task_id, _ in seen}), 2)

    def test_no_renderer_factory_means_delegated_agent_gets_none(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        captured_agents = []
        real_init = DelegatedCodingAgent.__init__

        def capturing_init(self, **kwargs):
            captured_agents.append(kwargs)
            real_init(self, **kwargs)

        async def fake_execute(self, task):
            return {"status": "completed", "summary": "done"}

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            asyncio.run(manager.execute_tasks(
                ["task a"], manager=object(), provider=object(), model=None, console=object(), workspace_root="/tmp",
            ))

        self.assertEqual(len(captured_agents), 1)
        self.assertIsNone(captured_agents[0]["renderer_factory"])


class ExecuteTasksAgentTypesTests(unittest.TestCase):
    """Declarative subagent types (agent_definitions.py): execute_tasks's
    agent_types param resolves a named definition per sub-task and
    overrides that ONE sub-task's system prompt/model/provider, without
    affecting the others or any existing caller that omits it entirely."""

    def _capture(self):
        captured_agent_kwargs = []
        real_init = DelegatedCodingAgent.__init__

        def capturing_init(self, **kwargs):
            captured_agent_kwargs.append(kwargs)
            real_init(self, **kwargs)

        async def fake_execute(self, task):
            return {"status": "completed", "summary": f"done:{task.description}"}

        return captured_agent_kwargs, capturing_init, fake_execute

    def test_named_agent_type_overrides_prompt_model_and_provider(self):
        from tamfis_code.agent_definitions import AgentDefinition

        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        definitions = {
            "reviewer": AgentDefinition(
                name="reviewer", description="Reviews code", system_prompt="You are a strict reviewer.",
                model="qwen/qwen3-coder", provider="openrouter", source="user config",
            )
        }
        captured, capturing_init, fake_execute = self._capture()

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agent_definitions.load_agent_definitions", return_value=definitions), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            asyncio.run(manager.execute_tasks(
                ["review this diff"], manager=object(), provider="shared-provider", model="shared-model",
                console=object(), workspace_root="/tmp", agent_types=["reviewer"],
            ))

        self.assertEqual(len(captured), 1)
        kwargs = captured[0]
        self.assertEqual(kwargs["extra_system_prompt"], "You are a strict reviewer.")
        self.assertEqual(kwargs["model"], "qwen/qwen3-coder")
        from tamfis_code.providers import ProviderType
        self.assertEqual(kwargs["provider"], ProviderType.OPENROUTER)

    def test_unknown_agent_type_falls_back_to_shared_model_and_provider(self):
        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        captured, capturing_init, fake_execute = self._capture()

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agent_definitions.load_agent_definitions", return_value={}), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            asyncio.run(manager.execute_tasks(
                ["do something"], manager=object(), provider="shared-provider", model="shared-model",
                console=object(), workspace_root="/tmp", agent_types=["not_a_real_agent"],
            ))

        kwargs = captured[0]
        self.assertIsNone(kwargs["extra_system_prompt"])
        self.assertEqual(kwargs["model"], "shared-model")
        self.assertEqual(kwargs["provider"], "shared-provider")

    def test_none_entries_in_agent_types_are_a_per_task_no_op(self):
        from tamfis_code.agent_definitions import AgentDefinition

        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        definitions = {"reviewer": AgentDefinition(
            name="reviewer", description="", system_prompt="reviewer prompt", model=None, provider=None,
            source="user config",
        )}
        captured, capturing_init, fake_execute = self._capture()

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agent_definitions.load_agent_definitions", return_value=definitions), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            asyncio.run(manager.execute_tasks(
                ["task a", "task b"], manager=object(), provider="shared", model="shared-model",
                console=object(), workspace_root="/tmp", agent_types=["reviewer", None],
            ))

        self.assertEqual(len(captured), 2)
        prompts = {kwargs.get("extra_system_prompt") for kwargs in captured}
        self.assertEqual(prompts, {"reviewer prompt", None})

    def test_definition_with_no_model_or_provider_only_adds_the_prompt(self):
        from tamfis_code.agent_definitions import AgentDefinition

        fake_workspace = MagicMock(session_id=1, workspace_root="/tmp")
        definitions = {"planner": AgentDefinition(
            name="planner", description="", system_prompt="Plan first, act second.", model=None, provider=None,
            source="project config",
        )}
        captured, capturing_init, fake_execute = self._capture()

        with patch("tamfis_code.workspace.resolve_swarm_subtask_workspace", return_value=fake_workspace), \
                patch("tamfis_code.agent_definitions.load_agent_definitions", return_value=definitions), \
                patch("tamfis_code.agents.DelegatedCodingAgent.__init__", new=capturing_init), \
                patch("tamfis_code.agents.DelegatedCodingAgent.execute", new=fake_execute):
            manager = AgentManager()
            asyncio.run(manager.execute_tasks(
                ["task a"], manager=object(), provider="shared-provider", model="shared-model",
                console=object(), workspace_root="/tmp", agent_types=["planner"],
            ))

        kwargs = captured[0]
        self.assertEqual(kwargs["extra_system_prompt"], "Plan first, act second.")
        self.assertEqual(kwargs["model"], "shared-model")
        self.assertEqual(kwargs["provider"], "shared-provider")


class SwarmDefaultConcurrencyTests(unittest.TestCase):
    def test_run_swarm_defaults_to_higher_concurrency_than_execute_tasks(self):
        import inspect

        run_swarm_default = inspect.signature(run_swarm).parameters["max_concurrency"].default
        execute_tasks_default = inspect.signature(AgentManager.execute_tasks).parameters["max_concurrency"].default
        self.assertEqual(run_swarm_default, 3)
        # execute_tasks's own bare default is deliberately untouched -- the
        # higher default only applies to the hardened run_swarm entry point.
        self.assertEqual(execute_tasks_default, 1)
        self.assertGreater(run_swarm_default, execute_tasks_default)


class RunSwarmTests(unittest.TestCase):
    def _fake_console(self, is_terminal=False):
        console = MagicMock()
        console.is_terminal = is_terminal
        return console

    def test_mutate_under_ask_policy_refuses_up_front_no_subtasks_run(self):
        with patch("tamfis_code.agents.AgentManager.execute_tasks") as fake_execute_tasks:
            with self.assertRaises(ValueError) as ctx:
                asyncio.run(run_swarm(
                    ["do a thing"], manager=object(), provider=object(), model=None,
                    console=self._fake_console(), workspace_root="/tmp",
                    approval_policy="ask", mutate=True,
                ))
        self.assertIn("cannot prompt for approval", str(ctx.exception))
        self.assertIn("--mutate", str(ctx.exception))
        fake_execute_tasks.assert_not_called()

    def test_mutate_under_auto_approving_policy_is_allowed(self):
        async def fake_execute_tasks(self, tasks, **kwargs):
            return [{"task_id": "t1", "description": tasks[0], "status": "completed", "result": {"summary": "done"}}]

        with patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            results = asyncio.run(run_swarm(
                ["do a thing"], manager=object(), provider=object(), model=None,
                console=self._fake_console(), workspace_root="/tmp",
                approval_policy="accept-edits", mutate=True,
            ))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "completed")

    def test_read_only_by_default_calls_execute_tasks_with_chat_mode(self):
        captured = {}

        async def fake_execute_tasks(self, tasks, **kwargs):
            captured.update(kwargs)
            return [{"task_id": "t1", "description": tasks[0], "status": "completed", "result": {}}]

        with patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            asyncio.run(run_swarm(
                ["look into this"], manager=object(), provider=object(), model=None,
                console=self._fake_console(), workspace_root="/tmp", session_id=7,
            ))

        self.assertEqual(captured["mode"], "chat")
        self.assertEqual(captured["parent_session_id"], 7)
        self.assertIsNotNone(captured["renderer_factory"])

    def test_mutate_true_with_allowed_policy_calls_execute_tasks_with_agent_mode(self):
        captured = {}

        async def fake_execute_tasks(self, tasks, **kwargs):
            captured.update(kwargs)
            return [{"task_id": "t1", "description": tasks[0], "status": "completed", "result": {}}]

        with patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            asyncio.run(run_swarm(
                ["fix this"], manager=object(), provider=object(), model=None,
                console=self._fake_console(), workspace_root="/tmp",
                approval_policy="auto", mutate=True,
            ))

        self.assertEqual(captured["mode"], "agent")

    def test_non_tty_console_never_touches_rich_live(self):
        async def fake_execute_tasks(self, tasks, **kwargs):
            # Exercise the renderer_factory the way execute_tasks really would.
            factory = kwargs["renderer_factory"]
            renderer = factory("real_task_id", tasks[0])
            renderer.handle_event({"type": "tool_call_requested", "name": "read_file"})
            return [{"task_id": "real_task_id", "description": tasks[0], "status": "completed", "result": {}}]

        with patch("rich.live.Live") as fake_live, \
                patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            asyncio.run(run_swarm(
                ["task a"], manager=object(), provider=object(), model=None,
                console=self._fake_console(is_terminal=False), workspace_root="/tmp",
            ))
        fake_live.assert_not_called()

    def test_tty_console_builds_and_stops_a_single_live(self):
        async def fake_execute_tasks(self, tasks, **kwargs):
            return [{"task_id": "t1", "description": tasks[0], "status": "completed", "result": {}}]

        fake_live_instance = MagicMock()
        with patch("rich.live.Live", return_value=fake_live_instance) as fake_live_cls, \
                patch("tamfis_code.agents.AgentManager.execute_tasks", new=fake_execute_tasks):
            asyncio.run(run_swarm(
                ["task a"], manager=object(), provider=object(), model=None,
                console=self._fake_console(is_terminal=True), workspace_root="/tmp",
            ))
        fake_live_cls.assert_called_once()
        fake_live_instance.start.assert_called_once()
        fake_live_instance.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
