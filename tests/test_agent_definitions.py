"""Tests for declarative subagent types (agent_definitions.py) -- tamfis-
code's `.claude/agents/*.md` equivalent: delegation already existed
(agents.py/swarm.py) but only as ad-hoc task strings sharing one model/
provider with no specialised instructions of their own."""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tamfis_code import agent_definitions as agent_definitions_module
from tamfis_code.agent_definitions import AgentDefinition, load_agent_definitions


class TestLoadAgentDefinitions:
    def setup_method(self):
        self._original_dir = agent_definitions_module.USER_AGENTS_DIR
        self.tmp = tempfile.TemporaryDirectory()
        agent_definitions_module.USER_AGENTS_DIR = Path(self.tmp.name) / "user" / "agents"

    def teardown_method(self):
        agent_definitions_module.USER_AGENTS_DIR = self._original_dir
        self.tmp.cleanup()

    def test_missing_directories_return_no_definitions(self):
        assert load_agent_definitions(str(Path(self.tmp.name) / "project")) == {}

    def test_loads_a_plain_definition_with_no_frontmatter(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "planner.md").write_text("Think before you act.\n")
        loaded = load_agent_definitions()
        assert set(loaded) == {"planner"}
        assert loaded["planner"].system_prompt == "Think before you act."
        assert loaded["planner"].model is None
        assert loaded["planner"].provider is None
        assert loaded["planner"].source == "user config"

    def test_loads_frontmatter_description_model_and_provider(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "reviewer.md").write_text(
            "---\ndescription: Reviews code for bugs\nmodel: qwen/qwen3-coder\nprovider: openrouter\n---\n"
            "You are a strict, terse code reviewer.\n"
        )
        loaded = load_agent_definitions()
        definition = loaded["reviewer"]
        assert definition.description == "Reviews code for bugs"
        assert definition.model == "qwen/qwen3-coder"
        assert definition.provider == "openrouter"
        assert definition.system_prompt == "You are a strict, terse code reviewer."

    def test_project_definition_overrides_same_named_user_definition(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "reviewer.md").write_text("user version")
        project_root = Path(self.tmp.name) / "project"
        (project_root / ".tamfis" / "agents").mkdir(parents=True)
        (project_root / ".tamfis" / "agents" / "reviewer.md").write_text("project version")
        loaded = load_agent_definitions(str(project_root))
        assert loaded["reviewer"].system_prompt == "project version"
        assert loaded["reviewer"].source == "project config"

    def test_non_md_files_are_ignored(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "notes.txt").write_text("not a definition")
        assert load_agent_definitions() == {}

    def test_invalid_name_characters_are_rejected(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "not valid!.md").write_text("hi")
        assert load_agent_definitions() == {}

    def test_empty_body_after_frontmatter_is_rejected(self):
        agent_definitions_module.USER_AGENTS_DIR.mkdir(parents=True)
        (agent_definitions_module.USER_AGENTS_DIR / "empty.md").write_text("---\ndescription: x\n---\n")
        assert load_agent_definitions() == {}
