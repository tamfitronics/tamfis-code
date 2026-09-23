from pathlib import Path

from tamfis_code.mcp import MCPServer
from tamfis_code.runner_local import _read_only_observation_key, _resolve_argument_path


def test_quoted_absolute_path_is_not_prefixed_by_workspace():
    root = "/home/tamfiscode"
    assert _resolve_argument_path('"/home/tamfiscode"', "/home") == Path(root).resolve()
    server = MCPServer(workspace_root="/home")
    assert server._resolve_in_workspace('"/home/tamfiscode"') == Path(root).resolve()


def test_equivalent_read_paths_share_one_observation_key():
    absolute = _read_only_observation_key(
        "list_directory", {"path": '"/home/tamfiscode"', "depth": 1}, "/home"
    )
    relative = _read_only_observation_key(
        "list_directory", {"path": "/home/tamfiscode", "depth": 1}, "/home"
    )
    assert absolute == relative


def test_mutating_and_command_tools_are_never_reusable_observations():
    assert _read_only_observation_key("write_file", {"path": "x"}, "/tmp") is None
    assert _read_only_observation_key("execute_command", {"command": "ls"}, "/tmp") is None
