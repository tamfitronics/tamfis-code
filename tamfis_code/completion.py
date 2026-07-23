#!/usr/bin/env python3
"""Shell completion for TAMFIS-CODE.

Command names/descriptions are introspected from the real Click CLI
(cli.py) rather than hand-maintained, so completions cannot drift out of
sync with the actual command surface again.
"""

from typing import Dict


class ShellCompleter:
    """Dynamic shell completion generator"""

    SUPPORTED_SHELLS = ['bash', 'zsh', 'fish', 'powershell']

    @classmethod
    def _commands(cls) -> Dict[str, str]:
        # Imported lazily -- cli.py imports this module (for the
        # `completion` subcommand itself), so a module-level import here
        # would be circular.
        from .cli import cli

        return {
            name: (cmd.get_short_help_str(limit=60) or name).replace("'", "")
            for name, cmd in sorted(cli.commands.items())
        }

    @classmethod
    def generate_bash(cls) -> str:
        opts = " ".join(cls._commands().keys())
        return f'''_tamfis_code_completion() {{
    local cur prev opts
    COMPREPLY=()
    cur="${{COMP_WORDS[COMP_CWORD]}}"
    prev="${{COMP_WORDS[COMP_CWORD-1]}}"

    opts="{opts}"

    if [ "${{COMP_CWORD}}" -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${{opts}}" -- "${{cur}}") )
        return 0
    fi

    COMPREPLY=( $(compgen -f -- "${{cur}}") )
    return 0
}}
complete -F _tamfis_code_completion tamfis-code
'''

    @classmethod
    def generate_zsh(cls) -> str:
        entries = "\n        ".join(
            f"'{name}:{desc}'" for name, desc in cls._commands().items()
        )
        return f'''#compdef tamfis-code

_tamfis_code() {{
    local -a commands
    commands=(
        {entries}
    )

    _arguments -C \\
        '1: :->command' \\
        '*:: :->args'

    case $state in
        command)
            _describe 'commands' commands
            ;;
        args)
            _files
            ;;
    esac
}}
compdef _tamfis_code tamfis-code
'''

    @classmethod
    def generate_fish(cls) -> str:
        lines = [
            f'complete -c tamfis-code -n "__fish_use_subcommand" -a {name} -d "{desc}"'
            for name, desc in cls._commands().items()
        ]
        body = "\n".join(lines)
        return f'''function __fish_tamfis_code_needs_file
    set cmd (commandline -opc)
    test (count $cmd) -ge 2
end

complete -c tamfis-code -f
{body}

complete -c tamfis-code -n "__fish_tamfis_code_needs_file" -a "(__fish_complete_path)"
'''

    @classmethod
    def generate_powershell(cls) -> str:
        commands = ", ".join(f'"{name}"' for name in cls._commands().keys())
        return f'''function _tamfis_code_completion {{
    param(
        $commandName,
        $parameterName,
        $wordToComplete,
        $commandAst,
        $fakeBoundParameters
    )

    $commands = @({commands})

    if ($parameterName -eq "command") {{
        $commands | Where-Object {{ $_ -like "$wordToComplete*" }}
    }}
}}

Register-ArgumentCompleter -Native -CommandName "tamfis-code" -ScriptBlock _tamfis_code_completion
'''
