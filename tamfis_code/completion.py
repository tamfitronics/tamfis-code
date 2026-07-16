#!/usr/bin/env python3
"""Shell completion for TAMFIS-CODE"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

class ShellCompleter:
    """Dynamic shell completion generator"""
    
    SUPPORTED_SHELLS = ['bash', 'zsh', 'fish', 'powershell']
    
    COMMANDS = {
        'chat': 'Start an interactive chat session',
        'run': 'Execute a command with context',
        'edit': 'Edit files with AI assistance',
        'review': 'Review code changes',
        'plan': 'Create an execution plan',
        'ingest': 'Ingest documents for context',
        'session': 'Manage sessions',
        'config': 'View/Edit configuration',
        'doctor': 'Run diagnostics',
        'version': 'Show version',
    }
    
    FILE_COMMANDS = ['edit', 'review', 'ingest']
    
    @classmethod
    def generate_bash(cls) -> str:
        return '''_tamfis_code_completion() {
    local cur prev opts files
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    
    opts="chat run edit review plan ingest session config doctor version"
    
    case "${prev}" in
        edit|review|ingest)
            COMPREPLY=( $(compgen -f -- "${cur}") )
            return 0
            ;;
        *)
            COMPREPLY=( $(compgen -W "${opts}" -- "${cur}") )
            return 0
            ;;
    esac
}
complete -F _tamfis_code_completion tamfis-code
'''
    
    @classmethod
    def generate_zsh(cls) -> str:
        return '''#compdef tamfis-code

_tamfis_code() {
    local -a commands
    commands=(
        'chat:Start interactive chat session'
        'run:Execute command with context'
        'edit:Edit files with AI assistance'
        'review:Review code changes'
        'plan:Create execution plan'
        'ingest:Ingest documents for context'
        'session:Manage sessions'
        'config:View/Edit configuration'
        'doctor:Run diagnostics'
        'version:Show version'
    )
    
    _arguments -C \\
        '1: :->command' \\
        '*:: :->args'
    
    case $state in
        command)
            _describe 'commands' commands
            ;;
        args)
            case $line[1] in
                edit|review|ingest)
                    _files
                    ;;
            esac
            ;;
    esac
}
compdef _tamfis_code tamfis-code
'''
    
    @classmethod
    def generate_fish(cls) -> str:
        return '''function __fish_tamfis_code_needs_file
    set cmd (commandline -opc)
    contains -- $cmd[2] edit review ingest
end

complete -c tamfis-code -f
complete -c tamfis-code -n "__fish_use_subcommand" -a chat -d "Start interactive chat session"
complete -c tamfis-code -n "__fish_use_subcommand" -a run -d "Execute command with context"
complete -c tamfis-code -n "__fish_use_subcommand" -a edit -d "Edit files with AI assistance"
complete -c tamfis-code -n "__fish_use_subcommand" -a review -d "Review code changes"
complete -c tamfis-code -n "__fish_use_subcommand" -a plan -d "Create execution plan"
complete -c tamfis-code -n "__fish_use_subcommand" -a ingest -d "Ingest documents"
complete -c tamfis-code -n "__fish_use_subcommand" -a session -d "Manage sessions"
complete -c tamfis-code -n "__fish_use_subcommand" -a config -d "View/Edit configuration"
complete -c tamfis-code -n "__fish_use_subcommand" -a doctor -d "Run diagnostics"
complete -c tamfis-code -n "__fish_use_subcommand" -a version -d "Show version"

complete -c tamfis-code -n "__fish_tamfis_code_needs_file" -a "(__fish_complete_path)"
'''
    
    @classmethod
    def generate_powershell(cls) -> str:
        return '''function _tamfis_code_completion {
    param(
        $commandName,
        $parameterName,
        $wordToComplete,
        $commandAst,
        $fakeBoundParameters
    )
    
    $commands = @(
        "chat", "run", "edit", "review", "plan", 
        "ingest", "session", "config", "doctor", "version"
    )
    
    if ($parameterName -eq "command") {
        $commands | Where-Object { $_ -like "$wordToComplete*" }
    }
}

Register-ArgumentCompleter -Native -CommandName "tamfis-code" -ScriptBlock _tamfis_code_completion
'''
