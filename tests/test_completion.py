#!/usr/bin/env python3
"""Test shell completion"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tamfis_code.completion import ShellCompleter

class TestShellCompleter:
    """Test shell completion generator"""

    def test_generate_bash(self):
        """Test bash completion generation"""
        result = ShellCompleter.generate_bash()
        assert '_tamfis_code_completion' in result
        assert 'complete -F' in result
        assert 'tamfis-code' in result

    def test_generate_zsh(self):
        """Test zsh completion generation"""
        result = ShellCompleter.generate_zsh()
        assert '#compdef' in result
        assert '_tamfis_code' in result
        assert 'tamfis-code' in result

    def test_generate_fish(self):
        """Test fish completion generation"""
        result = ShellCompleter.generate_fish()
        assert 'complete -c tamfis-code' in result
        assert '__fish_tamfis_code_needs_file' in result

    def test_generate_powershell(self):
        """Test PowerShell completion generation"""
        result = ShellCompleter.generate_powershell()
        assert 'Register-ArgumentCompleter' in result
        assert 'tamfis-code' in result

    def test_supported_shells(self):
        """Test supported shells list"""
        assert 'bash' in ShellCompleter.SUPPORTED_SHELLS
        assert 'zsh' in ShellCompleter.SUPPORTED_SHELLS
        assert 'fish' in ShellCompleter.SUPPORTED_SHELLS
        assert 'powershell' in ShellCompleter.SUPPORTED_SHELLS

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
