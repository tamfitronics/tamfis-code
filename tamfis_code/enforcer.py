"""TAMFIS-CODE test enforcer -- detects and runs the current workspace's own
test suite (Python/pytest, Node/npm, Rust/cargo), scoped to whatever
directory tamfis-code is actually pointed at (respecting --cwd), the same
way every other command resolves its workspace.
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from .workspace import _project_metadata

try:
    import click
except ImportError:
    click = None


class TestEnforcer:
    """Detects and runs the current workspace's own test/build commands."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.workspace_root),
            "python": {"tests": [], "summary": {}},
            "frontend_tests": {"status": "checking", "results": []},
        }
        self._print = print

    def _print_progress(self, message: str, status: str = "info", detail: str = ""):
        """Print progress with formatting"""
        emojis = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "running": "🔄",
            "python": "🐍",
            "node": "📦",
            "frontend": "🎨",
            "done": "🎉"
        }
        prefix = emojis.get(status, "ℹ️")
        if detail:
            self._print(f"{prefix} {message}\n   └─ {detail}")
        else:
            self._print(f"{prefix} {message}")
        sys.stdout.flush()

    def _run_cmd(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 120) -> Dict[str, Any]:
        cwd = cwd or self.workspace_root
        try:
            start = time.time()
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            elapsed = time.time() - start
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed": elapsed,
                "cmd": " ".join(cmd)
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {timeout}s",
                "elapsed": timeout,
                "cmd": " ".join(cmd)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed": 0,
                "cmd": " ".join(cmd)
            }

    def _metadata(self) -> Dict[str, Any]:
        try:
            top_level = list(self.workspace_root.iterdir()) if self.workspace_root.is_dir() else []
        except OSError:
            top_level = []
        return _project_metadata(self.workspace_root, top_level)

    def run(self) -> Dict[str, Any]:
        """Detect and run this workspace's own test commands."""
        self._print_progress("=" * 70, "info")
        self._print_progress("🔧 TAMFIS-CODE Test Enforcer", "info")
        self._print_progress("=" * 70, "info")
        self._print_progress(f"📂 Workspace: {self.workspace_root}", "info")

        metadata = self._metadata()
        test_commands = metadata["test_commands"]
        if not test_commands:
            self._print_progress(
                "No recognised test setup found in this workspace "
                "(no pyproject.toml/pytest.ini, package.json, or Cargo.toml at its root).",
                "warning",
            )

        if "pytest -q" in test_commands:
            self._run_backend_tests()
        if "npm test" in test_commands:
            self._run_frontend_tests()
        if "cargo test" in test_commands:
            self._run_cargo_tests()

        self._generate_summary()
        return self.results

    def _run_backend_tests(self):
        """Run this workspace's Python tests (pytest)."""
        self._print_progress("", "info")
        self._print_progress("🐍 Python Tests", "python")
        self._print_progress("-" * 50, "info")

        test_dir = self.workspace_root / "tests"
        test_files = sorted(test_dir.glob("test_*.py")) if test_dir.exists() else []

        if test_files:
            self._print_progress(f"📁 Found {len(test_files)} test files in tests/", "info")
            passed = 0
            for test_file in test_files:
                self._print_progress(f"▶️ Running {test_file.name}...", "running")
                result = self._run_cmd(
                    [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "-q"],
                    cwd=self.workspace_root,
                    timeout=60
                )
                if result["success"]:
                    passed += 1
                    self._print_progress(f"✅ {test_file.name} PASSED ({result['elapsed']:.2f}s)", "success")
                else:
                    self._print_progress(f"❌ {test_file.name} FAILED ({result['elapsed']:.2f}s)", "error")
                    error_lines = [l for l in result["stdout"].split('\n') if 'FAILED' in l or 'ERROR' in l]
                    for line in error_lines[:3]:
                        self._print_progress(f"   └─ {line.strip()}", "error")
            self.results["python"]["summary"]["total"] = len(test_files)
            self.results["python"]["summary"]["passed"] = passed
            self._print_progress(f"📊 Summary: {passed}/{len(test_files)} test files passed",
                               "success" if passed == len(test_files) else "warning")
            return

        # No tests/test_*.py layout -- still a recognised Python project
        # (pyproject.toml/pytest.ini present), so run pytest at the root and
        # let it do its own discovery (covers src-layout/colocated tests).
        self._print_progress("No tests/test_*.py found; running `pytest -q` at the workspace root", "running")
        result = self._run_cmd([sys.executable, "-m", "pytest", "-q"], cwd=self.workspace_root, timeout=300)
        self.results["python"]["summary"]["total"] = 1
        self.results["python"]["summary"]["passed"] = 1 if result["success"] else 0
        if result["success"]:
            self._print_progress(f"✅ pytest PASSED ({result['elapsed']:.2f}s)", "success")
        else:
            self._print_progress(f"❌ pytest FAILED ({result['elapsed']:.2f}s)", "error", result.get("error", ""))

    def _run_frontend_tests(self):
        """Run this workspace's Node/frontend test scripts (package.json)."""
        self._print_progress("", "info")
        self._print_progress("🎨 Frontend Tests", "frontend")
        self._print_progress("-" * 50, "info")

        # Check if this is actually a Node/JavaScript project
        from .workspace import _discover_project_type
        project_type = _discover_project_type(self.workspace_root)
        
        # Skip npm if not a Node/JavaScript project
        lang = project_type.get("language", "")
        if lang not in ["JavaScript/TypeScript", "JavaScript", "TypeScript"]:
            self._print_progress(f"Not a Node/JavaScript project (detected: {lang}), skipping npm", "info")
            self.results["frontend_tests"]["status"] = "skipped"
            return

        pkg_file = self.workspace_root / "package.json"
        if not pkg_file.exists():
            self._print_progress("No package.json found", "warning")
            self.results["frontend_tests"]["status"] = "no_package"
            return

        import json
        with open(pkg_file) as f:
            pkg = json.load(f)
        scripts = pkg.get("scripts", {})
        self._print_progress(f"📁 Found {len(scripts)} scripts in package.json", "info")

        npm_check = self._run_cmd(["npm", "--version"], cwd=self.workspace_root)
        if not npm_check["success"]:
            self._print_progress("❌ npm not available", "error")
            self.results["frontend_tests"]["status"] = "no_npm"
            return
        self._print_progress(f"✅ npm {npm_check['stdout'].strip()} available", "success")

        node_modules = self.workspace_root / "node_modules"
        if not node_modules.exists():
            self._print_progress("📦 Installing npm dependencies... (this may take a while)", "running")
            result = self._run_cmd(["npm", "install"], cwd=self.workspace_root, timeout=300)
            if result["success"]:
                self._print_progress("✅ Dependencies installed", "success")
            else:
                self._print_progress("❌ Dependency installation failed", "error")
                self.results["frontend_tests"]["status"] = "install_failed"
                return

        test_scripts = ["test", "test:unit", "test:integration", "build", "typecheck"]
        available = [s for s in test_scripts if s in scripts]

        if not available:
            self._print_progress("No test scripts found in package.json", "warning")
            self.results["frontend_tests"]["status"] = "no_scripts"
            return

        self._print_progress(f"📁 Found {len(available)} scripts to run", "info")

        results = []
        for script in available:
            self._print_progress(f"▶️ Running npm {script}...", "running")
            result = self._run_cmd(["npm", "run", script], cwd=self.workspace_root, timeout=120)
            if result["success"]:
                self._print_progress(f"✅ npm {script} PASSED ({result['elapsed']:.2f}s)", "success")
            else:
                self._print_progress(f"❌ npm {script} FAILED ({result['elapsed']:.2f}s)", "error")
                if result.get("stderr"):
                    error_lines = [l for l in result["stderr"].split('\n') if l.strip()]
                    for line in error_lines[:2]:
                        self._print_progress(f"   └─ {line.strip()[:100]}", "error")
            results.append({
                "script": script,
                "passed": result["success"],
                "elapsed": result.get("elapsed", 0)
            })

        self.results["frontend_tests"]["results"] = results
        self.results["frontend_tests"]["status"] = "done"

        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        self._print_progress(f"📊 Frontend Summary: {passed}/{total} scripts passed",
                           "success" if passed == total else "warning")

    def _run_cargo_tests(self):
        """Run this workspace's Rust tests (cargo test)."""
        self._print_progress("", "info")
        self._print_progress("🦀 Rust Tests", "info")
        self._print_progress("-" * 50, "info")
        result = self._run_cmd(["cargo", "test"], cwd=self.workspace_root, timeout=300)
        self.results["rust"] = {"passed": result["success"], "elapsed": result.get("elapsed", 0)}
        if result["success"]:
            self._print_progress(f"✅ cargo test PASSED ({result['elapsed']:.2f}s)", "success")
        else:
            self._print_progress(f"❌ cargo test FAILED ({result['elapsed']:.2f}s)", "error", result.get("error", ""))

    def _generate_summary(self):
        """Generate summary"""
        self._print_progress("", "info")
        self._print_progress("=" * 70, "info")
        self._print_progress("📊 TEST ENFORCEMENT SUMMARY", "info")
        self._print_progress("=" * 70, "info")

        py = self.results["python"]["summary"]
        if py.get("total", 0) > 0:
            passed = py.get("passed", 0)
            total = py.get("total", 0)
            self._print_progress(f"🐍 Python: {passed}/{total} passed",
                               "success" if passed == total else "warning")

        fe = self.results["frontend_tests"]
        if fe.get("results"):
            passed = sum(1 for r in fe["results"] if r["passed"])
            total = len(fe["results"])
            self._print_progress(f"🎨 Frontend: {passed}/{total} scripts passed",
                               "success" if passed == total else "warning")

        rust = self.results.get("rust")
        if rust:
            self._print_progress("🦀 Rust: " + ("passed" if rust["passed"] else "FAILED"),
                               "success" if rust["passed"] else "error")

        self._print_progress("=" * 70, "info")
        self._print_progress("✅ Test enforcement completed!", "success")


def run_enforcer(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run the test enforcer against workspace_root (or the current directory)."""
    enforcer = TestEnforcer(workspace_root)
    return enforcer.run()


def add_enforcer_command(cli):
    """Add enforcer command to CLI"""
    if click is None:
        return cli

    @cli.command('enforce')
    @click.option('--python', '-p', is_flag=True, help='Only run Python tests')
    @click.option('--node', '-n', is_flag=True, help="Only run Node.js/frontend tests")
    @click.option('--frontend', '-f', is_flag=True, help='Only run frontend tests')
    @click.pass_context
    def enforce_cmd(ctx, python: bool, node: bool, frontend: bool):
        """Enforce and run all tests with real-time progress"""
        workspace_root = ctx.obj.get("workspace_root") if ctx.obj else None
        enforcer = TestEnforcer(workspace_root)

        if python:
            enforcer._run_backend_tests()
        elif node or frontend:
            enforcer._run_frontend_tests()
        else:
            enforcer.run()

        print("\n✅ Test enforcement completed!")

    return cli
