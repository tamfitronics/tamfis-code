"""
TAMFIS-CODE Test Enforcer - Properly handles sibling directories
CWD: /home/tamfisgpt
- tamgpt6/ (backend, this repo)
- tamfis-frontend/ (React frontend, sibling)
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import click for CLI integration
try:
    import click
except ImportError:
    click = None


class TestEnforcer:
    """Test enforcer that properly handles sibling directories"""
    
    def __init__(self):
        self.base = Path("/home/tamfisgpt")
        self.backend = self.base / "tamgpt6"
        self.frontend = self.base / "tamfis-frontend"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "workspace": str(self.base),
            "backend": str(self.backend),
            "frontend": {
                "exists": self.frontend.exists(),
                "path": str(self.frontend),
            },
            "python": {"tests": [], "summary": {}},
            "node": {"scripts": [], "summary": {}},
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
        cwd = cwd or self.backend
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
    
    def run(self) -> Dict[str, Any]:
        """Run all enforcements"""
        self._print_progress("=" * 70, "info")
        self._print_progress("🔧 TAMFIS-CODE Test Enforcer", "info")
        self._print_progress("=" * 70, "info")
        self._print_progress(f"📂 Workspace: {self.base}", "info")
        self._print_progress(f"📂 Backend: {self.backend}", "info")
        
        # Check frontend
        if self.frontend.exists():
            self._print_progress(f"🎨 Frontend: {self.frontend}", "success")
            self.results["frontend"]["exists"] = True
        else:
            self._print_progress(f"🎨 Frontend: NOT FOUND at {self.frontend}", "error")
            self.results["frontend"]["exists"] = False
        
        # Run backend tests
        self._run_backend_tests()
        
        # Run frontend tests if it exists
        if self.frontend.exists():
            self._run_frontend_tests()
        
        self._generate_summary()
        return self.results
    
    def _run_backend_tests(self):
        """Run backend (tamgpt6) tests"""
        self._print_progress("", "info")
        self._print_progress("🐍 Backend Tests (tamgpt6)", "python")
        self._print_progress("-" * 50, "info")
        
        test_dir = self.backend / "tests"
        if not test_dir.exists():
            self._print_progress("No tests directory found", "warning")
            return
        
        test_files = list(test_dir.glob("test_*.py"))
        self._print_progress(f"📁 Found {len(test_files)} test files", "info")
        
        passed = 0
        for test_file in test_files:
            self._print_progress(f"▶️ Running {test_file.name}...", "running")
            result = self._run_cmd(
                [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short", "-q"],
                cwd=self.backend,
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
    
    def _run_frontend_tests(self):
        """Run frontend (tamfis-frontend) tests - ACTUALLY access the sibling directory"""
        self._print_progress("", "info")
        self._print_progress("🎨 Frontend Tests (tamfis-frontend)", "frontend")
        self._print_progress("-" * 50, "info")
        
        if not self.frontend.exists():
            self._print_progress("❌ Frontend directory not found", "error")
            return
        
        # List what's in the frontend directory
        self._print_progress(f"📁 Checking frontend at: {self.frontend}", "info")
        try:
            files = list(self.frontend.iterdir())
            self._print_progress(f"📁 Found {len(files)} items in frontend", "info")
            for f in files[:10]:
                self._print_progress(f"   └─ {f.name}", "info")
        except Exception as e:
            self._print_progress(f"❌ Cannot read frontend directory: {e}", "error")
            return
        
        # Check package.json
        pkg_file = self.frontend / "package.json"
        if not pkg_file.exists():
            self._print_progress("❌ No package.json found in frontend!", "error")
            self.results["frontend_tests"]["status"] = "no_package"
            return
        
        self._print_progress("✅ package.json found!", "success")
        
        # Read package.json
        import json
        with open(pkg_file) as f:
            pkg = json.load(f)
        scripts = pkg.get("scripts", {})
        self._print_progress(f"📁 Found {len(scripts)} scripts in package.json", "info")
        for script in scripts:
            self._print_progress(f"   └─ {script}: {scripts[script][:50]}...", "info")
        
        # Check npm
        npm_check = self._run_cmd(["npm", "--version"], cwd=self.frontend)
        if not npm_check["success"]:
            self._print_progress("❌ npm not available", "error")
            self.results["frontend_tests"]["status"] = "no_npm"
            return
        self._print_progress(f"✅ npm {npm_check['stdout'].strip()} available", "success")
        
        # Install dependencies if needed
        node_modules = self.frontend / "node_modules"
        if not node_modules.exists():
            self._print_progress("📦 Installing npm dependencies... (this may take a while)", "running")
            result = self._run_cmd(["npm", "install"], cwd=self.frontend, timeout=300)
            if result["success"]:
                self._print_progress("✅ Dependencies installed", "success")
            else:
                self._print_progress("❌ Dependency installation failed", "error")
                self.results["frontend_tests"]["status"] = "install_failed"
                return
        
        # Run test scripts
        test_scripts = ["test", "test:unit", "test:integration", "build", "typecheck", "dev"]
        available = [s for s in test_scripts if s in scripts]
        
        if not available:
            self._print_progress("No test scripts found in package.json", "warning")
            self.results["frontend_tests"]["status"] = "no_scripts"
            return
        
        self._print_progress(f"📁 Found {len(available)} scripts to run", "info")
        
        results = []
        for script in available:
            self._print_progress(f"▶️ Running npm {script}...", "running")
            result = self._run_cmd(["npm", "run", script], cwd=self.frontend, timeout=120)
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
    
    def _generate_summary(self):
        """Generate summary"""
        self._print_progress("", "info")
        self._print_progress("=" * 70, "info")
        self._print_progress("📊 TEST ENFORCEMENT SUMMARY", "info")
        self._print_progress("=" * 70, "info")
        
        # Backend Python
        py = self.results["python"]["summary"]
        if py.get("total", 0) > 0:
            passed = py.get("passed", 0)
            total = py.get("total", 0)
            self._print_progress(f"🐍 Backend: {passed}/{total} tests passed", 
                               "success" if passed == total else "warning")
        
        # Frontend
        fe = self.results["frontend_tests"]
        if fe.get("results"):
            passed = sum(1 for r in fe["results"] if r["passed"])
            total = len(fe["results"])
            self._print_progress(f"🎨 Frontend: {passed}/{total} scripts passed", 
                               "success" if passed == total else "warning")
        else:
            self._print_progress(f"🎨 Frontend: {fe.get('status', 'unknown')}", "warning")
        
        self._print_progress("=" * 70, "info")
        self._print_progress("✅ Test enforcement completed!", "success")


def run_enforcer():
    """Run the test enforcer"""
    enforcer = TestEnforcer()
    return enforcer.run()


def add_enforcer_command(cli):
    """Add enforcer command to CLI"""
    if click is None:
        return cli
    
    @cli.command('enforce')
    @click.option('--python', '-p', is_flag=True, help='Only run Python tests')
    @click.option('--node', '-n', is_flag=True, help='Only run Node.js tests')
    @click.option('--frontend', '-f', is_flag=True, help='Only run frontend tests')
    def enforce_cmd(python: bool, node: bool, frontend: bool):
        """Enforce and run all tests with real-time progress"""
        enforcer = TestEnforcer()
        
        if python:
            enforcer._run_backend_tests()
        elif node:
            # Run node tests (which is actually frontend)
            enforcer._run_frontend_tests()
        elif frontend:
            enforcer._run_frontend_tests()
        else:
            enforcer.run()
        
        print("\n✅ Test enforcement completed!")
    
    return cli
