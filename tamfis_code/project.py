"""Project stack detection - identify language, framework, and package manager."""

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class ProjectStack:
    """Detected project stack information."""
    language: str
    framework: Optional[str] = None
    package_manager: Optional[str] = None
    manifest_files: List[str] = field(default_factory=list)
    build_system: Optional[str] = None
    test_command: Optional[str] = None
    install_command: Optional[str] = None
    run_command: Optional[str] = None


class ProjectDetector:
    """Detect project type from manifest files and directory structure."""
    
    # Manifest file patterns for different stacks
    MANIFEST_PATTERNS = {
        "node": {
            "files": ["package.json"],
            "language": "javascript",
            "package_manager": "npm",
            "test_command": "npm test",
            "install_command": "npm install",
            "run_command": "npm start",
        },
        "python": {
            "files": ["pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"],
            "language": "python",
            "package_manager": "pip",
            "test_command": "pytest",
            "install_command": "pip install -r requirements.txt",
            "run_command": "python -m <module>",
        },
        "php": {
            "files": ["composer.json", "wp-content/themes", "wp-content/plugins"],
            "language": "php",
            "package_manager": "composer",
            "test_command": "phpunit",
            "install_command": "composer install",
            "run_command": "php -S localhost:8000",
        },
        "rust": {
            "files": ["Cargo.toml", "Cargo.lock"],
            "language": "rust",
            "package_manager": "cargo",
            "test_command": "cargo test",
            "install_command": "cargo build",
            "run_command": "cargo run",
        },
        "go": {
            "files": ["go.mod", "go.sum"],
            "language": "go",
            "package_manager": "go",
            "test_command": "go test ./...",
            "install_command": "go mod download",
            "run_command": "go run .",
        },
        "java": {
            "files": ["pom.xml", "build.gradle", "gradle.build"],
            "language": "java",
            "package_manager": "maven",
            "test_command": "mvn test",
            "install_command": "mvn install",
            "run_command": "mvn spring-boot:run",
        },
        "ruby": {
            "files": ["Gemfile", "Rakefile"],
            "language": "ruby",
            "package_manager": "bundler",
            "test_command": "rake test",
            "install_command": "bundle install",
            "run_command": "rails server",
        },
        "dotnet": {
            "files": ["*.csproj", "*.fsproj", "*.vbproj", "project.json"],
            "language": "csharp",
            "package_manager": "dotnet",
            "test_command": "dotnet test",
            "install_command": "dotnet restore",
            "run_command": "dotnet run",
        },
        "wordpress": {
            "files": ["wp-config.php", "wp-content/themes", "wp-content/plugins"],
            "language": "php",
            "framework": "wordpress",
            "package_manager": None,
            "test_command": None,
            "install_command": None,
            "run_command": None,
        },
        "django": {
            "files": ["manage.py", "settings.py", "requirements.txt"],
            "language": "python",
            "framework": "django",
            "package_manager": "pip",
            "test_command": "python manage.py test",
            "install_command": "pip install -r requirements.txt",
            "run_command": "python manage.py runserver",
        },
        "react": {
            "files": ["package.json"],
            "language": "javascript",
            "framework": "react",
            "package_manager": "npm",
            "test_command": "npm test",
            "install_command": "npm install",
            "run_command": "npm start",
        },
        "vue": {
            "files": ["package.json"],
            "language": "javascript",
            "framework": "vue",
            "package_manager": "npm",
            "test_command": "npm test",
            "install_command": "npm install",
            "run_command": "npm run serve",
        },
        "angular": {
            "files": ["package.json", "angular.json"],
            "language": "javascript",
            "framework": "angular",
            "package_manager": "npm",
            "test_command": "ng test",
            "install_command": "npm install",
            "run_command": "ng serve",
        },
    }
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root).resolve()
        self.cache: Dict[str, Any] = {}
    
    def detect(self, force: bool = False) -> ProjectStack:
        """Detect the project stack."""
        if not force and "stack" in self.cache:
            return self.cache["stack"]
        
        # Check for manifest files in order of specificity
        detected = self._detect_from_manifests()
        if detected:
            self.cache["stack"] = detected
            return detected
        
        # Default fallback - treat as generic project
        return ProjectStack(
            language="unknown",
            package_manager=None,
            manifest_files=[],
        )
    
    def _detect_from_manifests(self) -> Optional[ProjectStack]:
        """Detect project type by checking for manifest files."""
        root = self.workspace_root
        
        # Check for WordPress first (most specific)
        if self._is_wordpress():
            return ProjectStack(
                language="php",
                framework="wordpress",
                package_manager=None,
                manifest_files=["wp-config.php", "wp-content"],
                test_command=None,
                install_command=None,
                run_command=None,
            )
        
        # Check for Django
        if self._is_django():
            return ProjectStack(
                language="python",
                framework="django",
                package_manager="pip",
                manifest_files=["manage.py", "settings.py"],
                test_command="python manage.py test",
                install_command="pip install -r requirements.txt",
                run_command="python manage.py runserver",
            )
        
        # Check for Node.js projects
        if (root / "package.json").exists():
            # Determine if it's React, Vue, Angular, or plain Node
            framework = self._detect_node_framework()
            return ProjectStack(
                language="javascript",
                framework=framework,
                package_manager="npm",
                manifest_files=["package.json"],
                test_command="npm test",
                install_command="npm install",
                run_command="npm start" if framework == "node" else f"npm run {framework}",
            )
        
        # Check for Python projects
        if (root / "pyproject.toml").exists():
            return ProjectStack(
                language="python",
                package_manager="pip",
                manifest_files=["pyproject.toml"],
                test_command="pytest",
                install_command="pip install -e .",
                run_command="python -m <module>",
            )
        
        if (root / "requirements.txt").exists():
            return ProjectStack(
                language="python",
                package_manager="pip",
                manifest_files=["requirements.txt"],
                test_command="pytest" if (root / "pytest.ini").exists() else None,
                install_command="pip install -r requirements.txt",
                run_command=None,
            )
        
        # Check for PHP/Composer
        if (root / "composer.json").exists():
            return ProjectStack(
                language="php",
                package_manager="composer",
                manifest_files=["composer.json"],
                test_command="phpunit" if (root / "phpunit.xml").exists() else None,
                install_command="composer install",
                run_command=None,
            )
        
        # Check for Rust
        if (root / "Cargo.toml").exists():
            return ProjectStack(
                language="rust",
                package_manager="cargo",
                manifest_files=["Cargo.toml"],
                test_command="cargo test",
                install_command="cargo build",
                run_command="cargo run",
            )
        
        # Check for Go
        if (root / "go.mod").exists():
            return ProjectStack(
                language="go",
                package_manager="go",
                manifest_files=["go.mod"],
                test_command="go test ./...",
                install_command="go mod download",
                run_command="go run .",
            )
        
        # Check for Java/Maven
        if (root / "pom.xml").exists():
            return ProjectStack(
                language="java",
                package_manager="maven",
                manifest_files=["pom.xml"],
                test_command="mvn test",
                install_command="mvn install",
                run_command="mvn spring-boot:run",
            )
        
        if (root / "build.gradle").exists():
            return ProjectStack(
                language="java",
                package_manager="gradle",
                manifest_files=["build.gradle"],
                test_command="gradle test",
                install_command="gradle build",
                run_command="gradle bootRun",
            )
        
        return None
    
    def _is_wordpress(self) -> bool:
        """Check if this is a WordPress project."""
        root = self.workspace_root
        return (
            (root / "wp-config.php").exists() or
            (root / "wp-content").exists() or
            (root / "wp-admin").exists() or
            (root / "wp-includes").exists()
        )
    
    def _is_django(self) -> bool:
        """Check if this is a Django project."""
        root = self.workspace_root
        return (
            (root / "manage.py").exists() and
            any((root / f).exists() for f in ["settings.py", "settings", "urls.py"])
        )
    
    def _detect_node_framework(self) -> str:
        """Detect Node.js framework from package.json."""
        root = self.workspace_root
        package_json = root / "package.json"
        
        if not package_json.exists():
            return "node"
        
        import json
        try:
            with open(package_json, 'r') as f:
                data = json.load(f)
            
            dependencies = {**data.get("dependencies", {}), **data.get("devDependencies", {})}
            
            if "react" in dependencies or "react-dom" in dependencies:
                return "react"
            if "vue" in dependencies or "@vue/cli" in dependencies:
                return "vue"
            if "@angular/core" in dependencies or "angular" in dependencies:
                return "angular"
            if "next" in dependencies:
                return "next"
            if "@nestjs/core" in dependencies:
                return "nestjs"
            if "express" in dependencies:
                return "express"
            if "fastify" in dependencies:
                return "fastify"
            
            # Check scripts
            scripts = data.get("scripts", {})
            if "react-scripts" in str(scripts):
                return "react"
            if "vite" in str(scripts):
                return "vite"
            
        except:
            pass
        
        return "node"
    
    def get_install_command(self) -> Optional[str]:
        """Get the appropriate install command for this project."""
        stack = self.detect()
        return stack.install_command
    
    def get_test_command(self) -> Optional[str]:
        """Get the appropriate test command for this project."""
        stack = self.detect()
        return stack.test_command
    
    def get_package_manager(self) -> Optional[str]:
        """Get the appropriate package manager for this project."""
        stack = self.detect()
        return stack.package_manager
