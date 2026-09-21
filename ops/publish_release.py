"""Build a wheel and publish a checksum manifest (and release notes) for `tamfis-code update`.

    python3 ops/publish_release.py <target-dir> [--notes FILE]

Installed copies find out about a new version by fetching ``<target>/latest.json``; if that file is
not refreshed with every release, nobody is ever told (it sat at 1.6.21 while the build was 1.6.78).
Use ``ops/release.sh`` -- it runs this for both the frontend source tree and the live ``dist/`` and
verifies the URL users actually hit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from tamfis_code import __version__


def _prepend_release_notes(target: Path, notes: str) -> None:
    notes_path = target / "release-notes.md"
    previous = notes_path.read_text(encoding="utf-8") if notes_path.exists() else ""
    if previous.startswith(f"# Tamfis Code {__version__}"):
        return  # this version's notes are already there (re-publishing the same build)
    # the previous top entry "# Tamfis Code X" becomes an ordinary "## X" section
    previous = re.sub(r"\A# Tamfis Code (\S+)", r"## \1", previous)
    body = notes.strip() or "- Maintenance release."
    notes_path.write_text(f"# Tamfis Code {__version__}\n\n{body}\n\n{previous}".rstrip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--notes", help="file with the markdown bullet list for this version")
    args = parser.parse_args()
    target = Path(args.target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]

    published = None
    try:
        published = json.loads((target / "latest.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        pass
    key = lambda v: tuple(int(x) for x in v.split("."))
    if published and key(__version__) < key(published):
        raise SystemExit(f"Refusing to publish {__version__}: {published} is already published there.")

    with tempfile.TemporaryDirectory(prefix="tamfis-code-release-") as directory:
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", directory, str(repo)], check=True)
        wheel, = Path(directory).glob("*.whl")
        shutil.copy2(wheel, target / wheel.name)
        manifest = {
            "version": __version__,
            "url": "https://gpt.tamfitronics.com/releases/tamfis-code/" + wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
        if args.notes:
            _prepend_release_notes(target, Path(args.notes).read_text(encoding="utf-8"))
        pending = target / "latest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(target / "latest.json")
    print(f"published {__version__} -> {target}")


if __name__ == "__main__":
    main()
