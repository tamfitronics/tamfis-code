"""Build a wheel and publish a checksum manifest into the website public tree."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tamfis_code import __version__


def main():
    target = Path(sys.argv[1]).resolve()
    target.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="tamfis-code-release-") as directory:
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", directory, str(repo)], check=True)
        wheel, = Path(directory).glob("*.whl")
        shutil.copy2(wheel, target / wheel.name)
        manifest = {
            "version": __version__,
            "url": "https://gpt.tamfitronics.com/releases/tamfis-code/" + wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        }
        pending = target / "latest.json.tmp"
        pending.write_text(json.dumps(manifest, indent=2) + "\n")
        pending.replace(target / "latest.json")


if __name__ == "__main__":
    main()
