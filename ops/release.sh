#!/bin/sh
# Publish the CURRENT committed build so installed copies are alerted and can `tamfis-code update`.
#
#   ops/release.sh [notes.md]
#
# Order for every release:  bump version -> tests -> pip install -> commit -> push -> THIS SCRIPT.
# Skipping it is silent and total: installed apps only learn about a release from latest.json.
set -eu
repo=$(cd "$(dirname "$0")/.." && pwd)
public=${TAMFIS_RELEASE_PUBLIC:-/home/tamfisgpt/tamfis-frontend/public/releases/tamfis-code}  # survives `npm run build`
live=${TAMFIS_RELEASE_LIVE:-/home/tamfisgpt/tamfis-frontend/dist/releases/tamfis-code}          # what Caddy serves now
url=${TAMFIS_RELEASE_URL:-https://gpt.tamfitronics.com/releases/tamfis-code/latest.json}
notes=${1:-}
cd "$repo"
if [ -n "$(git status --porcelain -- tamfis_code pyproject.toml setup.py setup.cfg)" ]; then
  echo "Uncommitted changes in the package: commit first so the published wheel matches a commit." >&2
  exit 1
fi
export PYTHONPATH="$repo"
version=$(python3 -c 'import tamfis_code; print(tamfis_code.__version__)')
if [ -z "$notes" ] && [ -f "$repo/RELEASE_NOTES_${version}.md" ]; then
  notes="$repo/RELEASE_NOTES_${version}.md"
fi
if [ -n "$notes" ]; then
  python3 ops/publish_release.py "$public" --notes "$notes"
else
  python3 ops/publish_release.py "$public"
fi
mkdir -p "$live"
for metadata in latest.json release-notes.md install.sh; do
  test -f "$public/$metadata"
  cp -p "$public/$metadata" "$live/$metadata"
done
wheel="$public/tamfis_code-${version}-py3-none-any.whl"
test -f "$wheel"
cp -p "$wheel" "$live/"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["version"])' "$public/latest.json")" = "$version"
served=$(curl -fsS -m 15 "$url" | python3 -c 'import sys, json; print(json.load(sys.stdin)["version"])')
if [ "$served" != "$version" ]; then
  echo "MISMATCH: built $version but $url serves $served" >&2
  exit 1
fi
echo "OK: $url now serves $version -- installed copies will be alerted."
