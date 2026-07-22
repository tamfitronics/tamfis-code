# Tamfis-Code 0.4.44

Authoritative base: Tamfis-Code 0.4.43.

## Changes
- Coalesces fragmented provider deltas before Rich Markdown rendering.
- Avoids reparsing and repainting the entire answer for each character.
- Flushes on sentence/block boundaries, useful buffer sizes, and finalisation.
- Presents execution plans in a structured numbered panel.
- Includes assumptions and risks in plan output.
- Preserves standalone provider boundaries and existing safety/validation logic.

## Installation
Run `chmod +x install.sh && ./install.sh` from the extracted package.
