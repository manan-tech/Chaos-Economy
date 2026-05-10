"""Embed a replay JSON file into index.html as an inline REPLAY_DATA variable.

Usage:
    python embed_replay.py replay_logs/replay_test10.json
    python embed_replay.py replay_logs/replay_demo.json --out my_replay.html
"""

import argparse
import json
import sys
from pathlib import Path


def embed(replay_path: Path, html_path: Path, out_path: Path) -> None:
    data = json.loads(replay_path.read_text())
    html = html_path.read_text(encoding="utf-8")

    placeholder = "/* EMBED_REPLAY_JSON */ null"
    if placeholder not in html:
        print(f"[embed] Placeholder '{placeholder}' not found in {html_path}.")
        sys.exit(1)

    injected = html.replace(placeholder, json.dumps(data, separators=(",", ":")))
    out_path.write_text(injected, encoding="utf-8")
    steps = len(data.get("steps", []))
    print(f"[embed] {replay_path.name} ({steps} steps) → {out_path}")


def main():
    p = argparse.ArgumentParser(description="Embed replay JSON into index.html")
    p.add_argument("replay", help="Path to replay JSON file")
    p.add_argument("--html", default="index.html", help="Source HTML template")
    p.add_argument("--out", default=None, help="Output HTML path (default: overwrites --html)")
    args = p.parse_args()

    replay_path = Path(args.replay)
    html_path = Path(args.html)
    out_path = Path(args.out) if args.out else html_path

    if not replay_path.exists():
        print(f"[embed] Replay file not found: {replay_path}")
        sys.exit(1)
    if not html_path.exists():
        print(f"[embed] HTML template not found: {html_path}")
        sys.exit(1)

    embed(replay_path, html_path, out_path)


if __name__ == "__main__":
    main()
