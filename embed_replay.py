"""Embed replay JSON files into index.html as inline data variables.

Usage:
    # Embed both datasets at once (recommended)
    python embed_replay.py \
        --replay_3b  replay_logs/replay_trained_3b.json \
        --replay_17b replay_logs/replay_replay.json \
        --out index_embedded.html

    # Embed only one (the other stays null)
    python embed_replay.py --replay_3b replay_logs/replay_trained_3b.json
"""

import argparse
import json
import sys
from pathlib import Path


PLACEHOLDER_3B  = "/* EMBED_REPLAY_3B  */ null"
PLACEHOLDER_17B = "/* EMBED_REPLAY_17B */ null"


def embed(html: str, replay_path: Path | None, placeholder: str, label: str) -> tuple[str, int]:
    if replay_path is None:
        return html, 0
    if not replay_path.exists():
        print(f"[embed] {label} replay not found: {replay_path}")
        sys.exit(1)
    data = json.loads(replay_path.read_text())
    if placeholder not in html:
        print(f"[embed] Placeholder '{placeholder}' not found in HTML — skipping {label}.")
        return html, 0
    steps = len(data.get("steps", []))
    return html.replace(placeholder, json.dumps(data, separators=(",", ":"))), steps


def main():
    p = argparse.ArgumentParser(description="Embed replay JSON(s) into index.html")
    p.add_argument("--replay_3b",  default=None, help="3B LoRA ablation replay JSON")
    p.add_argument("--replay_17b", default=None, help="Maverick 17B replay JSON")
    p.add_argument("--html", default="index.html", help="Source HTML template")
    p.add_argument("--out", default=None, help="Output HTML path (default: overwrites --html)")
    args = p.parse_args()

    if not args.replay_3b and not args.replay_17b:
        p.error("Provide at least one of --replay_3b or --replay_17b")

    html_path = Path(args.html)
    out_path  = Path(args.out) if args.out else html_path

    if not html_path.exists():
        print(f"[embed] HTML template not found: {html_path}")
        sys.exit(1)

    html = html_path.read_text(encoding="utf-8")

    html, s3b  = embed(html, Path(args.replay_3b)  if args.replay_3b  else None, PLACEHOLDER_3B,  "3B")
    html, s17b = embed(html, Path(args.replay_17b) if args.replay_17b else None, PLACEHOLDER_17B, "17B")

    out_path.write_text(html, encoding="utf-8")
    if s3b:  print(f"[embed] 3B  replay ({s3b}  steps) → {out_path}")
    if s17b: print(f"[embed] 17B replay ({s17b} steps) → {out_path}")


if __name__ == "__main__":
    main()
