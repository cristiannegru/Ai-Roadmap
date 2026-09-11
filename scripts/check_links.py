"""Link checker — offline-safe.

Default `--offline` mode only parses markdown and reports counts
(no network), so CI stays fast and deterministic.
Pass `--online` to actually HEAD each http(s) URL (slow, for weekly cron).

Usage:
    python scripts/check_links.py --offline
    python scripts/check_links.py --online --timeout 10
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[.*?\]\((https?://[^)]+)\)")
BARE_RE = re.compile(r"(?<!\()(https?://[^\s)>\]]+)")


def collect_md_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    return [f for f in files if f.exists()]


def extract_urls(text: str) -> list[str]:
    urls = LINK_RE.findall(text)
    urls += BARE_RE.findall(text)
    # dedupe, strip trailing punctuation from bare matches
    return sorted({u.rstrip(".,;\"'") for u in urls})


def check_online(url: str, timeout: int) -> tuple[bool, str]:
    try:
        req = Request(url, method="HEAD", headers={"User-Agent": "ai-roadmap-linkcheck/1.1"})
        with urlopen(req, timeout=timeout) as resp:
            return (resp.status < 400, str(resp.status))
    except Exception as e:  # noqa: BLE001 — report any failure as broken in online mode
        return (False, f"{type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", default=True)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args(argv)
    online = args.online

    total = 0
    broken: list[tuple[str, str, str]] = []
    for md in collect_md_files():
        urls = extract_urls(md.read_text(encoding="utf-8", errors="ignore"))
        total += len(urls)
        print(f"{md.relative_to(ROOT)}: {len(urls)} urls")
        if online:
            for u in urls:
                ok, info = check_online(u, args.timeout)
                if not ok:
                    broken.append((str(md.relative_to(ROOT)), u, info))

    print(f"checked {total} urls across {len(collect_md_files())} files (online={online})")
    if broken:
        print("BROKEN:")
        for f, u, info in broken:
            print(f"  ✗ {f}: {u} -> {info}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
