"""Smoke-load a serving endpoint (stdlib only: urllib + threads).

Usage:
    uvicorn app:app --port 8000 &
    python scripts/load_test.py --url http://localhost:8000 --n 50 --workers 5

Sends GET /health + POST /predict (titanic sample payload) and reports
mean/p50/p95 latency plus status-code counts. Exit 1 on any 5xx or
connection failure — suitable as a post-deploy gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAYLOAD = {
    "pclass": 1,
    "sex": "female",
    "age": 29,
    "sibsp": 0,
    "fare": 100,
    "embarked": "S",
}


def _call(url: str, payload: dict[str, object], timeout: int) -> tuple[int, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/predict", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status, time.perf_counter() - start
    except Exception as e:  # noqa: BLE001 — failures are the report
        code = getattr(e, "code", 0) or 0
        return int(code), time.perf_counter() - start


def run_load_test(url: str, n: int = 50, workers: int = 5, timeout: int = 10) -> dict[str, object]:
    """Fire ``n`` POSTs across ``workers`` threads. Returns the report dict."""
    if n <= 0 or workers <= 0:
        raise ValueError("n and workers must be > 0")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda _: _call(url, PAYLOAD, timeout), range(n)))
    codes = [c for c, _ in results]
    lat = sorted(t for _, t in results)
    failures = sum(1 for c in codes if c >= 500 or c == 0)
    return {
        "n": n,
        "status_counts": {str(c): codes.count(c) for c in sorted(set(codes))},
        "failures": failures,
        "mean_s": round(statistics.mean(lat), 4) if lat else 0.0,
        "p50_s": round(statistics.median(lat), 4) if lat else 0.0,
        "p95_s": round(lat[min(int(0.95 * len(lat)), len(lat) - 1)], 4) if lat else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-load a serving endpoint")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=10)
    args = parser.parse_args(argv)

    # Fail fast when the server is down before burning the load.
    try:
        with urllib.request.urlopen(f"{args.url}/health", timeout=args.timeout) as resp:
            if resp.status != 200:
                print(f"health check failed: HTTP {resp.status}")
                return 1
    except Exception as e:  # noqa: BLE001
        print(f"server unreachable at {args.url}: {e}")
        return 1

    report = run_load_test(args.url, args.n, args.workers, args.timeout)
    print(json.dumps(report, indent=2))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
