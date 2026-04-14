"""Entry point for ``python -m whest_server``."""

from __future__ import annotations

import argparse
import sys

from whest_server._server import WhestServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whest budget-controlled compute server",
    )
    parser.add_argument(
        "--url",
        default="ipc:///tmp/whest.sock",
        help="ZMQ endpoint to bind (default: ipc:///tmp/whest.sock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Session inactivity timeout in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    print(
        f"[whest-server] binding to {args.url}  (timeout={args.timeout}s)",
        file=sys.stderr,
    )

    server = WhestServer(url=args.url, session_timeout_s=args.timeout)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[whest-server] shutting down", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
