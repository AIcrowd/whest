"""Entry point for ``python -m flopscope_server``."""

from __future__ import annotations

import argparse
import sys

from flopscope_server._server import FlopscopeServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flopscope budget-controlled compute server",
    )
    parser.add_argument(
        "--url",
        default="ipc:///tmp/flopscope.sock",
        help="ZMQ endpoint to bind (default: ipc:///tmp/flopscope.sock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Session inactivity timeout in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    print(
        f"[flopscope-server] binding to {args.url}  (timeout={args.timeout}s)",
        file=sys.stderr,
    )

    server = FlopscopeServer(url=args.url, session_timeout_s=args.timeout)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[flopscope-server] shutting down", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
