"""Entry point for ``python -m mechestim_server``."""

from __future__ import annotations

import argparse
import sys

from mechestim_server._server import MechestimServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MechEstim budget-controlled compute server",
    )
    parser.add_argument(
        "--url",
        default="ipc:///tmp/mechestim.sock",
        help="ZMQ endpoint to bind (default: ipc:///tmp/mechestim.sock)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Session inactivity timeout in seconds (default: 60.0)",
    )
    args = parser.parse_args()

    print(
        f"[mechestim-server] binding to {args.url}  (timeout={args.timeout}s)",
        file=sys.stderr,
    )

    server = MechestimServer(url=args.url, session_timeout_s=args.timeout)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n[mechestim-server] shutting down", file=sys.stderr)
        server.stop()


if __name__ == "__main__":
    main()
