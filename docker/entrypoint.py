#!/usr/bin/env python3
"""Hardened container entrypoint — waits for the flopscope server socket,
then runs the participant's submission.

Replaces entrypoint.sh so no shell is needed in the distroless image.
"""

import os
import sys
import time


def main() -> None:
    url = os.environ.get("FLOPSCOPE_SERVER_URL", "ipc:///tmp/ipc/flopscope.sock")
    sock_path = url.replace("ipc://", "")

    print("Waiting for flopscope server...", flush=True)
    for _ in range(30):
        if os.path.exists(sock_path):
            print("Server socket found. Running submission.", flush=True)
            import runpy

            sys.argv = ["/submission/run.py"]
            runpy.run_path("/submission/run.py", run_name="__main__")
            return
        time.sleep(0.2)

    print("ERROR: flopscope server socket not found after 6 seconds", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
