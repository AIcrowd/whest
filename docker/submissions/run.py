"""Adversarial tests — verifies lockdown blocks dangerous operations.

Defense layers:
1. Filesystem stripping — most dangerous modules deleted
2. open() restriction — can't read arbitrary files
3. Docker: network_mode=none, read_only, no capabilities, distroless (no shell)

Some modules (math, socket, subprocess) are available because pyzmq/stdlib
need them transitively, but they're harmless in the locked-down container.
"""

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"PASS: {name}")
        passed += 1
    else:
        print(f"FAIL: {name}")
        failed += 1

# Test 1: ctypes is stripped (filesystem layer)
try:
    import ctypes
    check("ctypes blocked", False)
except ImportError:
    check("ctypes blocked", True)

# Test 2: numpy is not available
try:
    import numpy
    check("numpy blocked", False)
except ImportError:
    check("numpy blocked", True)

# Test 3: open /etc/passwd blocked (open restriction layer)
try:
    open("/etc/passwd")
    check("open /etc/passwd blocked", False)
except PermissionError:
    check("open /etc/passwd blocked", True)

# Test 4: write to filesystem blocked (open restriction + read_only FS)
try:
    open("/tmp/test.txt", "w")
    check("write to /tmp blocked", False)
except (PermissionError, OSError):
    check("write to /tmp blocked", True)

# Test 5: can read own submission files
try:
    open("/submission/run.py", "r").read()
    check("can read /submission/", True)
except Exception:
    check("can read /submission/", False)

# Test 6: mechestim works
try:
    import mechestim as me
    check(f"mechestim loaded (v{me.__version__})", True)
except Exception:
    check("mechestim loaded", False)

# Test 7: allowed stdlib modules work
try:
    import itertools, functools, collections, time, json
    check("allowed stdlib modules work", True)
except Exception:
    check("allowed stdlib modules work", False)

# Test 8: socket exists but is useless (network_mode: none)
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("8.8.8.8", 53))
    check("network blocked (socket connect fails)", False)
except OSError:
    check("network blocked (socket connect fails)", True)

# Test 9: subprocess exists but is useless (no shell in distroless)
import subprocess
try:
    subprocess.run(["sh", "-c", "echo pwned"], capture_output=True, timeout=2)
    check("shell execution blocked", False)
except (FileNotFoundError, OSError):
    check("shell execution blocked", True)

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")
