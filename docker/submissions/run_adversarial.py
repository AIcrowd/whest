"""Adversarial tests — verifies lockdown blocks dangerous operations."""

# Test 1: import math
try:
    import math
    print("FAIL: math imported")
except ImportError as exc:
    print(f"PASS: {exc}")

# Test 2: import socket
try:
    import socket
    print("FAIL: socket imported")
except ImportError as exc:
    print(f"PASS: {exc}")

# Test 3: import subprocess
try:
    import subprocess
    print("FAIL: subprocess imported")
except ImportError as exc:
    print(f"PASS: {exc}")

# Test 4: import ctypes
try:
    import ctypes
    print("FAIL: ctypes imported")
except ImportError as exc:
    print(f"PASS: {exc}")

# Test 5: exec
try:
    exec("x = 1")
    print("FAIL: exec worked")
except RuntimeError as exc:
    print(f"PASS: {exc}")

# Test 6: eval
try:
    eval("1+1")
    print("FAIL: eval worked")
except RuntimeError as exc:
    print(f"PASS: {exc}")

# Test 7: open /etc/passwd
try:
    open("/etc/passwd")
    print("FAIL: open /etc/passwd worked")
except PermissionError as exc:
    print(f"PASS: {exc}")

# Test 8: write to filesystem
try:
    open("/tmp/test.txt", "w")
    print("FAIL: write worked")
except (PermissionError, OSError) as exc:
    print(f"PASS: {exc}")

# Test 9: mechestim still works
import mechestim as me
print(f"PASS: mechestim loaded, version={me.__version__}")

# Test 10: allowed modules work
import itertools
import functools
import collections
import time
import json
print("PASS: all allowed modules imported")

print("\nAll adversarial tests complete.")
