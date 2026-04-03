"""Introspection attack tests — verifies sandbox limits.

Tests that even with Python introspection tricks, an attacker can't
reach anything useful in the locked-down container.
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


# Test 1: __subclasses__ escape — can we find dangerous classes?
dangerous_found = []
for cls in ().__class__.__bases__[0].__subclasses__():
    name = cls.__name__
    if name in ("Popen", "FileIO"):
        dangerous_found.append(name)
check("no Popen/FileIO via __subclasses__", len(dangerous_found) == 0)

# Test 2: even if we find os module via introspection, network is blocked
import os

try:
    os.listdir("/submission")
    check("os.listdir /submission works (expected)", True)
except Exception:
    check("os.listdir /submission works (expected)", False)

# Test 3: os.system should fail (no shell in image)
ret = os.system("echo pwned")
check("os.system fails (no shell)", ret != 0)

# Test 4: exec/eval exist but are useless without dangerous modules
# (eval/exec can't be disabled — stdlib needs them internally)
try:
    result = eval("__import__('ctypes')")
    check("eval+import ctypes blocked", False)
except ImportError:
    check("eval+import ctypes blocked", True)

# Test 5: can't read sensitive files even via eval
try:
    eval("open('/etc/shadow').read()")
    check("eval+open /etc/shadow blocked", False)
except PermissionError:
    check("eval+open /etc/shadow blocked", True)

# Test 6: can't spawn processes even via exec
try:
    exec("import subprocess; subprocess.run(['id'])")
    check("exec+subprocess fails (no shell)", False)
except FileNotFoundError:
    check("exec+subprocess fails (no shell)", True)

print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")
