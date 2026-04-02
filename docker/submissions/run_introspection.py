"""Introspection attack tests — verifies __subclasses__ escape doesn't work."""

# Try the classic __subclasses__ escape
try:
    for cls in ().__class__.__bases__[0].__subclasses__():
        name = cls.__name__
        if name in ("Popen", "socket", "FileIO"):
            print(f"FAIL: found {name} via __subclasses__")
            break
    else:
        print("PASS: no dangerous classes found via __subclasses__")
except Exception as exc:
    print(f"PASS: introspection blocked: {exc}")

# Try to reimport builtins and undo lockdown
try:
    import builtins
    builtins.eval = lambda x: None  # try to restore eval
    eval("1+1")
    print("FAIL: eval restored")
except (RuntimeError, TypeError) as exc:
    print(f"INFO: eval restore attempt result: {exc}")
