"""Canonical allowlist of Python stdlib modules for the hardened participant image.

Used by:
- strip_stdlib.py (build time): keep only these modules on the filesystem
- sitecustomize.py (runtime): gate __import__ to only allow these modules

Participant packages (mechestim, pyzmq, msgpack) are always allowed — they
live in site-packages, not in the stdlib directory.
"""

# fmt: off

# Modules that participants may use directly
PARTICIPANT_MODULES = frozenset({
    "itertools",
    "functools",
    "collections",
    "time",
    "json",
    "contextlib",
})

# Modules needed by Python internals, mechestim-client, pyzmq, or msgpack
INFRASTRUCTURE_MODULES = frozenset({
    # Python startup / core
    "sys",
    "builtins",
    "_thread",
    "threading",
    "abc",
    "types",
    "typing",
    "enum",
    "io",
    "_io",
    "os",
    "os.path",
    "posixpath",
    "stat",
    "errno",
    "struct",
    "warnings",
    "traceback",
    "linecache",
    "token",
    "tokenize",
    "encodings",
    "codecs",
    "copyreg",
    "operator",
    "keyword",
    "reprlib",
    "weakref",

    # import system (needed for initial imports, then gated)
    "importlib",
    "importlib._bootstrap",
    "importlib._bootstrap_external",
    "importlib.abc",
    "importlib.machinery",
    "importlib.util",
    "importlib.metadata",
    "importlib.resources",

    # runpy (entrypoint uses it)
    "runpy",
    "pkgutil",

    # textwrap, string, re — used by various stdlib internals
    "re",
    "_sre",
    "sre_compile",
    "sre_constants",
    "sre_parse",
    "string",
    "textwrap",

    # dataclasses — used by some internal modules
    "dataclasses",

    # needed by typing internals
    "typing_extensions",

    # select — needed by pyzmq
    "select",
    "selectors",

    # platform detection
    "_sysconfigdata__linux_x86_64-linux-gnu",
    "platform",
})

# All allowed top-level module names (union of above)
ALLOWED_MODULES = PARTICIPANT_MODULES | INFRASTRUCTURE_MODULES

# Modules to explicitly poison (set to None in sys.modules)
# This prevents import even if the .py file was somehow missed during stripping.
POISONED_MODULES = frozenset({
    "math",
    "cmath",
    "numpy",
    "subprocess",
    "socket",
    "http",
    "http.client",
    "http.server",
    "urllib",
    "urllib.request",
    "urllib.parse",
    "ctypes",
    "ctypes.util",
    "cffi",
    "pickle",
    "shelve",
    "multiprocessing",
    "concurrent",
    "concurrent.futures",
    "asyncio",
    "ssl",
    "sqlite3",
    "xml",
    "email",
    "ftplib",
    "telnetlib",
    "tkinter",
    "code",
    "codeop",
    "compileall",
    "py_compile",
    "zipimport",
    "zipfile",
    "tarfile",
    "shutil",
    "tempfile",
    "glob",
    "fnmatch",
    "pathlib",
    "signal",
    "mmap",
    "resource",
    "pty",
    "termios",
    "tty",
    "curses",
    "readline",
    "rlcompleter",
    "cmd",
    "pdb",
    "profile",
    "cProfile",
    "dis",
    "inspect",
    "ast",
})

# fmt: on
