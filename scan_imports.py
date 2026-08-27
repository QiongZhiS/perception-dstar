"""scan_imports.py - list unique top-level import module names under vision/*.py.

Uses only ast: parses each file's syntax tree and collects module names from
top-level Import / ImportFrom nodes. It never prints source code or file
contents - only the unique top-level module names (one per line, sorted).

Usage:
  python scan_imports.py [dir]     # default dir = vision
"""
import ast
import os
import sys


def top_level_modules(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    mods = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # absolute import; relative (from . import x) skipped
                mods.add(node.module.split(".")[0])
    return mods


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "vision"
    all_mods = set()
    for name in sorted(os.listdir(target)):
        if name.endswith(".py"):
            all_mods |= top_level_modules(os.path.join(target, name))
    for mod in sorted(all_mods):
        print(mod)


if __name__ == "__main__":
    main()
