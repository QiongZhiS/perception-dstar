"""vision/mt_extract.py - pure-python regex extraction of R_MT_* summary lines.

Safety discipline (repo rule): never print raw log content, only ASCII
label=value metadata lines matched by a regex. Usage:

  python vision/mt_extract.py <logfile> [<logfile> ...]
"""
import re
import sys

PAT = re.compile(r"^(R_MT_[A-Z0-9_.]+)=([0-9.\-eE]+|[A-Za-z0-9_]+)$")


def read_lines(path):
    """Read text robustly: PowerShell `*>` writes UTF-16LE; plain files are UTF-8."""
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8", "utf-16"):
        try:
            text = data.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
        if "\x00" not in text:
            return text.splitlines()
    return data.decode("utf-8", errors="replace").splitlines()


def main():
    for path in sys.argv[1:]:
        for line in read_lines(path):
            m = PAT.match(line.rstrip("\r\n"))
            if m:
                print(m.group(1) + "=" + m.group(2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
