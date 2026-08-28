"""vision/extract_r.py — 安全模式数字抽取（docs/230/234 纪律）。

读取运行日志，用纯 python 正则抽取 R_* 摘要行（ASCII 标签 + 每行一个数字），
stdout 只回显这些行——绝不显示原始日志内容。编码自动识别（UTF-8 / UTF-16：
Windows PowerShell 的 *> 重定向会把子进程 stdout 写成 UTF-16LE，带 BOM）。

用法：
  python vision/extract_r.py <log1> [<log2> ...]
"""
import re
import sys

PAT = re.compile(r"^R_[A-Z0-9_]+=")


def read_text(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le").lstrip("\ufeff")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be").lstrip("\ufeff")
    return raw.decode("utf-8", errors="replace").lstrip("\ufeff")


def main():
    for path in sys.argv[1:]:
        for line in read_text(path).splitlines():
            if PAT.match(line):
                sys.stdout.write(line + "\n")


if __name__ == "__main__":
    main()
