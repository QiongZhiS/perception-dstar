"""vision/log_diag.py — 运行日志诊断辅助（只打印异常/错误相关行，不打印日志内容）。

安全纪律：stdout 只输出异常行（Traceback/Error/line 片段），用于技术故障排查；
不是实验输出抽取（实验数字用 vision/extract_r.py）。
用法：python vision/log_diag.py logs/<name>.log
"""
import re
import sys

PAT = re.compile(r"Error|Traceback|line \d+")


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
            if PAT.search(line):
                sys.stdout.write(line.rstrip("\r") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
